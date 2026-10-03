"""
V2N Handover Environment
=========================
Reproduces the MDP formulation of:

    Sohaib, R.M., Onireti, O., Tan, K., Sambo, Y., Swash, R., Imran, M.
    "Meta-Transfer Learning-Based Handover Optimization for V2N Communication"
    IEEE Trans. Veh. Technol., 2024.  (base DRL component only: DDQN + Thompson
    Sampling, Algorithm 1 -- this is the "base paper" you asked to reproduce)

WHAT IS EXACT (matches the paper's equations):
  - State space  s_l = {RSRP_l ; one-hot(serving_BS_id)}         -> Eq. (11)-(12)
  - Action space a_t = {BS_0, ..., BS_k}                          -> Eq. (13)
  - Reward       r(s_l, a, s_l+1)                                 -> Eq. (14)
  - Handover cost / normalized delay ζc, throughput Ω             -> Eq. (2)-(4)
  - Scenario configs (BS count, VUE speed, sim time) from Table I

WHAT IS A STAND-IN (the paper generates all of this from ns-3's LENA LTE
module + SUMO-generated trajectories; reproducing that full stack is a
separate systems-engineering task, not an RL-algorithm task):
  - Path-loss / shadowing / fast-fading model -> replaced with a simplified
    3GPP-style log-distance + log-normal shadowing + Rayleigh-like fast
    fading model, parameterised per scenario.
  - VUE mobility -> replaced with a simple synthetic trajectory generator
    (straight line for highway, Manhattan-grid-like turns for urban/rural)
    instead of an actual SUMO-exported trace.
  - SINR/interference -> approximated from serving-cell RSRP and noise only
    (the paper's ns-3 simulation has full downlink scheduling and inter-cell
    interference from the LENA module).

Swap `VUE.step()` and `Environment._compute_rsrp()` for real ns-3+SUMO trace
readers once you have that pipeline running, without touching agent.py.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Scenario configs (values taken from Table I of the paper where given;
# BS layout / path-loss exponents are reasonable stand-ins for the 3GPP
# UMa / RMa / highway models the paper references, see module docstring).
# ---------------------------------------------------------------------------
SCENARIOS = {
    "urban": dict(
        num_bs=15,              # 5 sites x 3 cells/site (Table I)
        area_size=(1299.0, 750.0),   # metres, from the paper's Manhattan grid
        vue_speed_kmh=50.0,
        sim_time_s=800,
        bs_height=25.0,
        pathloss_exponent=3.6,       # NLOS urban grid (3GPP UMa-like)
        shadow_sigma_db=8.0,
        fast_fade_sigma_db=3.0,
    ),
    "rural": dict(
        num_bs=9,               # 3 sites x 3 cells/site (Table I)
        area_size=(3000.0, 3000.0),
        vue_speed_kmh=108.0,
        sim_time_s=1000,
        bs_height=35.0,
        pathloss_exponent=2.7,       # 3GPP RMa-like
        shadow_sigma_db=6.0,
        fast_fade_sigma_db=2.5,
    ),
    "highway": dict(
        num_bs=9,
        area_size=(1500.0, 30.0),    # 1500 m segment, 6 lanes x 5 m
        vue_speed_kmh=120.0,
        sim_time_s=600,
        bs_height=35.0,
        pathloss_exponent=2.2,       # near-LOS along a highway
        shadow_sigma_db=4.0,
        fast_fade_sigma_db=2.0,
    ),
}

MAX_BS = max(cfg["num_bs"] for cfg in SCENARIOS.values())  # fixed obs/action size
# so a network trained on one scenario (e.g. urban, 15 BSs) can be evaluated
# on another (e.g. highway, 9 BSs) without a shape mismatch -- padded slots
# get a sentinel "no signal" RSRP and are masked out of the action space.

TX_POWER_DBM = 40.0        # Table I: BS transmit power
NOISE_FIGURE_DB = 5.0      # UE noise figure, Table I
CARRIER_FREQ_MHZ = 2115.0
BANDWIDTH_HZ = 2 * 10e6    # 2x10 MHz, Table I
THERMAL_NOISE_DBM = -174.0 + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
RSRP_SAMPLE_INTERVAL_S = 0.2   # "RSRP: every 200 ms", Table I


def _place_base_stations(num_bs: int, area_size, rng: np.random.Generator):
    """Randomly place BSs while keeping reasonable inter-site spacing,
    mirroring the paper's 'randomly set their locations ... with predefined
    rules to guarantee overall coverage' (Sec. VI)."""
    w, h = area_size
    positions = []
    attempts = 0
    while len(positions) < num_bs and attempts < 5000:
        p = rng.uniform([0, 0], [w, h])
        if all(np.linalg.norm(p - q) > 0.35 * min(w, h) / np.sqrt(num_bs) for q in positions):
            positions.append(p)
        attempts += 1
    while len(positions) < num_bs:  # fallback if spacing constraint too strict
        positions.append(rng.uniform([0, 0], [w, h]))
    return np.array(positions)


class VUETrajectory:
    """Synthetic stand-in for a SUMO-exported vehicle trajectory."""

    def __init__(self, area_size, speed_kmh, scenario: str, rng: np.random.Generator):
        self.area_size = np.array(area_size)
        self.speed = speed_kmh / 3.6  # m/s
        self.rng = rng
        self.scenario = scenario
        self.pos = np.array([0.0, area_size[1] / 2.0]) if scenario == "highway" \
            else rng.uniform([0, 0], area_size)
        self.heading = 0.0  # radians

    def step(self, dt: float):
        if self.scenario == "highway":
            # straight line, wrap-around at the end (matches paper's highway setup)
            self.pos[0] += self.speed * dt
            if self.pos[0] > self.area_size[0]:
                self.pos[0] = 0.0
        else:
            # Manhattan-grid-like motion: mostly straight, occasional turn
            if self.rng.random() < 0.02:  # ~ turning probability per step
                self.heading += self.rng.choice([-1, 1]) * np.pi / 2
            dx = self.speed * dt * np.cos(self.heading)
            dy = self.speed * dt * np.sin(self.heading)
            self.pos = np.clip(self.pos + [dx, dy], [0, 0], self.area_size)
        return self.pos.copy()


class V2NHandoverEnv:
    """
    Gym-like environment implementing Eqs. (1)-(14) of the base paper.

    Observation: np.ndarray of shape (num_bs * 2,)  -> [normalized RSRP_1..k, one-hot(serving)]
    Action:      integer in [0, num_bs)              -> target BS to connect to
    """

    def __init__(self, scenario: str = "urban", seed: int = 0):
        assert scenario in SCENARIOS, f"unknown scenario {scenario}"
        self.scenario = scenario
        self.cfg = SCENARIOS[scenario]
        self.rng = np.random.default_rng(seed)
        self.num_bs = self.cfg["num_bs"]
        self.bs_pos = _place_base_stations(self.num_bs, self.cfg["area_size"], self.rng)
        self.dt = RSRP_SAMPLE_INTERVAL_S
        self.max_steps = int(self.cfg["sim_time_s"] / self.dt)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.vue = VUETrajectory(self.cfg["area_size"], self.cfg["vue_speed_kmh"],
                                  self.scenario, self.rng)
        self.t = 0
        self.ho_events_in_window = 0
        self.time_spent_in_ho = 0.0
        self.ho_delay_s = 0.02  # a single hard-HO delay, order-of-ms per paper's discussion
        rsrp = self._compute_rsrp(self.vue.pos)
        self.serving_bs = int(np.argmax(rsrp))
        self._last_rsrp = rsrp
        return self._make_state(rsrp, self.serving_bs)

    # ------------------------------------------------------------------
    def _compute_rsrp(self, pos):
        """Log-distance path loss + shadow fading + fast fading -> RSRP per BS (dBm)."""
        d = np.linalg.norm(self.bs_pos - pos[None, :], axis=1)
        d = np.maximum(d, 1.0)  # avoid log(0)
        pl0 = 32.4 + 20 * np.log10(CARRIER_FREQ_MHZ / 1000.0)  # free-space ref at 1 m (simplified)
        pathloss = pl0 + 10 * self.cfg["pathloss_exponent"] * np.log10(d)
        shadow = self.rng.normal(0, self.cfg["shadow_sigma_db"], size=self.num_bs)
        fast_fade = self.rng.normal(0, self.cfg["fast_fade_sigma_db"], size=self.num_bs)
        rsrp = TX_POWER_DBM - pathloss + shadow + fast_fade
        return rsrp

    def _make_state(self, rsrp, serving_bs):
        rsrp_norm = (rsrp - rsrp.min()) / (rsrp.max() - rsrp.min() + 1e-9)
        one_hot = np.zeros(self.num_bs)
        one_hot[serving_bs] = 1.0
        # pad to MAX_BS so state/action dims are identical across scenarios
        # (needed to transfer a trained network across scenarios, Sec. VI-A/B)
        rsrp_pad = np.zeros(MAX_BS)
        onehot_pad = np.zeros(MAX_BS)
        rsrp_pad[: self.num_bs] = rsrp_norm
        onehot_pad[: self.num_bs] = one_hot
        return np.concatenate([rsrp_pad, onehot_pad]).astype(np.float32)

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask of length MAX_BS: True for BS indices that exist in
        this scenario. Pass to agent.select_action() so it never picks a
        padded, non-existent BS when evaluating a network trained on a
        scenario with a different BS count."""
        mask = np.zeros(MAX_BS, dtype=bool)
        mask[: self.num_bs] = True
        return mask

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        action: index of the BS the agent chooses to be connected to this step.
        Implements Eq. (14):
            r = B(1-ζc)log2(1+SINR)   if a handover happens
            r = B log2(1+SINR)        otherwise
        """
        assert 0 <= action < self.num_bs
        ho_happened = action != self.serving_bs

        # advance VUE, recompute channel
        pos = self.vue.step(self.dt)
        rsrp = self._compute_rsrp(pos)
        self.serving_bs = action

        # SINR proxy from serving-cell RSRP vs. thermal noise (see module docstring)
        sinr_db = rsrp[self.serving_bs] - THERMAL_NOISE_DBM
        sinr_lin = 10 ** (sinr_db / 10.0)

        # handover delay cost, Eq. (2)-(3)
        if ho_happened:
            self.time_spent_in_ho += self.ho_delay_s
            self.ho_events_in_window += 1
        elapsed = (self.t + 1) * self.dt
        zeta_c = min(self.time_spent_in_ho / max(elapsed, 1e-9), 1.0)

        if ho_happened:
            reward = BANDWIDTH_HZ * (1 - zeta_c) * np.log2(1 + sinr_lin)
        else:
            reward = BANDWIDTH_HZ * np.log2(1 + sinr_lin)
        reward = reward / BANDWIDTH_HZ  # normalize so rewards are O(1) for stable training

        # ping-pong: handover back to the BS we were connected to 2 steps ago
        ping_pong = ho_happened and hasattr(self, "_prev_serving") and action == self._prev_serving
        self._prev_serving = self.serving_bs if not ho_happened else getattr(self, "_last_serving_before_ho", self.serving_bs)
        self._last_serving_before_ho = self.serving_bs

        self.t += 1
        done = self.t >= self.max_steps
        next_state = self._make_state(rsrp, self.serving_bs)
        info = dict(handover=ho_happened, ping_pong=bool(ping_pong),
                    sinr_db=sinr_db, zeta_c=zeta_c)
        return next_state, float(reward), done, info

    @property
    def state_dim(self):
        return MAX_BS * 2

    @property
    def action_dim(self):
        return MAX_BS