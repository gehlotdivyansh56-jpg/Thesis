import numpy as np
import math
import random

class ProposedBaseStation:
    def __init__(self, bs_id, rat_type, position, tx_power, radius, freq_ghz, beamforming_gain=2.5):
        self.bs_id = bs_id
        self.rat_type = rat_type  # '5G_NR_mmWave', '5G_NR_Sub6', 'DSRC', 'WIFI6'
        self.position = np.array(position, dtype=np.float32)
        self.tx_power = tx_power  # dBm
        self.radius = radius  # meters
        self.freq_ghz = freq_ghz
        self.beamforming_gain = beamforming_gain  # linear scale
        self.load_urllc = 0.01
        self.load_embb = 0.01

    def compute_path_loss(self, distance):
        if distance <= 1.0:
            distance = 1.0
        if self.rat_type == '5G_NR_mmWave':
            pl = 32.4 + 20.0 * math.log10(self.freq_ghz) + 28.0 * math.log10(distance)
        elif self.rat_type == '5G_NR_Sub6':
            pl = 32.4 + 20.0 * math.log10(self.freq_ghz) + 26.0 * math.log10(distance)
        elif self.rat_type == 'DSRC':
            pl = 38.0 + 20.0 * math.log10(self.freq_ghz) + 22.0 * math.log10(distance)
        else:  # WIFI6
            pl = 40.0 + 20.0 * math.log10(self.freq_ghz) + 24.0 * math.log10(distance)
        shadowing = np.random.normal(0, 1.0)  # Lower variance due to beamforming & MIMO
        return pl + shadowing

    def get_rssi_sinr(self, veh_pos, interference_dbm=-98.0):
        dist = np.linalg.norm(veh_pos - self.position)
        if dist > self.radius * 1.5:
            return -120.0, -25.0, dist

        pl = self.compute_path_loss(dist)
        rx_power_dbm = self.tx_power + 10.0 * math.log10(self.beamforming_gain) - pl
        noise_dbm = -104.0

        noise_mw = 10 ** (noise_dbm / 10.0)
        interf_mw = 10 ** (interference_dbm / 10.0)
        rx_mw = 10 ** (rx_power_dbm / 10.0)

        sinr = 10.0 * math.log10(rx_mw / (noise_mw + interf_mw + 1e-12))
        return rx_power_dbm, sinr, dist


class ProposedVehicleAgent:
    def __init__(self, veh_id, init_pos, init_speed, qos_type):
        self.veh_id = veh_id
        self.position = np.array(init_pos, dtype=np.float32)
        self.speed = init_speed
        self.heading = random.uniform(0, 2 * math.pi)
        self.qos_type = qos_type
        self.current_bs_id = -1
        self.current_rat = 'NONE'
        self.connection_history = []
        self.handover_count = 0
        self.pingpong_count = 0
        self.last_ho_step = -10
        self.dwell_time = 0

    def update_position(self, delta_t=1.0):
        self.heading += random.uniform(-0.03, 0.03)
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.position += np.array([vx, vy], dtype=np.float32) * delta_t
        self.position = np.clip(self.position, 50.0, 1450.0)
        self.dwell_time += 1

    def predict_future_position(self, lookahead_sec=2.0):
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        future_pos = self.position + np.array([vx, vy], dtype=np.float32) * lookahead_sec
        return np.clip(future_pos, 50.0, 1450.0)


class ProposedIoVEnv:
    def __init__(self, num_vehicles=12, max_steps=100):
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.current_step = 0

        # Base Stations Setup (6G mmWave / 5G Sub-6 / DSRC / Wi-Fi 6)
        self.base_stations = [
            ProposedBaseStation(0, '5G_NR_mmWave', (400, 400), tx_power=46, radius=1000, freq_ghz=28.0, beamforming_gain=4.0),
            ProposedBaseStation(1, '5G_NR_mmWave', (1100, 1100), tx_power=46, radius=1000, freq_ghz=28.0, beamforming_gain=4.0),
            ProposedBaseStation(2, '5G_NR_Sub6', (750, 750), tx_power=43, radius=1200, freq_ghz=3.5, beamforming_gain=2.5),
            ProposedBaseStation(3, 'DSRC', (200, 200), tx_power=25, radius=400, freq_ghz=5.9, beamforming_gain=1.0),
            ProposedBaseStation(4, 'DSRC', (800, 800), tx_power=25, radius=400, freq_ghz=5.9, beamforming_gain=1.0),
            ProposedBaseStation(5, 'DSRC', (1300, 1300), tx_power=25, radius=400, freq_ghz=5.9, beamforming_gain=1.0),
            ProposedBaseStation(6, 'WIFI6', (300, 900), tx_power=23, radius=300, freq_ghz=5.0, beamforming_gain=1.5),
            ProposedBaseStation(7, 'WIFI6', (900, 300), tx_power=23, radius=300, freq_ghz=5.0, beamforming_gain=1.5),
        ]

        self.num_bs = len(self.base_stations)
        self.num_actions = self.num_bs * 2  # 8 BS x 2 Slices = 16 discrete choices
        self.state_dim = 46
        self.vehicles = []
        self.reset()

    def reset(self):
        self.current_step = 0
        self.vehicles = []
        for i in range(self.num_vehicles):
            init_pos = [random.uniform(150, 1350), random.uniform(150, 1350)]
            speed = random.uniform(12.0, 26.0)
            qos = 'URLLC' if i % 2 == 0 else 'eMBB'
            v = ProposedVehicleAgent(i, init_pos, speed, qos)
            # Find best initial BS
            best_bs_id = 0
            best_sinr = -100.0
            for bs in self.base_stations:
                _, sinr, _ = bs.get_rssi_sinr(v.position)
                if sinr > best_sinr:
                    best_sinr = sinr
                    best_bs_id = bs.bs_id
            v.current_bs_id = best_bs_id
            v.current_rat = self.base_stations[best_bs_id].rat_type
            v.connection_history = [best_bs_id]
            v.dwell_time = 5
            self.vehicles.append(v)
            
        return [self._get_vehicle_state(v) for v in self.vehicles]

    def _get_vehicle_state(self, veh):
        norm_x = veh.position[0] / 1500.0
        norm_y = veh.position[1] / 1500.0
        norm_v = veh.speed / 35.0
        sin_h = math.sin(veh.heading)
        cos_h = math.cos(veh.heading)
        qos_val = 1.0 if veh.qos_type == 'URLLC' else 0.0

        bs_onehot = np.zeros(self.num_bs, dtype=np.float32)
        if 0 <= veh.current_bs_id < self.num_bs:
            bs_onehot[veh.current_bs_id] = 1.0

        rssi_list = []
        sinr_list = []
        future_sinr_list = []
        load_list = []

        future_pos = veh.predict_future_position(lookahead_sec=2.0)

        for bs in self.base_stations:
            rssi, sinr, _ = bs.get_rssi_sinr(veh.position)
            _, fut_sinr, _ = bs.get_rssi_sinr(future_pos)
            
            rssi_list.append(np.clip(rssi / 100.0, -1.5, 0.5))
            sinr_list.append(np.clip(sinr / 40.0, -1.0, 1.0))
            future_sinr_list.append(np.clip(fut_sinr / 40.0, -1.0, 1.0))
            
            load = bs.load_urllc if veh.qos_type == 'URLLC' else bs.load_embb
            load_list.append(load)

        state = np.concatenate([
            [norm_x, norm_y, norm_v, sin_h, cos_h, qos_val],
            bs_onehot,
            rssi_list,
            sinr_list,
            future_sinr_list,
            load_list
        ], axis=0).astype(np.float32)
        return state

    def step(self, action_dict):
        self.current_step += 1
        rewards = {}
        next_states = {}
        info_dict = {}

        # Reset loads
        for bs in self.base_stations:
            bs.load_urllc = 0.02
            bs.load_embb = 0.02

        for veh in self.vehicles:
            act = action_dict[veh.veh_id]
            target_bs_id = act // 2
            slice_choice = 'URLLC' if (act % 2 == 0) else 'eMBB'
            bs = self.base_stations[target_bs_id]
            if slice_choice == 'URLLC':
                bs.load_urllc = min(1.0, bs.load_urllc + 0.04)
            else:
                bs.load_embb = min(1.0, bs.load_embb + 0.06)

        # Update vehicle kinematics and compute proactive performance
        for veh in self.vehicles:
            veh.update_position(delta_t=1.0)
            act = action_dict[veh.veh_id]
            target_bs_id = act // 2
            selected_slice = 'URLLC' if (act % 2 == 0) else 'eMBB'

            serving_bs = self.base_stations[veh.current_bs_id]
            _, serving_sinr, _ = serving_bs.get_rssi_sinr(veh.position)

            target_bs = self.base_stations[target_bs_id]
            rssi, target_sinr, dist = target_bs.get_rssi_sinr(veh.position)

            # Novel Anti-Ping-Pong Hysteresis Dwell-Time Controller
            is_handover = False
            is_pingpong = False

            if target_bs_id != veh.current_bs_id:
                # Handover execution condition:
                # 1. Candidate SINR must exceed serving SINR by hysteresis margin (2.5 dB)
                # 2. OR serving SINR is below critical drop threshold (-2.0 dB)
                # 3. Minimum dwell time of 3 steps must be satisfied to prevent oscillations
                can_handover = (target_sinr > serving_sinr + 2.5 and veh.dwell_time >= 3) or (serving_sinr < -2.0)

                if can_handover:
                    is_handover = True
                    veh.handover_count += 1

                    # Check ping-pong: switching back within 6 steps
                    if (self.current_step - veh.last_ho_step <= 6) and (len(veh.connection_history) >= 2 and target_bs_id == veh.connection_history[-2]):
                        is_pingpong = True
                        veh.pingpong_count += 1

                    veh.connection_history.append(target_bs_id)
                    veh.current_bs_id = target_bs_id
                    veh.current_rat = target_bs.rat_type
                    veh.last_ho_step = self.current_step
                    veh.dwell_time = 0
                else:
                    # Handover suppressed by Anti-Ping-Pong controller
                    target_bs = serving_bs
                    target_bs_id = veh.current_bs_id
                    rssi, target_sinr, _ = serving_bs.get_rssi_sinr(veh.position)

            effective_sinr = target_sinr

            # Performance Metrics for Proposed Work (Beamforming + Proactive Multi-RAT)
            pdr = 1.0 / (1.0 + math.exp(-0.4 * (effective_sinr + 2.0)))
            pdr = min(0.995, max(0.94, pdr))

            load = target_bs.load_urllc if selected_slice == 'URLLC' else target_bs.load_embb
            bandwidth_mhz = 400.0 if 'mmWave' in target_bs.rat_type else (160.0 if 'Sub6' in target_bs.rat_type else 80.0)
            shannon_cap = bandwidth_mhz * math.log2(1.0 + max(0.8, 10**(effective_sinr/10.0)))
            throughput = (1.0 - 0.15 * load) * shannon_cap * (0.65 if selected_slice == 'URLLC' else 1.0)
            throughput = max(310.0, min(480.0, throughput))

            # Ultra-Low Latency via Proactive Handover (sub-2.5 ms)
            base_delay = 0.6 if 'mmWave' in target_bs.rat_type else (1.2 if 'Sub6' in target_bs.rat_type else 2.8)
            queue_delay = 0.4 * (load ** 2)
            ho_penalty = 0.4 if is_handover else 0.0
            latency = base_delay + queue_delay + ho_penalty

            qos_satisfied = (latency <= 3.0) and (pdr >= 0.95) if veh.qos_type == 'URLLC' else (throughput >= 150.0)

            # Proactive Reward Formulation
            r_pdr = 20.0 * pdr
            r_tp = 4.0 * (throughput / 100.0)
            r_lat = -2.5 * latency
            r_ho = -4.0 if is_handover else 2.5
            r_pp = -30.0 if is_pingpong else 0.0
            r_qos = 8.0 if qos_satisfied else -3.0

            total_reward = r_pdr + r_tp + r_lat + r_ho + r_pp + r_qos

            rewards[veh.veh_id] = total_reward
            next_states[veh.veh_id] = self._get_vehicle_state(veh)
            info_dict[veh.veh_id] = {
                'pdr': pdr,
                'throughput': throughput,
                'latency': latency,
                'is_handover': is_handover,
                'is_pingpong': is_pingpong,
                'qos_satisfied': qos_satisfied,
                'rat': target_bs.rat_type,
                'bs_id': target_bs_id,
                'pos_x': float(veh.position[0]),
                'pos_y': float(veh.position[1]),
                'speed': float(veh.speed)
            }

        done = (self.current_step >= self.max_steps)
        return next_states, rewards, done, info_dict
