import numpy as np
import math
import random

class BaseStation:
    def __init__(self, bs_id, rat_type, position, tx_power, radius, freq_ghz):
        self.bs_id = bs_id
        self.rat_type = rat_type  # 'DSRC', '5G_NR', 'WIFI6'
        self.position = np.array(position, dtype=np.float32)  # (x, y)
        self.tx_power = tx_power  # dBm
        self.radius = radius  # meters
        self.freq_ghz = freq_ghz  # GHz
        self.load_urllc = 0.0  # current normalized load [0, 1]
        self.load_embb = 0.0

    def compute_path_loss(self, distance):
        if distance <= 1.0:
            distance = 1.0
        # 3GPP Urban Micro / Macro path loss formula
        if self.rat_type == '5G_NR':
            pl = 32.4 + 20.0 * math.log10(self.freq_ghz) + 30.0 * math.log10(distance)
        elif self.rat_type == 'DSRC':
            pl = 38.0 + 20.0 * math.log10(self.freq_ghz) + 22.0 * math.log10(distance)
        else:  # WIFI6
            pl = 40.0 + 20.0 * math.log10(self.freq_ghz) + 25.0 * math.log10(distance)
        # Add log-normal shadowing
        shadowing = np.random.normal(0, 3.0)
        return pl + shadowing

    def get_rssi_sinr(self, veh_pos, interference_dbm=-95.0):
        dist = np.linalg.norm(veh_pos - self.position)
        if dist > self.radius * 1.5:
            return -120.0, -20.0, dist # Out of range
        
        pl = self.compute_path_loss(dist)
        rssi = self.tx_power - pl
        noise_dbm = -104.0
        
        # Total noise + interference in mW
        noise_mw = 10 ** (noise_dbm / 10.0)
        interf_mw = 10 ** (interference_dbm / 10.0)
        rssi_mw = 10 ** (rssi / 10.0)
        
        sinr = 10.0 * math.log10(rssi_mw / (noise_mw + interf_mw + 1e-12))
        return rssi, sinr, dist


class VehicleAgent:
    def __init__(self, veh_id, init_pos, init_speed, qos_type):
        self.veh_id = veh_id
        self.position = np.array(init_pos, dtype=np.float32)
        self.speed = init_speed  # m/s
        self.qos_type = qos_type  # 'URLLC' or 'eMBB'
        self.current_bs_id = -1
        self.current_rat = 'NONE'
        self.connection_history = []  # track last RAT connections for ping-pong detection
        self.handover_count = 0
        self.pingpong_count = 0

    def update_position(self, delta_t=1.0):
        # Simulate motion along road vector with small trajectory variation
        angle = random.choice([0, math.pi/2, math.pi, 3*math.pi/2])
        vx = self.speed * math.cos(angle) + random.uniform(-1.0, 1.0)
        vy = self.speed * math.sin(angle) + random.uniform(-1.0, 1.0)
        self.position += np.array([vx, vy], dtype=np.float32) * delta_t
        # Keep inside bounds [0, 1500]
        self.position = np.clip(self.position, 0.0, 1500.0)


class IoVHeterogeneousEnv:
    def __init__(self, num_vehicles=10, max_steps=100):
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define Base Stations layout across grid (1500m x 1500m)
        self.base_stations = [
            # 5G gNodeBs (Wide coverage)
            BaseStation(0, '5G_NR', (300, 300), tx_power=43, radius=1000, freq_ghz=3.5),
            BaseStation(1, '5G_NR', (1200, 1200), tx_power=43, radius=1000, freq_ghz=3.5),
            
            # DSRC RSUs (Short coverage, highway/intersections)
            BaseStation(2, 'DSRC', (0, 0), tx_power=23, radius=350, freq_ghz=5.9),
            BaseStation(3, 'DSRC', (500, 500), tx_power=23, radius=350, freq_ghz=5.9),
            BaseStation(4, 'DSRC', (1000, 1000), tx_power=23, radius=350, freq_ghz=5.9),
            
            # Wi-Fi 6 APs (Local high-throughput hotspots)
            BaseStation(5, 'WIFI6', (250, 750), tx_power=20, radius=200, freq_ghz=5.0),
            BaseStation(6, 'WIFI6', (750, 250), tx_power=20, radius=200, freq_ghz=5.0),
            BaseStation(7, 'WIFI6', (1250, 750), tx_power=20, radius=200, freq_ghz=5.0),
        ]
        
        self.num_bs = len(self.base_stations)
        # Action space for each vehicle: Choose BS (0 to 7) and Slice (0: URLLC, 1: eMBB)
        # Total discrete actions per agent = num_bs * 2 = 16
        self.num_actions = self.num_bs * 2
        # State space size per vehicle:
        # [x, y, speed, qos_type_val, curr_bs_onehot (8), RSSI_all (8), SINR_all (8), load_all (8)] = 3 + 1 + 8 + 8 + 8 + 8 = 36
        self.state_dim = 36

        self.vehicles = []
        self.reset()

    def reset(self):
        self.current_step = 0
        self.vehicles = []
        for i in range(self.num_vehicles):
            init_pos = [random.uniform(100, 1400), random.uniform(100, 1400)]
            speed = random.uniform(10.0, 25.0)  # ~36 - 90 km/h
            qos = 'URLLC' if i % 2 == 0 else 'eMBB'
            v = VehicleAgent(i, init_pos, speed, qos)
            # Default connect to nearest 5G
            v.current_bs_id = 0
            v.current_rat = '5G_NR'
            v.connection_history = [0]
            self.vehicles.append(v)
            
        return [self._get_vehicle_state(v) for v in self.vehicles]

    def _get_vehicle_state(self, veh):
        # 1. Pos (normalized) and Speed
        norm_x = veh.position[0] / 1500.0
        norm_y = veh.position[1] / 1500.0
        norm_v = veh.speed / 30.0
        qos_val = 1.0 if veh.qos_type == 'URLLC' else 0.0

        # 2. Current BS One-hot
        bs_onehot = np.zeros(self.num_bs, dtype=np.float32)
        if 0 <= veh.current_bs_id < self.num_bs:
            bs_onehot[veh.current_bs_id] = 1.0

        # 3. RSSI, SINR, BS Loads
        rssi_list = []
        sinr_list = []
        load_list = []
        for bs in self.base_stations:
            rssi, sinr, _ = bs.get_rssi_sinr(veh.position)
            rssi_list.append(np.clip(rssi / 100.0, -1.5, 0.5))  # Normalized
            sinr_list.append(np.clip(sinr / 40.0, -1.0, 1.0))
            load = bs.load_urllc if veh.qos_type == 'URLLC' else bs.load_embb
            load_list.append(load)

        state = np.concatenate([
            [norm_x, norm_y, norm_v, qos_val],
            bs_onehot,
            rssi_list,
            sinr_list,
            load_list
        ], axis=0).astype(np.float32)
        return state

    def step(self, action_dict):
        """
        action_dict: {veh_id: action_idx} where action_idx in [0, num_actions-1]
        action_idx maps to (target_bs_id, slice_choice)
        """
        self.current_step += 1
        rewards = {}
        next_states = {}
        info_dict = {}

        # Reset BS load calculations for current step
        for bs in self.base_stations:
            bs.load_urllc = 0.01
            bs.load_embb = 0.01

        # First pass: Process vehicle actions & compute new network loads
        for veh in self.vehicles:
            act = action_dict[veh.veh_id]
            target_bs_id = act // 2
            slice_choice = 'URLLC' if (act % 2 == 0) else 'eMBB'
            
            # Increment load on target BS
            bs = self.base_stations[target_bs_id]
            if slice_choice == 'URLLC':
                bs.load_urllc += 0.08
            else:
                bs.load_embb += 0.12
            bs.load_urllc = min(1.0, bs.load_urllc)
            bs.load_embb = min(1.0, bs.load_embb)

        # Second pass: Update positions, compute QoS metrics, rewards
        for veh in self.vehicles:
            veh.update_position(delta_t=1.0)
            act = action_dict[veh.veh_id]
            target_bs_id = act // 2
            selected_slice = 'URLLC' if (act % 2 == 0) else 'eMBB'
            
            target_bs = self.base_stations[target_bs_id]
            rssi, sinr, dist = target_bs.get_rssi_sinr(veh.position)
            
            # Check Handover & Ping-Pong
            is_handover = (target_bs_id != veh.current_bs_id)
            is_pingpong = False
            if is_handover:
                veh.handover_count += 1
                if len(veh.connection_history) >= 2 and target_bs_id == veh.connection_history[-2]:
                    is_pingpong = True
                    veh.pingpong_count += 1
                veh.connection_history.append(target_bs_id)
                veh.current_bs_id = target_bs_id
                veh.current_rat = target_bs.rat_type

            # Compute Channel Quality & Performance Metrics
            # PDR (Packet Delivery Ratio) sigmoid based on SINR
            pdr = 1.0 / (1.0 + math.exp(-0.3 * (sinr - 5.0)))
            
            # Throughput computation (Mbps) based on Shannon formula + slice factor
            load = target_bs.load_urllc if selected_slice == 'URLLC' else target_bs.load_embb
            bandwidth_mhz = 20.0 if target_bs.rat_type == 'DSRC' else (100.0 if target_bs.rat_type == '5G_NR' else 40.0)
            shannon_cap = bandwidth_mhz * math.log2(1.0 + max(0.1, 10**(sinr/10.0)))
            throughput = (1.0 - 0.7 * load) * shannon_cap * (0.3 if selected_slice == 'URLLC' else 1.0)

            # E2E Latency computation (ms)
            base_delay = 1.0 if target_bs.rat_type == '5G_NR' else (4.0 if target_bs.rat_type == 'DSRC' else 10.0)
            queue_delay = 5.0 * (load ** 2)
            ho_penalty = 15.0 if is_handover else 0.0
            latency = base_delay + queue_delay + ho_penalty

            # QoS Satisfaction Check
            if veh.qos_type == 'URLLC':
                qos_satisfied = (latency <= 10.0) and (pdr >= 0.95)
            else:
                qos_satisfied = (throughput >= 25.0)

            # Multi-objective Reward Calculation (Wu et al. [13] & Yacheur et al. [14])
            # R = w1*PDR + w2*Throughput_norm - w3*Latency_norm - w4*Handover - w5*PingPong
            r_pdr = 10.0 * pdr
            r_tp = 2.0 * min(5.0, throughput / 20.0)
            r_lat = -1.5 * (latency / 10.0)
            r_ho = -3.0 if is_handover else 0.5
            r_pp = -8.0 if is_pingpong else 0.0
            r_qos = 5.0 if qos_satisfied else -5.0

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
                'bs_id': target_bs_id
            }

        done = (self.current_step >= self.max_steps)
        return next_states, rewards, done, info_dict
