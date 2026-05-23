import json
import numpy as np
import os
import collections

class NeuralOffloadingRegulator:
    def __init__(self, weights_path="rf_model.json"):
        if not os.path.exists(weights_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(script_dir, "rf_model.json")
            
        if not os.path.exists(weights_path):
            # Fallback to the provided model_weights.json if rf_model.json is not around
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Golden Brain not found at {weights_path}")
            
        with open(weights_path, 'r') as f:
            self.model = json.load(f)

        self.is_rf = "trees" in self.model
        
        if self.is_rf:
            self.feature_names = self.model["feature_names"]
            print(f"[NeuralRegulator] {len(self.model['trees'])}-Tree RF Brain Loaded successfully from {weights_path}")
            
            # History buffers for temporal features
            self.history = collections.defaultdict(lambda: collections.deque([0.0]*5, maxlen=5))
        else:
            # Legacy MLP support
            self.mean = np.array(self.model['scaler_mean'], dtype=np.float32)
            self.scale = np.array(self.model['scaler_scale'], dtype=np.float32)
            self.weights = []
            self.biases = []
            i = 1
            while f'w{i}' in self.model:
                self.weights.append(np.array(self.model[f'w{i}'], dtype=np.float32))
                self.biases.append(np.array(self.model[f'b{i}'], dtype=np.float32))
                i += 1
            print(f"[NeuralRegulator] {len(self.weights)}-Layer MLP Brain Loaded successfully from {weights_path}")

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _evaluate_tree(self, node, features):
        if node["leaf"]:
            return node["value"][1] # probability of class 1 (CLOUD)
        
        feat_idx = node["feature"]
        threshold = node["threshold"]
        
        if features[feat_idx] <= threshold:
            return self._evaluate_tree(node["left"], features)
        else:
            return self._evaluate_tree(node["right"], features)

    def extract_features(self, metrics):
        """Builds the 48 feature array based on latest metrics and history."""
        rtt = float(metrics.get("rtt_ms", 0.0))
        bw = float(metrics.get("bandwidth_kbps", 0.0))
        cpu = float(metrics.get("cpu_load", 0.0))
        ram = float(metrics.get("ram_usage", 0.0))
        
        prev_rtt = self.history['rtt_ms'][-1]
        rtt_delta = rtt - prev_rtt
        
        # 1. Update history
        self.history['rtt_ms'].append(rtt)
        self.history['bandwidth_kbps'].append(bw)
        self.history['cpu_load'].append(cpu)
        self.history['ram_usage'].append(ram)
        self.history['prev_rtt_ms'].append(prev_rtt)
        self.history['rtt_delta'].append(rtt_delta)
        
        # Helper to get lags 
        def get_lag(k, lag):
            return self.history[k][4 - lag]
            
        def get_roll(k, window):
            vals = list(self.history[k])[5 - window:]
            return sum(vals) / len(vals)

        # Base Features dict
        F = {}
        F['rtt_ms'] = rtt
        F['bandwidth_kbps'] = bw
        F['cpu_load'] = cpu
        F['ram_usage'] = ram
        F['prev_rtt_ms'] = prev_rtt
        F['rtt_delta'] = rtt_delta
        F['avg_bw_last_6s'] = get_roll('bandwidth_kbps', 3)
        F['local_latency_ms'] = float(metrics.get('local_latency_ms', 0.0))
        F['queue_depth'] = float(metrics.get('queue_depth', 0.0))
        F['error_rate_pct'] = float(metrics.get('error_rate_pct', 0.0))
        F['network_ok'] = 1.0 if metrics.get('network_ok', False) else 0.0
        F['bandwidth_ok'] = 1.0 if metrics.get('bandwidth_ok', False) else 0.0
        
        # Temporal
        for k in ['rtt_ms', 'bandwidth_kbps', 'cpu_load', 'ram_usage', 'prev_rtt_ms', 'rtt_delta']:
            F[f"{k}_lag1"] = get_lag(k, 1)
            F[f"{k}_lag2"] = get_lag(k, 2)
            F[f"{k}_roll3"] = get_roll(k, 3)
            F[f"{k}_roll5"] = get_roll(k, 5)
            
        F['rtt_delta_lag'] = F['rtt_delta_lag1']
        bw_delta = bw - get_lag('bandwidth_kbps', 1)
        bw_delta_lag1 = get_lag('bandwidth_kbps', 1) - get_lag('bandwidth_kbps', 2)
        F['bw_delta_lag'] = bw_delta_lag1
        
        F['rtt_increasing'] = 1.0 if rtt > get_lag('rtt_ms', 1) else 0.0
        F['bw_increasing'] = 1.0 if bw > get_lag('bandwidth_kbps', 1) else 0.0
        F['rtt_bw_ratio'] = rtt / max(bw, 1.0)
        F['cpu_ram_combined'] = cpu + ram
        F['network_quality'] = 1.0 if (rtt < 50 and bw > 300) else 0.0
        F['poll_rank'] = float(metrics.get('poll_id', 0))
        F['time_in_session'] = F['poll_rank'] * 2.0
        
        trend = rtt_delta
        F['rtt_trend_down'] = 1.0 if trend < -5 else 0.0
        F['rtt_trend_stable'] = 1.0 if -5 <= trend <= 5 else 0.0
        F['rtt_trend_up'] = 1.0 if trend > 5 else 0.0
        
        # Build vector matching exact order trained in JSON
        vec = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            vec[i] = F.get(name, 0.0)
            
        return vec

    def predict(self, rtt=None, bw=None, cpu=None, ram=None, cloud_lat=None, local_lat=None, metrics=None):
        if metrics is None:
            # Fallback for legacy calls (test_brain)
            metrics = {
                "rtt_ms": rtt or 0.0,
                "bandwidth_kbps": bw or 0.0,
                "cpu_load": cpu or 20.0,
                "ram_usage": ram or 30.0,
                "local_latency_ms": local_lat or 30.0,
                "cloud_latency_ms": cloud_lat or 50.0,
            }

        if self.is_rf:
            features = self.extract_features(metrics)
            probs = [self._evaluate_tree(tree, features) for tree in self.model["trees"]]
            prob = sum(probs) / len(probs)
        else:
            # Legacy MLP
            x = np.array([metrics["rtt_ms"], metrics["bandwidth_kbps"], metrics["cpu_load"], metrics["ram_usage"]], dtype=np.float32)
            x = (x - self.mean) / self.scale
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = np.dot(x, w) + b
                if i == len(self.weights) - 1:
                    x = z
                else:
                    x = self._relu(z)
            prob = self._sigmoid(x)[0]

        cloud_win = prob > 0.5
        mode_str = "CLOUD" if cloud_win else "LOCAL"
        
        rtt_val = float(metrics.get("rtt_ms", 0.0))
        cpu_val = float(metrics.get("cpu_load", 0.0))
        
        print(f"[NeuralRegulator] RTT:{rtt_val:3.0f}ms CPU:{cpu_val:4.1f}% -> Mode: {mode_str} (p={prob:.3f})")
        return cloud_win, float(prob)
