import sys
import os
import time
import requests

try:
    from neural_regulator import NeuralOffloadingRegulator
except ImportError:
    print("Error: neural_regulator.py not found.")
    sys.exit(1)

# =============================================================================
# LATENCY MEASUREMENT TEMPLATES (from Sprint 5 Task)
# =============================================================================

def measure_cloud_latency(cloud_url="http://192.168.1.72/ping"):
    """HTTP round-trip to the cloud inference endpoint in ms"""
    try:
        t0 = time.perf_counter()
        requests.get(cloud_url, timeout=2)
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return 999.0  # unreachable -> penalize cloud

def measure_local_latency(simulated_delay=0.079):
    """Run a local inference and time it in ms (simulated or real)"""
    try:
        t0 = time.perf_counter()
        # Simulate local load/inference time
        time.sleep(simulated_delay)
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return 999.0

# =============================================================================
# VALIDATION SUITE
# =============================================================================

def test_brain():
    weights_path = "model_weights.json"
    if not os.path.exists(weights_path):
        print("Error: %s not found. Please transfer it first." % weights_path)
        sys.exit(1)
        
    try:
        brain = NeuralOffloadingRegulator(weights_path)
        print("\n" + "="*50)
        print("--- BRAIN VALIDATION SUITE ---")
        print("="*50)
        
        # Test Case 1: Ideal Conditions
        # Logic: Low latency cloud + fast network -> CLOUD
        print("\n[Test 1] Ideal Conditions (Low RTT, High BW)")
        brain.predict(rtt=10.0, bw=80000.0, cloud_lat=50.0, local_lat=80.0)
        
        # Test Case 2: Poor Network
        # Logic: Reliable network, but RTT is high -> LOCAL
        print("\n[Test 2] Poor Network (High RTT, Low BW)")
        brain.predict(rtt=250.0, bw=500.0, cloud_lat=300.0, local_lat=80.0)
        
        # Test Case 3: Busy Jetson (High Local Latency)
        # Logic: Network is perfect, but Local Latency is high -> CLOUD
        # This confirms the regulator realizes offloading helps even on a good network.
        print("\n[Test 3] Busy Jetson (High Local Latency)")
        # We simulate a local latency of 650ms (e.g. thermal throttling)
        brain.predict(rtt=15.0, bw=90000.0, cloud_lat=80.0, local_lat=650.0)
        
        # Test Case 4: Slow Cloud
        # Logic: Local is fine, Cloud is slow -> LOCAL
        # Prevents offloading to a backend that is more laggy than the device.
        print("\n[Test 4] Slow Cloud (High Cloud Latency)")
        brain.predict(rtt=20.0, bw=50000.0, cloud_lat=800.0, local_lat=80.0)

        print("\n" + "="*50)
        print("Success: All test scenarios processed.")
        print("If probabilities (p) follow the logic above, you are ready to deploy!")
        print("="*50)
        
    except Exception as exc:
        print("Error during test: %s" % exc)
        sys.exit(1)

if __name__ == "__main__":
    test_brain()
