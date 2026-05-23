#!/usr/bin/env python3
import numpy as np
import time
import os

def stress_test():
    print("==========================================")
    print("   NEUROEDGEFLOW JETSON LOAD GENERATOR    ")
    print("==========================================")
    print("Target: High Local Latency Simulation")
    print("Method: Large Matrix Multiplication (CPU/GPU)")
    print("Press Ctrl+C to Stop")
    
    # Pre-allocate matrices to reduce allocation noise
    size = 1500
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    import math
    import random
    
    t0 = time.time()
    try:
        while True:
            # This generates significant heat and cycle consumption
            _ = np.dot(a, b)
            
            # Dynamic sleep: oscillate load over a 20-second period
            elapsed = time.time() - t0
            # Sine wave from 0.0 to 1.0
            wave = (math.sin(elapsed * math.pi / 10.0) + 1.0) / 2.0 
            
            # Add some random noise (+/- 10%)
            noise = random.uniform(-0.1, 0.1)
            factor = max(0.0, min(1.0, wave + noise))
            
            # Interpolate sleep: 
            # 0.01s -> ~66% CPU load (high)
            # 0.08s -> ~30% CPU load (low)
            sleep_time = 0.01 + (factor * 0.07)
            
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[!] Load generator interrupted by user.")

if __name__ == "__main__":
    stress_test()
