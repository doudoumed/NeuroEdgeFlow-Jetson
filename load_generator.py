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
    
    try:
        while True:
            # This generates significant heat and cycle consumption
            _ = np.dot(a, b)
            # Small sleep to prevent kernel lockup but keep load high (~90%)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[!] Load generator interrupted by user.")

if __name__ == "__main__":
    stress_test()
