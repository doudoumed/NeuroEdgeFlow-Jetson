from neural_regulator import NeuralOffloadingRegulator

def run_tests():
    try:
        brain = NeuralOffloadingRegulator("model_weights.json")
    except Exception as e:
        print("Failed to load brain:", e)
        return

    print("\n--- NEURAL REGULATOR VALIDATION ---")
    
    # [RTT, BW, CPU, RAM]
    tests = [
        # Idle Jetson -> LOCAL 
        (1.0, 100000.0, 10.0, 40.0, "LOCAL"),
        # Busy Jetson -> CLOUD
        (1.0, 100000.0, 95.0, 80.0, "CLOUD"),
        # Busy Jetson but Terrible Network -> LOCAL
        (200.0, 500.0, 95.0, 80.0, "LOCAL"),
    ]

    passed = 0
    for i, (rtt, bw, cpu, ram, expected) in enumerate(tests):
        print(f"\nTest {i+1} [Expect: {expected}]")
        is_cloud, prob = brain.predict(rtt, bw, cpu, ram)
        got = "CLOUD" if is_cloud else "LOCAL"
        
        status = "✅ PASS" if got == expected else "❌ FAIL"
        print(status)
        if got == expected:
            passed += 1

    print(f"\nResult: {passed}/{len(tests)} Tests Passed.\n")

if __name__ == "__main__":
    run_tests()
