import cv2
import numpy as np
import tritonclient.http as httpclient
import time

TRITON_URL  = "10.0.20.10:8000"
MODEL_NAME  = "yolov5su"
INPUT_NAME  = "images"
OUTPUT_NAME = "output0"
INPUT_SIZE  = 640

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img, axis=0))

frame = cv2.imread("bus.jpg")
orig_h, orig_w = frame.shape[:2]
client = httpclient.InferenceServerClient(url=TRITON_URL)

tensor = preprocess(frame)
inputs = [httpclient.InferInput(INPUT_NAME, tensor.shape, "FP32")]
inputs[0].set_data_from_numpy(tensor, binary_data=True)
outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME, binary_data=True)]

print("Running HTTP Warmup...")
client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

print("Running 10 inferences...")
times = []
for i in range(10):
    t0 = time.time()
    client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    ms = (time.time() - t0) * 1000
    times.append(ms)
    print(f"  Run {i+1}: {ms:.1f} ms")

avg_ms = sum(times[1:]) / len(times[1:])
print(f"\nAverage HTTP RTT (Runs 2-10): {avg_ms:.1f} ms | Est. FPS: {1000/avg_ms:.1f}")
