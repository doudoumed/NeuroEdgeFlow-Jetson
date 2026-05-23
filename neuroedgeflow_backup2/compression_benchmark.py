#!/usr/bin/env python3
import time, cv2, numpy as np, os
import tritonclient.grpc as grpcclient

SERVER_URL = "10.0.20.10:8001"
MODEL_NAME = "yolov5su"
IMAGE_PATH = os.path.expanduser("~/yolov5/data/images/bus.jpg")

def run_bench():
    client = grpcclient.InferenceServerClient(url=SERVER_URL)
    img = cv2.imread(IMAGE_PATH)
    if img is None: print("Err: No image"); return
    
    # Modes to test
    strats = [("Raw", None), ("JPEG 70", 70), ("JPEG 90", 90)]
    print(f"{'Mode':<10} | {'Size':<8} | {'Latency':<10}")
    print("-" * 35)
    
    for name, q in strats:
        if q is None:
            # RAW
            t = cv2.resize(img, (640, 640))
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            t = np.transpose(t, (2, 0, 1))[np.newaxis]
            payload = np.ascontiguousarray(t)
            size = payload.nbytes / 1024
        else:
            # JPEG
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
            size = len(buf) / 1024
            # Decode for inference prep
            dec = cv2.imdecode(buf, 1)
            t = cv2.resize(dec, (640, 640))
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            t = np.transpose(t, (2, 0, 1))[np.newaxis]
            payload = np.ascontiguousarray(t)

        # Infer
        inp = grpcclient.InferInput("input", payload.shape, "FP32")
        inp.set_data_from_numpy(payload)
        t1 = time.time()
        client.infer(MODEL_NAME, inputs=[inp], outputs=[grpcclient.InferRequestedOutput("output")])
        ms = (time.time()-t1)*1000
        print(f"{name:<10} | {size:>6.1f} KB | {ms:>8.1f} ms")

run_bench()
