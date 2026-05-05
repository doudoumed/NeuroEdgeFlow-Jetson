#!/usr/bin/env python3
"""
test_jetson_to_triton.py — Simple gRPC test for YOLOv5su on Triton

This script sends a single image (bus.jpg) to the Triton server
running on the host PC (192.168.55.100) to verify the connection.
"""

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import os

# CONFIGURATION
SERVER_URL = "192.168.1.72:8001"
MODEL_NAME = "yolov5su"
IMAGE_PATH = os.path.expanduser("/home/nvidia/bus.jpg")

def test_inference():
    print(f"Connecting to Triton at {SERVER_URL}...")
    try:
        client = grpcclient.InferenceServerClient(url=SERVER_URL)
        if not client.is_server_ready():
            print("ERROR: Server is not ready. Check if Docker is running and IP is correct.")
            return
        
        if not client.is_model_ready(MODEL_NAME):
            print(f"ERROR: Model '{MODEL_NAME}' is not ready on server.")
            return
        
        print(f"SUCCESS: Connected to {MODEL_NAME}!")

        # Load and preprocess image
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"ERROR: Could not find image at {IMAGE_PATH}")
            return
        
        orig_h, orig_w = img.shape[:2]
        
        # YOLOv5su preprocessing: [1, 3, 640, 640]
        input_img = cv2.resize(img, (640, 640))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.ascontiguousarray(np.expand_dims(input_img, axis=0))

        # Build gRPC request
        inputs = [grpcclient.InferInput("input", input_img.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_img)
        outputs = [grpcclient.InferRequestedOutput("output")]

        print("Sending inference request...")
        resp = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        
        raw_output = resp.as_numpy("output")
        print(f"Received result with shape: {raw_output.shape}")
        
        # Simple postprocess (count detections above 0.5)
        # Output: [1, 25200, 6] -> [cx, cy, w, h, conf, cls]
        detections = raw_output[0]
        confs = detections[:, 4]
        valid_dets = detections[confs > 0.5]
        
        print(f"Detections found: {len(valid_dets)}")
        for i, det in enumerate(valid_dets):
            print(f"  Det {i+1}: Class {int(det[5])} | Conf: {det[4]:.2f}")

        print("\nTEST COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")

if __name__ == "__main__":
    test_inference()
