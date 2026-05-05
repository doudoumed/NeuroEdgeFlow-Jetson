import argparse
import time
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import threading
import subprocess
from collections import deque
import prometheus_client as prom

class NetworkMonitor:
    def __init__(self, server_host="localhost"):
        self.server_host = server_host
        self.metric_fps = prom.Gauge('yolo_fps', 'FPS')
        self.metric_rtt = prom.Gauge('yolo_rtt_ms', 'RTT ms')
        try: prom.start_http_server(8003)
        except: pass
        self._running = True
        self._rtt_thread = threading.Thread(target=self._rtt_loop, daemon=True)
        self._rtt_thread.start()

    def _rtt_loop(self):
        while self._running:
            try:
                res = subprocess.run(["ping", "-c", "1", "-W", "1", self.server_host], capture_output=True, text=True)
                if res.returncode == 0 and 'time=' in res.stdout:
                    rtt = float(res.stdout.split('time=')[1].split(' ')[0])
                    self.metric_rtt.set(rtt)
            except: pass
            time.sleep(1)

    def update(self, inf_time):
        fps = 1.0 / inf_time if inf_time > 0 else 0
        self.metric_fps.set(fps)

    def stop(self): self._running = False

def run_demo(source, server_url):
    print(f"🚀 Starting Headless Demo. Sending metrics to PC Dashboard...")
    client = grpcclient.InferenceServerClient(url=server_url)
    monitor = NetworkMonitor(server_host=server_url.split(":")[0])
    cap = cv2.VideoCapture(source)
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)
        
        inputs = [grpcclient.InferInput("input", img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img)
        
        t0 = time.time()
        try:
            client.infer("yolov5", inputs=inputs, outputs=[grpcclient.InferRequestedOutput("output")])
            inf_time = time.time() - t0
            monitor.update(inf_time)
            if count % 30 == 0:
                print(f"Processed {count} frames... FPS: {1.0/inf_time:.1f}")
        except Exception as e:
            print(f"Triton Error: {e}")
            
        count += 1
        
    cap.release()
    monitor.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--server", type=str, required=True)
    args = parser.parse_args()
    run_demo(args.source, args.server)

