import argparse
import time
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import threading
import subprocess
import prometheus_client as prom

class NetworkMonitor:
    def __init__(self, server_host):
        self.server_host = server_host
        self.metric_fps = prom.Gauge('yolo_fps', 'FPS')
        self.metric_rtt = prom.Gauge('yolo_rtt_ms', 'RTT ms')
        prom.start_http_server(8003)
        self._running = True
        threading.Thread(target=self._rtt_loop, daemon=True).start()

    def _rtt_loop(self):
        while self._running:
            try:
                res = subprocess.run(["ping", "-c", "1", "-W", "1", self.server_host], capture_output=True, text=True)
                if "time=" in res.stdout:
                    self.metric_rtt.set(float(res.stdout.split('time=')[1].split(' ')[0]))
            except: pass
            time.sleep(1)

    def update(self, inf_time): self.metric_fps.set(1.0/inf_time)

def run_demo(source, server_url):
    print(f"🚀 Cloud-Inference started! Sending results to PC dashboard...")
    client = grpcclient.InferenceServerClient(url=server_url)
    monitor = NetworkMonitor(server_host=server_url.split(":")[0])
    cap = cv2.VideoCapture(source)
    
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
            monitor.update(time.time() - t0)
        except Exception as e:
            print(f"Server Error: {e}")
            time.sleep(1)
            
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--server", type=str, required=True)
    args = parser.parse_args()
    run_demo(args.source, args.server)

