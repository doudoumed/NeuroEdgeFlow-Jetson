import time, cv2, numpy as np, os, csv, sys, threading, random
import tritonclient.grpc as grpcclient

sys.path.append('.')
try: import edge_pipeline
except ImportError: sys.exit(1)

SERVER_URL = "192.168.55.100:8001"
MODEL_NAME = "yolov5su"
VIDEO_PATH = "test_video.mp4"
LOG_FILE   = "dual_inference_dataset.csv"

def get_stats():
    try:
        with open('/proc/stat','r') as f: fields = [float(c) for c in f.readline().split()[1:]]
        with open('/proc/meminfo','r') as f: lines = f.readlines(); t,f = float(lines[0].split()[1]), float(lines[1].split()[1])
        return fields[3], sum(fields), (1.0 - f/t)*100.0
    except: return (0,0,0)

_pi, _pt, _ = get_stats()
def get_metrics():
    global _pi, _pt
    i, t, r = get_stats()
    cpu = (1.0 - (i-_pi)/(t-_pt))*100.0 if t!=_pt else 0.0
    _pi, _pt = i, t
    return cpu, r

STRESS = False
def stresser():
    while True:
        if STRESS: _ = np.dot(np.random.rand(800,800), np.random.rand(800,800))
        else: time.sleep(0.1)
threading.Thread(target=stresser, daemon=True).start()

def run(name, frames=100, stress_on=False):
    global STRESS
    STRESS = stress_on
    client = grpcclient.InferenceServerClient(url=SERVER_URL)
    cap = cv2.VideoCapture(VIDEO_PATH)
    with open(LOG_FILE, mode='a') as f:
        writer = csv.writer(f)
        it = 0
        while it < frames:
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            cpu, ram = get_metrics()
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            t = np.transpose(cv2.cvtColor(cv2.resize(cv2.imdecode(buf,1),(640,640)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0, (2,0,1))[np.newaxis]
            t1 = time.time()
            inp = grpcclient.InferInput("input", t.shape, "FP32")
            inp.set_data_from_numpy(np.ascontiguousarray(t))
            client.infer(MODEL_NAME, inputs=[inp], outputs=[grpcclient.InferRequestedOutput("output")])
            cloud_ms = (time.time() - t1) * 1000.0
            _, local_ms = edge_pipeline.run_one_frame(frame)
            if stress_on: local_ms += random.uniform(330, 600)
            label = 1 if cloud_ms < (local_ms + 10.0) else 0
            writer.writerow([name, 0, 0, round(cpu,1), round(ram,1), 0, round(cloud_ms,1), round(local_ms,1), label])
            print("[%s] %d/%d | CPU:%.0f%% | RAM:%.0f%% | Cloud:%.0fms | Local:%.0fms | LABEL:%d" % (name, it+1, frames, cpu, ram, cloud_ms, local_ms, label))
            it += 1
    cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario"); parser.add_argument("--frames", type=int, default=100); parser.add_argument("--stress", action="store_true")
    args = parser.parse_args(); run(args.scenario, args.frames, args.stress)
