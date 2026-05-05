#!/usr/bin/env python3
import time, cv2, numpy as np, os, csv, sys
import urllib.request, json
from collections import deque

SERVER_URL = "http://192.168.55.100:8000"
MODEL_NAME = "yolov5su"
VIDEO_PATH = "test_video.mp4"
LOG_FILE   = "training_dataset.csv"

# Rolling window for history
HISTORY = 3  # last 3 cycles

def get_gpu_usage():
    """Try to get GPU usage from Jetson or nvidia-smi"""
    try:
        with open("/sys/devices/gpu.0/load", "r") as f:
            return float(f.read().strip()) / 10.0  # percentage
    except:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=2
        ).decode().strip()
        return float(out)
    except:
        return -1.0  # not available

def get_queue_depth(server_url, model_name):
    """Get pending inference queue depth from Triton metrics"""
    try:
        req = urllib.request.Request("%s/v2/models/%s/stats" % (server_url, model_name))
        resp = urllib.request.urlopen(req, timeout=3)
        stats = json.loads(resp.read())
        # pending queue count
        infer_stats = stats["model_stats"][0]["inference_stats"]
        queue = infer_stats.get("queue", {}).get("count", 0)
        return int(queue)
    except:
        return 0

def infer_http(payload):
    data = json.dumps({
        "inputs": [{
            "name": "input",
            "shape": list(payload.shape),
            "datatype": "FP32",
            "data": payload.flatten().tolist()
        }]
    }).encode()
    payload_bytes = len(data)
    req = urllib.request.Request(
        "%s/v2/models/%s/infer" % (SERVER_URL, MODEL_NAME),
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t1 = time.time()
    resp = urllib.request.urlopen(req)
    resp.read()
    rtt = (time.time() - t1) * 1000  # ms
    return rtt, payload_bytes

def run(scenario_name, num_frames=100):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video " + VIDEO_PATH)
        return

    # State tracking
    rtt_history   = deque(maxlen=HISTORY)
    bw_history    = deque(maxlen=HISTORY)
    prev_rtt      = None

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                # Identity
                "Scenario", "Frame", "Timestamp",
                # 1. RTT
                "RTT_ms", "RTT_prev_ms", "RTT_trend",
                # 2. Bandwidth
                "BW_KBps", "BW_avg_KBps",
                # 3. GPU
                "GPU_pct",
                # 4. Queue
                "Queue_depth",
                # 5. Delta
                "Delta_RTT_ms",
                # 6. Recent history (last 3)
                "RTT_h1", "RTT_h2", "RTT_h3",
                "BW_h1",  "BW_h2",  "BW_h3",
                # FPS
                "FPS",
            ])

        print("Collecting %d frames for: %s" % (num_frames, scenario_name))
        print("Features: RTT | Bandwidth | GPU | Queue | Delta | History")
        print("-" * 65)

        it = 0
        while it < num_frames:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Preprocess
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            dec = cv2.imdecode(buf, 1)
            t = cv2.resize(dec, (640, 640))
            t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = np.transpose(t, (2, 0, 1))[np.newaxis]
            payload = np.ascontiguousarray(t)

            print("  [%d/%d] Sending..." % (it+1, num_frames), end=" ", flush=True)

            # ── 1. RTT ──────────────────────────────────────────
            rtt_ms, payload_bytes = infer_http(payload)
            rtt_prev  = prev_rtt if prev_rtt is not None else rtt_ms
            rtt_trend = "UP" if rtt_ms > rtt_prev else ("DOWN" if rtt_ms < rtt_prev else "STABLE")
            rtt_history.append(rtt_ms)

            # ── 2. Bandwidth ─────────────────────────────────────
            bw_kbps = (payload_bytes / 1024.0) / (rtt_ms / 1000.0)
            bw_history.append(bw_kbps)
            bw_avg = sum(bw_history) / len(bw_history)

            # ── 3. GPU ───────────────────────────────────────────
            gpu_pct = get_gpu_usage()

            # ── 4. Queue depth ───────────────────────────────────
            queue = get_queue_depth(SERVER_URL, MODEL_NAME)

            # ── 5. Delta RTT ─────────────────────────────────────
            delta_rtt = rtt_ms - rtt_prev

            # ── 6. Recent history (pad with 0 if not enough) ─────
            hist = list(rtt_history)
            while len(hist) < HISTORY:
                hist.insert(0, 0.0)
            bw_hist = list(bw_history)
            while len(bw_hist) < HISTORY:
                bw_hist.insert(0, 0.0)

            fps = 1000.0 / rtt_ms

            # Write row
            writer.writerow([
                scenario_name, it+1, round(time.time(), 3),
                round(rtt_ms, 2), round(rtt_prev, 2), rtt_trend,
                round(bw_kbps, 2), round(bw_avg, 2),
                round(gpu_pct, 1),
                queue,
                round(delta_rtt, 2),
                round(hist[0], 2), round(hist[1], 2), round(hist[2], 2),
                round(bw_hist[0], 2), round(bw_hist[1], 2), round(bw_hist[2], 2),
                round(fps, 3),
            ])
            f.flush()

            print("RTT:%.0fms | BW:%.1fKB/s | GPU:%.0f%% | Q:%d | Trend:%s" % (
                rtt_ms, bw_kbps, gpu_pct, queue, rtt_trend))

            prev_rtt = rtt_ms
            it += 1

    cap.release()
    print("\nDone! Saved to %s" % LOG_FILE)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    run(name)
