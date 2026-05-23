import argparse
import time
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from network_monitor import NetworkMonitor

# COCO names
COCO_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

def preprocess(frame, size=416):
    img = cv2.resize(frame, (size, size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)

def postprocess(output, frame, input_size=416, conf_thresh=0.25):
    h, w = frame.shape[:2]
    sx, sy = w / input_size, h / input_size
    dets = output[0]
    boxes = []
    for det in dets:
        if det[4] < conf_thresh: continue
        scores = det[5:]
        cls_id = int(np.argmax(scores))
        if det[4] * scores[cls_id] < conf_thresh: continue
        cx, cy, bw, bh = det[:4]
        boxes.append((int((cx-bw/2)*sx), int((cy-bh/2)*sy), int((cx+bw/2)*sx), int((cy+bh/2)*sy), cls_id, det[4]*scores[cls_id]))
    for (x1, y1, x2, y2, cls_id, score) in boxes:
        label = f"{COCO_NAMES[cls_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def run_demo(source, server_url, duration=600):
    client = grpcclient.InferenceServerClient(url=server_url)
    monitor = NetworkMonitor(server_host=server_url.split(":")[0])
    monitor.start_background_rtt()
    cap = cv2.VideoCapture(source)
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret: break
        
        tensor = preprocess(frame)
        inputs = [grpcclient.InferInput("images", tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(tensor)
        
        t0 = time.time()
        try:
            result = client.infer("yolov7", inputs=inputs, outputs=[grpcclient.InferRequestedOutput("output")])
            latency = (time.time() - t0) * 1000
            monitor.update(time.time()-t0, tensor.nbytes, latency*0.1, False)
            frame = postprocess(result.as_numpy("output"), frame)
        except Exception as e:
            print(f"Error: {e}")
            
        cv2.imshow("Jetson Distributed Demo", frame)
        if cv2.waitKey(1) == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    monitor.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--server", type=str, required=True)
    args = parser.parse_args()
    run_demo(args.source, args.server)

