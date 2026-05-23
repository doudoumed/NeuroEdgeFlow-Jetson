# System Architecture

- **Jetson Exporter** (`jetson_exporter.py`): Exposes GPU/CPU/Temp/Power/FPS via Prometheus.
- **Adaptive Engine** (`adaptive_engine.py`): Decides Edge ↔ Cloud using CSV features and a light‑weight model.
- **Edge Pipeline** (`edge_pipeline.py`): Local YOLO inference.
- **Cloud Client** (`cloud_client.py`): Sends frames to Triton Inference Server.
- **Network Monitor** (`network_monitor.py`): Logs bandwidth / latency.
