#!/usr/bin/env python3
"""
Patches the existing grafana_dashboard.json to add a mode timeline strip.
Run on your laptop:
  python3 make_timeline_panel.py \
    --dashboard ~/neuroedgeflow-monitoring/monitoring/grafana_dashboard.json \
    --out       ~/neuroedgeflow-monitoring/monitoring/grafana_dashboard.json
"""
import json, argparse, os, sys, copy

# =============================================================================
# THE NEW PANEL — State Timeline strip
# =============================================================================
# Grafana 10.4.2 state-timeline panel.
# Datasource UID from Sprint 4: ffg3t65m1jshse
# Metric: neuro_mode pushed by adaptive_engine.py to Pushgateway
#   0 = LOCAL  (green)
#   1 = CLOUD  (blue)
#
# Panel is placed in a new row at the bottom of the dashboard,
# spanning the full 24-column width, height 4 (thin strip).
# =============================================================================

TIMELINE_PANEL = {
    "id": 99,                          # will be reassigned to max_id+1
    "type": "state-timeline",
    "title": "Inference Mode Timeline — LOCAL / CLOUD",
    "description": (
        "Live mode decisions from the Sprint 5 Adaptive Engine. "
        "GREEN = LOCAL (TensorRT FP16 on Jetson). "
        "BLUE = CLOUD (Triton gRPC on laptop). "
        "Metric: neuro_mode{job=\"adaptive_engine\"} via Pushgateway. "
        "0 = LOCAL, 1 = CLOUD."
    ),
    "datasource": {
        "type": "prometheus",
        "uid":  "ffg3t65m1jshse"       # Sprint 4 Prometheus datasource UID
    },
    "gridPos": { "x": 0, "y": 99, "w": 24, "h": 4 },   # y reassigned below
    "targets": [
        {
            "datasource": { "type": "prometheus", "uid": "ffg3t65m1jshse" },
            "expr":         "neuro_mode{job=\"adaptive_engine\"}",
            "legendFormat": "Inference Mode",
            "refId":        "A",
            "instant":      False,
            "range":        True
        }
    ],
    "options": {
        "mergeValues":    True,
        "showValue":      "always",
        "alignValue":     "center",
        "rowHeight":      0.9,
        "legend": {
            "displayMode": "list",
            "placement":   "bottom",
            "showLegend":  True
        },
        "tooltip": { "mode": "single", "sort": "none" }
    },
    "fieldConfig": {
        "defaults": {
            "custom": {
                "lineWidth":    1,
                "fillOpacity":  80,
                "spanNulls":    False,
                "insertNulls":  False,
                "hideFrom":     { "legend": False, "tooltip": False, "viz": False }
            },
            "color": { "mode": "thresholds" },
            "thresholds": {
                "mode": "absolute",
                "steps": [
                    { "color": "green", "value": None },   # 0 = LOCAL = green
                    { "color": "blue",  "value": 1    }    # 1 = CLOUD = blue
                ]
            },
            "mappings": [
                {
                    "type": "value",
                    "options": {
                        "0": {
                            "text":  "LOCAL",
                            "color": "green",
                            "index": 0
                        },
                        "1": {
                            "text":  "CLOUD",
                            "color": "blue",
                            "index": 1
                        }
                    }
                }
            ],
            "min": 0,
            "max": 1,
            "unit": "short"
        },
        "overrides": []
    },
    "transparent": False,
    "links":        []
}

# Separator row panel that groups the timeline visually
SEPARATOR_ROW = {
    "id":      98,
    "type":    "row",
    "title":   "Sprint 5 — Adaptive Engine",
    "collapsed": False,
    "gridPos": { "x": 0, "y": 98, "w": 24, "h": 1 },
    "panels":  []
}


def patch(dashboard_path, out_path):
    if not os.path.exists(dashboard_path):
        print("ERROR: dashboard not found: " + dashboard_path)
        sys.exit(1)

    with open(dashboard_path) as f:
        dash = json.load(f)

    panels = dash.get("panels", [])

    # ── Check if timeline panel already exists ────────────────────────────────
    existing_titles = [p.get("title","") for p in panels]
    if "Inference Mode Timeline — LOCAL / CLOUD" in existing_titles:
        print("Timeline panel already exists — skipping add.")
        print("If you want to replace it, remove the existing panel first.")
        return

    # ── Find max panel id and max y position ──────────────────────────────────
    max_id = max((p.get("id", 0) for p in panels), default=0)
    max_y  = max(
        (p.get("gridPos", {}).get("y", 0) + p.get("gridPos", {}).get("h", 0)
         for p in panels),
        default=0
    )

    # ── Assign ids and y positions ────────────────────────────────────────────
    sep = copy.deepcopy(SEPARATOR_ROW)
    sep["id"]             = max_id + 1
    sep["gridPos"]["y"]   = max_y

    panel = copy.deepcopy(TIMELINE_PANEL)
    panel["id"]           = max_id + 2
    panel["gridPos"]["y"] = max_y + 1

    panels.append(sep)
    panels.append(panel)
    dash["panels"] = panels

    # ── Bump dashboard version ────────────────────────────────────────────────
    dash["version"] = dash.get("version", 1) + 1

    with open(out_path, "w") as f:
        json.dump(dash, f, indent=2)

    print("Dashboard patched successfully.")
    print("  Added row:   '{0}' (id={1})".format(sep["title"],   sep["id"]))
    print("  Added panel: '{0}' (id={1})".format(panel["title"], panel["id"]))
    print("  New version: {0}".format(dash["version"]))
    print("  Output:      {0}".format(out_path))
    print()
    print("Next step: reload Grafana")
    print("  docker compose -f ~/neuroedgeflow-monitoring/docker-compose.yml restart grafana")
    print("  Then open http://localhost:3000 and import the updated dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", required=True,
                        help="Path to existing grafana_dashboard.json")
    parser.add_argument("--out", required=True,
                        help="Output path (can be same as --dashboard to patch in place)")
    args = parser.parse_args()
    patch(args.dashboard, args.out)
