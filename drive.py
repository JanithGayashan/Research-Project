"""
drive.py
========
Live bridge to the Udacity Self-Driving Car Simulator (Socket.IO/Flask/
Eventlet), for closed-loop demonstration AND Phase 3 telemetry logging.

FIXES APPLIED vs. the original codebase:
- The GABNet class is no longer duplicated here; it is imported from
  models.py via build_model(). This eliminates the risk of the local
  copy silently drifting out of sync with the canonical architecture.
- --model / --seed are CLI arguments (any of pilotnet/resnet18/cbam/
  soft_attn/gabnet), so this single script can drive ANY of the 5
  models for a fair Phase-3 comparison, instead of being hard-wired to
  one GAB-Net checkpoint.
- The steering gain is a single documented Config.STEERING_GAIN applied
  IDENTICALLY regardless of which model is loaded (previously a
  hard-coded `* 1.2` was applied only in the GAB-Net script, which would
  have made any closed-loop stability comparison against other models
  unfair had they been added later).
- Every frame's telemetry (steering command, throttle, speed, elapsed
  time) is now logged to a CSV under Config.TELEMETRY_LOG_DIR, tagged
  with --track_label, so Phase-3 distribution-shift runs (e.g.
  "training_track" vs "jungle_track") produce comparable, persisted
  data instead of only a live cv2 window.
- Only intrinsic models (cbam/soft_attn/gabnet) draw a live heatmap
  overlay; black-box models (pilotnet/resnet18) show the raw feed only,
  since they have no attention map without a Grad-CAM pass (which is too
  slow to run per-frame in a real-time control loop).
"""

import argparse
import base64
import csv
import os
import time
from io import BytesIO

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from PIL import Image
import torch
from torchvision import transforms
import cv2

from config import Config
from models import build_model
from utils import load_checkpoint_into


def parse_args():
    parser = argparse.ArgumentParser(description="GAB-Net closed-loop simulator bridge")
    parser.add_argument("--model", default="gabnet", choices=list(Config.MODEL_KEYS.keys()),
                         help="Which model to drive with.")
    parser.add_argument("--seed", type=int, default=Config.SEEDS[0],
                         help="Which trained seed's checkpoint to load.")
    parser.add_argument("--track_label", default="unlabeled_track",
                         help="Free-text label for this run, e.g. 'training_track' "
                              "or 'jungle_track' (you must select the actual track "
                              "manually in the simulator GUI).")
    parser.add_argument("--headless", action="store_true",
                         help="Disable the live cv2 visualization window.")
    return parser.parse_args()


def build_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TelemetryRecorder:
    def __init__(self, model_key, seed, track_label):
        os.makedirs(Config.TELEMETRY_LOG_DIR, exist_ok=True)
        fname = f"{Config.MODEL_KEYS[model_key]}_seed_{seed}_{track_label}_{int(time.time())}.csv"
        self.path = os.path.join(Config.TELEMETRY_LOG_DIR, fname)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame", "timestamp", "steering_angle", "throttle", "speed"])
        self._frame_idx = 0
        print(f"[TELEMETRY] Logging to {self.path}")

    def log(self, steering_angle, throttle, speed):
        self._writer.writerow([self._frame_idx, time.time(), steering_angle, throttle, speed])
        self._frame_idx += 1
        if self._frame_idx % 50 == 0:
            self._file.flush()

    def close(self):
        self._file.close()


def main():
    args = parse_args()
    device = torch.device("cpu")  # CPU for local real-time inference (avoids GPU-transfer lag)

    model = build_model(args.model).to(device)
    load_checkpoint_into(model, args.model, args.seed, Config, map_location=device)
    model.eval()

    transform = build_transform()
    is_intrinsic = args.model in Config.INTRINSIC_MODEL_KEYS
    recorder = TelemetryRecorder(args.model, args.seed, args.track_label)

    sio = socketio.Server()
    app = Flask(__name__)

    def send_control(steering_angle, throttle):
        sio.emit("steer", data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        }, skip_sid=True)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if not data:
            sio.emit("manual", data={}, skip_sid=True)
            return

        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            steering_pred, attn_map = model(img_tensor)

        # Same gain applied regardless of model choice -- fair comparison.
        final_steer = float(steering_pred.item()) * Config.STEERING_GAIN
        throttle = Config.DEFAULT_THROTTLE
        speed = float(data.get("speed", 0.0))

        recorder.log(final_steer, throttle, speed)

        if not args.headless:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            if is_intrinsic and attn_map is not None:
                mask = attn_map.squeeze().detach().cpu().numpy()
                mask = cv2.resize(mask, (cv_img.shape[1], cv_img.shape[0]))
                heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
                viz = cv2.addWeighted(cv_img, 0.6, heatmap, 0.4, 0)
            else:
                viz = cv_img
            cv2.imshow(f"{Config.MODEL_KEYS[args.model]}: LIVE VIEW", viz)
            cv2.waitKey(1)

        send_control(final_steer, throttle)
        print(f"Steer: {final_steer:.3f} | Speed: {speed:.2f}", end="\r")

    @sio.on("connect")
    def connect(sid, environ):
        print("\n[SIM] Connected.")
        send_control(0, 0)

    print(f"[BRIDGE] Model={Config.MODEL_KEYS[args.model]} | Seed={args.seed} | "
          f"Track label='{args.track_label}' | Port={Config.TELEMETRY_PORT}")
    wsgi_app = socketio.Middleware(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(("", Config.TELEMETRY_PORT)), wsgi_app)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
