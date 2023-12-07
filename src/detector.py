import logging
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg, zones):
        # TODO add zones
        self.model = YOLO(model=cfg["model_path"], task="detect")
        self.device = torch.device(cfg["device"])
        self.cfg = cfg
        # Dummy inference for model warmup
        for _ in range(100):
            dummy_imgs = [
                np.random.randint(
                    low=0,
                    high=255,
                    size=(cfg["orig_img_h"], cfg["orig_img_w"], 3),
                    dtype=np.uint8,
                )
                for _ in range(cfg["inference_bs"])
            ]
            self.model(
                source=dummy_imgs,
                device=self.device,
                imgsz=cfg["inference_imgsz"],
                conf=cfg["inference_conf"],
                stream=False,
                verbose=False,
                half=True,
            )
        self.time_logging_period = cfg["time_logging_period"]
        self.n_calls = -1
        self.zones = zones

    def __call__(self, imgs: list) -> Any:
        self.n_calls += 1
        return self.inference(imgs)

    def inference(self, imgs: list):
        start_time_ns = perf_counter_ns()
        for i in range(len(imgs)):
            h, w, _ = imgs[i].shape
            if (h, w) != (
                self.cfg["orig_img_h"],
                self.cfg["orig_img_w"],
            ):
                imgs[i] = cv2.resize(
                    imgs[i],
                    (self.cfg["orig_img_w"], self.cfg["orig_img_h"]),
                    # Default YOLO interpolation
                    interpolation=cv2.INTER_AREA,
                )

        results = self.model(
            source=imgs,
            device=self.device,
            imgsz=self.cfg["inference_imgsz"],
            conf=self.cfg["inference_conf"],
            stream=False,
            verbose=False,
            half=True,
        )

        sv_dets = [
            sv.Detections.from_ultralytics(res) for res in results
        ]
        detection_zones = []
        for i, zones in self.zones.items():
            if len(sv_dets[i]) == 0:
                detection_zones.append([])
                continue
            mask = np.array(["normal "] * len(sv_dets[i]), dtype=str)
            for zone_type, zone in zones.items():
                zone_triggers = zone.trigger(sv_dets[i])
                mask[zone_triggers] = zone_type
            detection_zones.append(mask)
        dets = [
            {
                "bboxes": detection.boxes.xyxyn.cpu().numpy(),
                "zones": detection_zone,
            }
            for detection, detection_zone in zip(
                results, detection_zones
            )
        ]
        end_time_ns = perf_counter_ns()
        time_spent_ns = end_time_ns - start_time_ns
        time_spent_ms = time_spent_ns / 1e6
        if self.n_calls % self.time_logging_period == 0:
            logging.info(
                f"Detector inference on {len(imgs)} images took {time_spent_ms:.1f} ms"
            )
        return dets
