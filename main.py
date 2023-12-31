import argparse
import logging
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np
import pytz
import requests as re
import supervision as sv
import yaml

from cranpose.estimators import PoseMultiple, PoseSingle
from cranpose.utils import ARUCO_DICT
from src.config import Config
from src.dataloader import StreamLoader
from src.detector import Detector

TIMEZONE = pytz.timezone(
    "Europe/Moscow"
)  # UTC, Asia/Shanghai, Europe/Berlin


def timetz(*args):
    return datetime.now(TIMEZONE).timetuple()


logging.Formatter.converter = timetz


class Worker:
    _TIMEOUT = 2

    def __init__(self, cfg: dict, cams_cfg: dict, debug: bool = False):
        self.cfg = cfg

        # Init separate process
        self.queue = Queue(maxsize=30)
        self.done = Event()
        self.pool = Thread(target=self.worker, daemon=True)

        # Streams
        logging.info(f"Initializing stream loader...")
        self.cams_cfg = Config(cams_cfg)
        self.dataloader = StreamLoader(
            self.cams_cfg.links, vid_fps=cfg["fps"]
        )
        logging.info(f"Stream loader initialized")
        # Models
        self.zones = {
            k: {
                zone_type: sv.PolygonZone(
                    np.array(zone),
                    self.cams_cfg.frame_size[k],
                    sv.Position.BOTTOM_CENTER,
                )
                for zone_type, zone in zones.items()
            }
            for k, zones in self.cams_cfg.zones.items()
        }
        self.detector = Detector(cfg["detector"], self.zones)
        logging.info(f"People Detector initialized")
        # Crane positioning
        self.pose = self._init_pose_estimator(
            cfg["pose_estimator"], self.cams_cfg
        )
        logging.info(f"Pose Estimator initialized")
        # Sender info
        self.api_url = cfg["api_url"]
        self.cache_buffer = []
        # Debug
        self.debug = debug
        if self.debug:
            self.cfg["debug"]["save_img_path"] = Path(
                self.cfg["debug"]["save_img_path"]
            ) / datetime.now().isoformat("T", "seconds").replace(
                ":", "_"
            )
            logging.info(
                f"Debug mode: ON, saving data to {self.cfg['debug']['save_img_path']}"
            )
            save_img_path = self.cfg["debug"]["save_img_path"]
            (save_img_path / "images").mkdir(
                exist_ok=True, parents=True
            )
            (save_img_path / "labels").mkdir(
                exist_ok=True, parents=True
            )
            (save_img_path / "poses").mkdir(exist_ok=True, parents=True)
            self.save_img_path = save_img_path
        else:
            logging.info(f"Debug mode: OFF")

        self.pool.start()

    def _init_pose_estimator(self, cfg, cams_cfg):
        k = np.load(cfg["calibration_matrix_path"])
        d = np.load(cfg["distortion_coefficients_path"])
        aruco_dict_type = ARUCO_DICT[cfg["aruco_dict_type"]]
        crane_cams = cams_cfg.pose_cams
        pose_estimators = {}
        for crane_idx, cam_idxs in crane_cams.items():
            pose_estimators[crane_idx] = [
                PoseMultiple(
                    [  # TODO add biases for each single
                        PoseSingle(
                            aruco_dict_type=aruco_dict_type,
                            camera_orientation=1,
                            n_markers=cfg["n_markers"],
                            marker_poses=cfg["poses"],
                            marker_edge_len=cfg["edge_len"],
                            matrix_coefficients=k,
                            distortion_coefficients=d,
                        ),
                        PoseSingle(
                            aruco_dict_type=aruco_dict_type,
                            camera_orientation=-1,
                            n_markers=cfg["n_markers"],
                            marker_poses=cfg["poses"],
                            marker_edge_len=cfg["edge_len"],
                            matrix_coefficients=k,
                            distortion_coefficients=d,
                        ),
                    ]
                ),
                cam_idxs,
            ]
        return pose_estimators

    def worker(self) -> None:
        while not self.done.is_set():
            try:
                imgs = self.queue.get(
                    block=True,
                    timeout=self._TIMEOUT,
                )
                try:
                    self.run_on_images(imgs)
                except Exception as e:
                    print(e)
                finally:
                    self.queue.task_done()
            except Empty as e:
                pass
        return

    def __del__(self):
        self.pool.signal_exit()

    def run(self):
        for imgs in self.dataloader:
            self.queue.put(imgs)

    def run_on_images(self, imgs):
        timestamp = datetime.now(TIMEZONE)
        poses = self.get_poses(imgs)
        dets = self.detector(imgs)
        if self.debug:
            self.log_debug(imgs, poses, dets, timestamp)
        self.send_results(poses, dets, timestamp)

    def get_poses(self, imgs):
        res = {}
        for crane_idx, pose_info in self.pose.items():
            est = pose_info[0]
            pose_cam_idx_rear = pose_info[1][1]
            pose_cam_idx_front = pose_info[1][0]
            # TODO check None
            img_front = (
                imgs[pose_cam_idx_front]
                if pose_cam_idx_front is not None
                else None
            )
            img_rear = (
                imgs[pose_cam_idx_rear]
                if pose_cam_idx_rear is not None
                else None
            )
            coords = est([img_front, img_rear])[0, 3]
            res[crane_idx] = coords
        return res

    def send_results(self, poses, dets, timestamp):
        # TODO cooperate with another team
        timestamp = timestamp.isoformat()
        ppl_det_res = {}
        for cam_idx in self.cams_cfg.ppl_cams:
            ppl_det_res[self.cams_cfg.cam_names[cam_idx]] = [
                {
                    "bbox": bbox.tolist(),  # Attention on bboxes format (0-1), xtl, ytl, xbr, ybr
                    "zone": dets[cam_idx]["zones"][i].tolist(),
                    "timestamp": timestamp,
                }
                for i, bbox in enumerate(dets[cam_idx]["bboxes"])
            ]
        api_json = {
            "poses": [
                {
                    "name": crane_idx,
                    "x": pose_x,
                    "close_to": [],  # TODO
                    "timestamp": timestamp,
                }
                for crane_idx, pose_x in poses.items()
            ],
            "people": ppl_det_res,
        }
        self.cache_buffer.append(api_json)
        resp = re.post(
            self.api_url, json=self.cache_buffer, timeout=0.1
        )
        if resp.status_code == 200:
            self.cache_buffer = []
        if resp.status_code != 200:
            logging.critical(
                f"Bad api response code: {resp.status_code} with error: {resp.text}"
                "Caching results"
            )

    def log_debug(self, imgs, poses, dets, timestamp):
        timestamp_str = timestamp.isoformat("T", "seconds").replace(
            ":", "_"
        )
        for i, (img, det) in enumerate(zip(imgs, dets)):
            h, w, _ = img.shape
            boxes = det["bboxes"][:, :4]
            for box in boxes:
                xtl = int(box[0] * w)
                ytl = int(box[1] * h)
                xbr = int(box[2] * w)
                ybr = int(box[3] * h)
                img = cv2.rectangle(
                    img,
                    pt1=(xtl, ytl),
                    pt2=(xbr, ybr),
                    color=(0, 0, 255),
                    thickness=2,
                )
            cv2.imwrite(
                str(
                    self.save_img_path
                    / "images"
                    / f"{self.cams_cfg.cam_names[i]}_{timestamp_str}.jpg"
                ),
                img,
            )
            labels_str = [
                " ".join([str(i) for i in lb]) + z + "\n"
                for lb, z in zip(det["bboxes"], det["zones"])
            ]
            with (
                self.save_img_path
                / "labels"
                / f"{self.cams_cfg.cam_names[i]}_{timestamp_str}.txt"
            ).open("w") as f:
                f.writelines(labels_str)
        poses_str = "\n".join(
            [f"{k}:{v:.2f}" for k, v in poses.items()]
        )
        with (
            self.save_img_path / "poses" / f"{timestamp_str}.txt"
        ).open("w") as f:
            f.write(poses_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", type=str, help="PetSearch config path")
    parser.add_argument("-cam", type=str, help="Camera config path")
    parser.add_argument(
        "-log", "--log_path", type=str, help="Logging path"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode, that save images and predictions",
    )
    args = parser.parse_args()
    cfg_path = args.cfg
    cams = args.cam
    log_path = args.log_path
    debug = args.debug
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/{TIMEZONE.localize(datetime.now()).isoformat('T', 'seconds').replace(':', '_')}_logs.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(cams, "r") as f:
        cams_cfg = yaml.safe_load(f)
    worker = Worker(cfg, cams_cfg, debug)
    worker.run()


if __name__ == "__main__":
    main()
