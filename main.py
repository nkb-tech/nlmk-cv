import sys
import numpy as np
import cv2
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import pytz
import logging
from queue import Empty, Queue
from threading import Thread, Event

from src.dataloader import StreamLoader
from cranpose.core.estimators import PoseSingle, PoseMultiple
from cranpose.core.utils import ARUCO_DICT
from src.detector import Detector
from src.config import Config

TIMEZONE = pytz.timezone('Europe/Moscow') # UTC, Asia/Shanghai, Europe/Berlin
def timetz(*args):
    return datetime.now(TIMEZONE).timetuple()
logging.Formatter.converter = timetz


class Worker():
    _TIMEOUT = 2
    
    def __init__(self, cfg: dict, cams_cfg: dict, debug: bool = False):
        self.cfg = cfg
        
        # Init separate process
        self.queue = Queue(maxsize=30)
        self.done = Event()
        self.pool = Thread(target=self.worker, daemon=True)
        
        # Streams
        logging.info(f'Initializing stream loader...')
        self.cams_cfg = Config(cams_cfg)
        self.dataloader = StreamLoader(self.cams_cfg.links, vid_fps=cfg['fps'])
        logging.info(f'Stream loader initialized')
        # Models
        # TODO add zones
        self.detector = Detector(cfg['detector'], self.cams_cfg)
        logging.info(f'People Detector initialized')
        # Crane positioning
        self.pose = self._init_pose_estimator(cfg['pose_estimator'], self.cams_cfg)
        logging.info(f'Pose Estimator initialized')
        # Sender info
        # TODO cooperate with another team
        
        # Debug
        self.debug = debug
        if self.debug:
            self.cfg['debug']['save_img_path'] = Path(self.cfg['debug']['save_img_path']) / \
                                                      datetime.now().isoformat('T', 'seconds').replace(':', '_')
            logging.info(f"Debug mode: ON, saving data to {self.cfg['debug']['save_img_path']}")
        else:
            logging.info(f"Debug mode: OFF")
            
        self.pool.start()
    
    def _init_pose_estimator(self, cfg, cams_cfg):
        k = np.load(cfg['calibration_matrix_path'])
        d = np.load(cfg['distortion_coefficients_path'])
        aruco_dict_type = ARUCO_DICT[cfg['aruco_dict_type']]
        crane_cams = cams_cfg.pose_cams
        pose_estimators = {}
        for crane_idx, cam_idxs in crane_cams.items():
            pose_estimators[crane_idx] = [PoseMultiple(
                PoseSingle(
                    aruco_dict_type=aruco_dict_type,
                    camera_orientation=1,
                    n_markers=cfg['n_markers'],
                    marker_step=cfg['marker_step'],
                    marker_edge_len=cfg['edge_len'],
                    matrix_coefficients=k,
                    distortion_coefficients=d,
                ),
                PoseSingle(
                    aruco_dict_type=aruco_dict_type,
                    camera_orientation=-1,
                    n_markers=cfg['n_markers'],
                    marker_step=cfg['marker_step'],
                    marker_edge_len=cfg['edge_len'],
                    matrix_coefficients=k,
                    distortion_coefficients=d,
                )
            ), cam_idxs]
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
            self.log_debug(
                imgs, poses, dets, timestamp
            )
        self.send_results(poses, dets, timestamp)

    def get_poses(self, imgs):
        res = {}
        for crane_idx, pose_info in self.pose.items():
            est = pose_info[0]
            pose_cam_idx_rear = pose_info[1][1]
            pose_cam_idx_front = pose_info[1][0]
            coords = est(imgs[pose_cam_idx_front], imgs[pose_cam_idx_rear])[0, 3]
            res[crane_idx] = coords
        return res
            

    def send_results(self, bs64, cam, dogs_params, timestamp):
        # TODO cooperate with another team
        logging.info(f'Found {len(dogs_params)} dogs')
        
    def log_debug(self, imgs, poses, dets, timestamp):
        save_img_path = self.cfg['debug']['save_img_path']
        (save_img_path / 'images').mkdir(exist_ok=True, parents=True)
        (save_img_path / 'labels').mkdir(exist_ok=True, parents=True)
        (save_img_path / 'poses').mkdir(exist_ok=True, parents=True)
        timestamp_str = timestamp.isoformat('T', 'seconds').replace(':', '_')
        for i, (img, det) in enumerate(zip(imgs, dets)):
            boxes = det[:, :4].astype(int)
            for box in boxes:
                img = cv2.rectangle(img, 
                                    pt1=box[:2], 
                                    pt2=box[2:], 
                                    color=(0, 0, 255), 
                                    thickness=2)
            cv2.imwrite(str(save_img_path / 'images' / f'{self.cams_cfg.cam_names[i]}_{timestamp_str}.jpg'), img)
            labels_str = [' '.join([str(i) for i in lb]) + '\n' for lb in det]
            with (save_img_path / 'labels' / f'{self.cams_cfg.cam_names[i]}_{timestamp_str}.txt').open('w') as f:
                f.writelines(labels_str)
        poses_str = '\n'.join([f'{k}:{v:.2f}' for k, v in poses.items()])
        with (save_img_path / 'poses' / f'{timestamp_str}.txt').open('w') as f:
            f.write(poses_str)

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, help='PetSearch config path')
    parser.add_argument('-cam', type=str, help='Camera config path')
    parser.add_argument('-log', '--log_path', type=str, help='Logging path')
    parser.add_argument('-d', '--debug', action='store_true', 
                        help='Debug mode, that save images and predictions')
    args = parser.parse_args()
    cfg_path = args.cfg
    cams = args.cam
    log_path = args.log_path
    debug = args.debug
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.DEBUG, 
                        filename=f"{log_path}/{TIMEZONE.localize(datetime.now()).isoformat('T', 'seconds').replace(':', '_')}_logs.log",
                        filemode="w",
                        format="%(asctime)s %(levelname)s %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(cams, 'r') as f:
        cams_cfg = yaml.safe_load(f)
    worker = Worker(cfg, cams_cfg, debug)
    worker.run()

if __name__ == '__main__':
    main()