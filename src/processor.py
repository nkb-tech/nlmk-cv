import pandas as pd
import logging
from pathlib import Path
import requests as re

from ultralytics import YOLO

from dataloader import LoadStreams
from aruco import ArucoProcessor
from ppl_detection import PplDetector
from config import Config



class GeneralCamProcessor():
    '''
    General camera CV processor
    '''
    def __init__(self, cam_cfg: pd.DataFrame,
                 cfg: Config):
        self.list_streams = Path('./tmp_list.streams')
        with self.list_streams.open('w') as f:
            for i, cam in cam_cfg.iterrows():
                f.write(f'{cam.rtsp}\n')
        self.cfg = cfg
        self.dataloader = LoadStreams(sources=self.list_streams, vid_fps=self.cfg.vid_fps)
        self.aruco_processor = ArucoProcessor(self.cfg)
        self.ppl_model = PplDetector(self.cfg)
    
    def run(self):
        for imgs in self.dataloader():
            # TODO add img to tensor for batching, resizing etc.
            ppl_det_res = self.ppl_model(imgs)
            crane_posX = {}
            for crane in self.cfg.crane_ids:
                fr_cam_idx = self.cfg.get_crane_aruco_cam_idx(crane)
                crane_posX[crane] = self.aruco_processor(imgs[fr_cam_idx])
            res = self._convert_res(ppl_det_res, crane_posX)
            self.send_res(res)
    
    def _convert_res(self, ppl_det_res, crane_posX):
        pass
    
    def send_res(self, res):
        resp = re.post(self.cfg.send_url, json=res)
            
            