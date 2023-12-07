import logging
from collections import defaultdict

import pandas as pd


class Config:
    def __init__(self, yaml_cfg: dict):
        # with open(yaml_cfg, "r") as f:
        #     self.cfg = yaml.safe_load(f)
        self.cfg = yaml_cfg
        self.links = [c["link"] for c in self.cfg.values()]
        self.cam_names = [c for c in self.cfg.keys()]
        cams = {
            "cam_id": [],
            "pose": [],
            "crane_id": [],
            "side": [],
            "people": [],
            "poly_zone": [],
            "link": [],
            "frame_size": [],
        }
        for cam_id, args in self.cfg.items():
            cams["cam_id"].append(cam_id)
            cams["pose"].append("pose" in args["task"])
            cams["crane_id"].append(args.get("crane_id", None))
            cams["side"].append(args.get("side", None))
            cams["people"].append("ppl" in args["task"])
            cams["poly_zone"].append(args.get("detection_zones", None))
            cams["link"].append(args["link"])
            cams["frame_size"].append(args["frame_size"])
        self.cams = pd.DataFrame(cams)
        # Crane pose cameras (indexing): {crane_idx : (front_side_cam_idx, rear_side_cam_idx)} for each crame
        self.pose_cams = defaultdict(lambda: [None, None])
        for i, row in self.cams[self.cams.pose == True].iterrows():
            if row["side"] == 1:
                self.pose_cams[row["crane_id"]][0] = i
            else:
                self.pose_cams[row["crane_id"]][1] = i
        self.pose_cams = dict(self.pose_cams)
        self.ppl_cams = []
        self.zones = {}
        for i, row in self.cams[self.cams.people == True].iterrows():
            self.ppl_cams.append(i)
            zones = row.poly_zone
            if zones is None:
                logging.error(
                    f"No zones for camera {row.cam_id}, set zones!!!"
                )
                raise ValueError()
            self.zones[i] = zones
        self.frame_size = {
            i: row.frame_size for i, row in self.cams.iterrows()
        }
