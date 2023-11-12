from collections import defaultdict

import pandas as pd
import yaml


class Config:
    def __init__(self, yaml_cfg):
        with open(yaml_cfg, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.links = [c["link"] for c in self.cfg.values()]
        self.cam_names = [c["cam_id"] for c in self.cfg.values()]
        cams = {
            "cam_id": [],
            "pose": [],
            "crane_id": [],
            "side": [],
            "people": [],
            "poly_zone": [],
            "link": [],
        }
        for cam_id, args in self.cfg.items():
            cams["cam_id"].append(cam_id)
            cams["pose"].append("pose" in args["task"])
            cams["crane_id"].append(args.get("crane_id", None))
            cams["side"].append(args.get("side", None))
            cams["people"].append("ppl" in args["task"])
            cams["poly_zone"].append(args.get("detection_zone", None))
            cams["link"].append(args["link"])
        self.cams = pd.DataFrame(cams)
        # Crane pose cameras (indexing): {crane_idx : (front_side_cam_idx, rear_side_cam_idx)} for each crame
        self.pose_cams = defaultdict(lambda: [None, None])
        for i, row in self.cams[self.cams.pose == True].iterrows():
            if row["side"] == 1:
                self.pose_cams[row["crane_id"]][0] = i
            else:
                self.pose_cams[row["crane_id"]][1] = i
        self.pose_cams = dict(self.pose_cams)
