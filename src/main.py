import supervisely as sly
from typing_extensions import Literal
from typing import List, Any, Dict, Optional
import warnings

warnings.filterwarnings("ignore")
import torch
from dotenv import load_dotenv
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import numpy as np
import os
from src.keypoints_template import template

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

weights_url = "https://4mizfq.sn.files.1drv.com/y4mmN4HVKiAoyjCvPyKAWSK2Tkv5UaooeY2XmcUdxRwftMfZZ35N2kOIeyvgHzCiB2wW6yhYBdjU_nsoa2eHkSE7iWL903bTmUPrFWR3U5fPeMEXWOLVZwN2HaD-JRETuuDiLF249A_zeR3ZyxCLjnF4svHU2RLo3lgy918r59l5yA5UBrOCIE2-KpUFiF3nFo8Ae4Hf8ybzWYv7t7mbwotTQ"

class MyModel(sly.nn.inference.PoseEstimation):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu"
    ):
        # download model weights
        dst_weights_path = f"{model_dir}/vitpose-b.pth"
        if not os.path.exists(dst_weights_path):
            self.download(weights_url, dst_weights_path)
        # define model config and checkpoint
        pose_config = os.path.join(model_dir, "pose_config.py")
        pose_checkpoint = os.path.join(model_dir, "vitpose-b.pth")
        # buid model
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        # define class names
        self.class_names = ["person_keypoints"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")
        
    def get_classes(self) -> List[str]:
        return self.class_names  # e.g. ["cat", "dog", ...]
    
    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionKeypoints]:
        # transfer crop from annotation tool to bounding box
        input_image = sly.image.read(image_path)
        img_height, img_width = input_image.shape[:2]
        bbox = [{"bbox": np.array([0, 0, img_width, img_height, 1.0])}]

        # get point labels
        point_labels = self.keypoints_template.point_names

        # inference pose estimator
        if "local_bboxes" in settings:
            bboxes = settings["local_bboxes"]
        elif "detected_bboxes" in settings:
            bboxes = settings["detected_bboxes"]
            for i in range(len(bboxes)):
                box = bboxes[i]["bbox"]
                bboxes[i] = {"bbox": np.array(box)}
        else:
            bboxes = bbox

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image_path,
            bboxes,
            format="xyxy",
            dataset=self.pose_model.cfg.data.test.type,
        )

        # postprocess results
        point_threshold = settings.get("point_threshold", 0.01)
        results = []
        for result in pose_results:
            included_labels, included_point_coordinates = [], []
            point_coordinates, point_scores = result["keypoints"][:, :2], result["keypoints"][:, 2]
            for i, (point_coordinate, point_score) in enumerate(
                zip(point_coordinates, point_scores)
            ):
                if point_score >= point_threshold:
                    included_labels.append(point_labels[i])
                    included_point_coordinates.append(point_coordinate)
            results.append(
                sly.nn.PredictionKeypoints(
                    "person_keypoints", included_labels, included_point_coordinates
                )
            )
        return results
    
model_dir = "my_model"  # model weights will be downloaded into this dir
settings = {"point_threshold": 0.1}

if not sly.is_production():
    # proposal bboxes are hardcoded for the example image.
    local_bboxes = [
        {"bbox": np.array([245, 72, 411, 375, 1.0])},
        {"bbox": np.array([450, 204, 633, 419, 1.0])},
        {"bbox": np.array([35, 69, 69, 164, 1.0])},
        {"bbox": np.array([551, 99, 604, 216, 1.0])},
        {"bbox": np.array([440, 72, 458, 106, 1.0])},
    ]
    settings["local_bboxes"] = local_bboxes

m = MyModel(
    model_dir=model_dir,
    custom_inference_settings=settings,
    keypoints_template=template,
)

m.load_on_device(model_dir=model_dir, device=device)

if sly.is_production():
    # this code block is running on Supervisely platform in production
    # just ignore it during development
    m.serve()
else:
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings)

    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=2)
    print(f"Predictions and visualization have been saved: {vis_path}")