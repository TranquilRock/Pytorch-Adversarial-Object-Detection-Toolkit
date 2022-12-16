from typing import Dict, List, Tuple
from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    ssdlite320_mobilenet_v3_large
)
from torchvision.utils import draw_bounding_boxes, save_image
from util import LABEL_DICT
from attack.gradient_based import FasterRCNNLinfPGDWithBound
import torch
from torch import nn

ROOT = "/home/ordinaryhuman/Adversarial-Composition/assets"  # "./assets"
DEVICE = torch.device("cuda:0")


def get_image(path: str = f"{ROOT}/sample.png") -> Tuple[torch.Tensor, torch.Tensor]:
    source_img = read_image(path=path)
    img = (source_img.clone() / 255).unsqueeze(0)
    return source_img.to(DEVICE), img.to(DEVICE)

def main() -> None:
    source_img, img = get_image()
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=0.8,
    )
    model = model.eval().to(DEVICE)


    attack = FasterRCNNLinfPGDWithBound(
        model=model,
        eps=0.1,
        nb_iter=30,
        # eps_iter=0.001,
    )
    bound = (0, 0, img.size(2), img.size(3))
    # bound = (60, 320, 110, 430)
    attacked_img = attack(x=img, bound=bound)
    save_image(attacked_img, f"{ROOT}/attacked.png")

    outputs = model(attacked_img)
    for i, output in enumerate(outputs):
        print(output)
        boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
        labels = [
            LABEL_DICT[label] + str(score)[:4]
            for label, score in zip(labels.tolist(), scores.tolist())
        ]
        result = draw_bounding_boxes(source_img, boxes.detach(), labels) / 255
        save_image(result, f"{ROOT}/attacked_result.png")
    save_image((attacked_img - img) ** 2, f"{ROOT}/diff.png")


if __name__ == "__main__":
    main()
