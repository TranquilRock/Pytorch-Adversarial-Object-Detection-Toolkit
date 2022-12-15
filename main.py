from typing import Dict, List, Tuple
from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes, save_image
from util import LABEL_DICT
from attack.gradient_based import LinfPGDWithBound
import torch
from torch import nn

ROOT = "/home/ordinaryhuman/Adversarial-Composition/assets"  # "./assets"
DEVICE = torch.device("cuda:0")


def get_image(path: str = f"{ROOT}/sample.png") -> Tuple[torch.Tensor, torch.Tensor]:
    source_img = read_image(path=path)
    img = (source_img.clone() / 255).unsqueeze(0)
    return source_img.to(DEVICE), img.to(DEVICE)


class SumScoreLoss(nn.Module):
    def forward(self, x: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        return -torch.concat(x).sum()  # Minimize -score -> Maximize score.


def main() -> None:
    source_img, img = get_image()
    model = (
        fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.5
        )
        .train()
        # .eval()
        .to(DEVICE)
    )

    truth = [
        {
            "boxes": torch.Tensor(
                [
                    [415.1139, 150.9288, 545.6686, 414.7146],
                    [1.0540, 174.2773, 134.2004, 353.4934],
                    [95.4936, 180.1695, 235.8846, 291.5413],
                    [182.9897, 178.8918, 257.3154, 255.9231],
                    [416.5106, 185.4812, 445.5121, 261.6208],
                    [333.7903, 192.2578, 351.0151, 259.7337],
                    [437.7644, 292.8659, 501.6620, 405.6678],
                    [394.6197, 190.9600, 415.4427, 245.7377],
                    [282.3032, 151.4133, 310.8425, 211.5469],
                    [628.7678, 173.0963, 649.4948, 285.8481],
                    [704.1951, 171.3914, 735.0613, 258.0855],
                    [718.6891, 190.4808, 736.1738, 222.0931],
                ],
            ).to(DEVICE),
            "labels": torch.Tensor(
                [1, 3, 3, 3, 1, 1, 28, 1, 1, 1, 1, 31]
            ).to(torch.int64).to(DEVICE),
        }
    ]
    print(model(img, truth))  # , boxes = [0,0, 1,1]

    attack = LinfPGDWithBound(
        model=model,
        loss_fn=SumScoreLoss(),
        eps=1,
        nb_iter=1000,
        eps_iter=0.01,
    )

    attacked_img = attack(
        x=img,
        y=torch.zeros(1),
        bound=(
            60,
            320,
            110,
            430,
        ),
        collate_fn=lambda outputs: [output["boxes"] for output in outputs],
    )
    save_image(attacked_img, f"{ROOT}/attacked.png")

    outputs = model(attacked_img)
    for i, output in enumerate(outputs):
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
