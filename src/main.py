"""Demonstration for PAOT."""
from typing import Tuple
from torchvision.io.image import read_image

# Support the following models.
from torchvision.models.detection import *
from torchvision.utils import draw_bounding_boxes, save_image
from util import COCO_2017_LABEL_DICT
from attack import *
import torch
from torch import nn


"""Supports all models in PyTorch's torchvision.models.detection package.
model = fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)
model = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)
model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
)
model = fcos_resnet50_fpn(
    weights=FCOS_ResNet50_FPN_Weights.DEFAULT
)
model = keypointrcnn_resnet50_fpn(
    weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
)
model = maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
)
model = maskrcnn_resnet50_fpn_v2(
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
)
model = retinanet_resnet50_fpn(
    weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT
)
model = retinanet_resnet50_fpn_v2(
    weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
)
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
)  # This model uses batchnorm, which need batchsize > 1
"""


ASSETS_ROOT = "../assets"
DEVICE = torch.device("cuda:0")
SAMPLE_BOUND = (60, 320, 110, 430)  # You can set a desired region to perform attack.
SETTINGS = {
    "bounded_targeted": {
        "save_path": f"{ASSETS_ROOT}/experiments/bounded/targeted/",
        "eps": 255 / 255,
        "y": [
            {
                "boxes": torch.Tensor(
                    [[330, 70, 420, 100]],
                ).to(DEVICE),
                "labels": torch.Tensor([1]).to(DEVICE).to(torch.int64),
            }
        ],
        "bound": (60, 320, 110, 430),
        "target_losses": [
            "loss_classifier",
            "loss_objectness",
            "loss_box_reg",
            "loss_rpn_box_reg",
        ],
    },
    "bounded_untarget": {
        "save_path": f"{ASSETS_ROOT}/experiments/bounded/untarget/",
        "eps": 255 / 255,
        "bound": (60, 320, 110, 430),
        "target_losses": [
            "loss_classifier",
            "loss_objectness",
            "loss_box_reg",
            "loss_rpn_box_reg",
        ],
    },
    "unbounded_targeted_box": {
        "save_path": f"{ASSETS_ROOT}/experiments/unbounded/box/targeted",
        "eps": 16 / 255,
        "y": [
            {
                "boxes": torch.Tensor(
                    [[400, 0, 563, 1024], [200, 200, 400, 400], [400, 400, 563, 1024]]
                ).to(DEVICE),
                "labels": torch.Tensor([24, 24, 24]).to(DEVICE).to(torch.int64),
            }
        ],
        "target_losses": [
            "loss_box_reg",
            "loss_rpn_box_reg",
        ],
    },
    "unbounded_targeted_cls": {
        "save_path": f"{ASSETS_ROOT}/experiments/unbounded/cls/targeted",
        "eps": 16 / 255,
        "y": [
            {
                "boxes": torch.Tensor(
                    [[400, 0, 563, 1024], [200, 200, 400, 400], [400, 400, 563, 1024]]
                ).to(DEVICE),
                "labels": torch.Tensor([24, 24, 24]).to(DEVICE).to(torch.int64),
            }
        ],
        "target_losses": [
            "loss_classifier",
            "loss_objectness",
        ],
    },
    "unbounded_untarget_box": {
        "save_path": f"{ASSETS_ROOT}/experiments/unbounded/box/untarget",
        "eps": 16 / 255,
        "target_losses": [
            "loss_box_reg",
            "loss_rpn_box_reg",
        ],
    },
    "unbounded_untarget_cls": {
        "save_path": f"{ASSETS_ROOT}/experiments/unbounded/cls/untarget",
        "eps": 16 / 255,
        "target_losses": [
            "loss_classifier",
            "loss_objectness",
        ],
    },
    "standard_targeted": {
        "save_path": f"{ASSETS_ROOT}/experiments/standard/targeted",
        "eps": 4 / 255,
        "y": [
            {
                "boxes": torch.Tensor(
                    [[400, 0, 563, 1024], [200, 200, 400, 400], [400, 400, 563, 1024]]
                ).to(DEVICE),
                "labels": torch.Tensor([24, 24, 24]).to(DEVICE).to(torch.int64),
            }
        ],
    },
    "standard_untarget": {
        "save_path": f"{ASSETS_ROOT}/experiments/standard/untarget",
        "eps": 4 / 255,
    },
}


def get_image(
    path: str = f"{ASSETS_ROOT}/sample.png",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch a single image for demo.

    :param path: Path to the image.
    :return: returns image tensors in int8 and float32 format.
    """
    source_img = read_image(path=path)
    img = (source_img.clone() / 255).unsqueeze(0)  # Form a size-1 mini-batch.
    return source_img.to(DEVICE), img.to(DEVICE)


def main(
    model: nn.Module,
    eps: float,
    save_path: str,
    y: Optional[Any] = None,
    target_losses: Optional[List[str]] = None,
    bound: Optional[Any] = None,
) -> None:
    """Main entry for demo."""
    source_img, img = get_image()

    model = model.eval().to(DEVICE)
    outputs = model(img)
    output = outputs[0]
    print("Before attack:\n", output)

    attack = ObjectDetectionLinfPGD(
        model=model,
        eps=eps,
        nb_iter=100,
    )

    attacked_img = attack(
        x=img,
        y=y,
        target_losses=target_losses,
        bound=bound,
    )

    outputs = model(attacked_img)
    output = outputs[0]
    print("After attack:\n", output)

    boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
    labels = [
        COCO_2017_LABEL_DICT[label] + str(score)[:4]
        for label, score in zip(labels.tolist(), scores.tolist())
    ]
    result = draw_bounding_boxes(
        (attacked_img.squeeze(0) * 255).to(torch.uint8),
        boxes.detach(),
        labels,
        # Use these if you want to customize font & color
        # colors=["#0080FF" for _ in range(len(labels))],
        # font=f"{ASSETS_ROOT}/16020_FUTURAM.ttf",
        # font_size=24,
    )
    result = result / 255

    save_image(attacked_img, f"{save_path}/attacked.png")
    save_image(result, f"{save_path}/attacked_result.png")
    save_image((attacked_img - img) ** 2, f"{save_path}/diff.png")


if __name__ == "__main__":
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.9
    )
    for _, setting in SETTINGS.items():
        print(f"Running {setting['save_path']}")
        main(model, **setting)
