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
SAVE_ROOT = "../assets/bounded"
DEVICE = torch.device("cuda:0")
SAMPLE_BOUND = (60, 320, 110, 430)  # You can set a desired region to perform attack.


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


def main(model: nn.Module) -> None:
    """Main entry for demo."""
    source_img, img = get_image()

    model = model.eval().to(DEVICE)
    outputs = model(img)
    output = outputs[0]
    print("Before attack:\n", output)

    attack = ObjectDetectionLinfPGD(
        model=model,
        eps=255 / 255,
        nb_iter=1000,
    )

    attacked_img = attack(
        x=img,
        y=[
            {
                "boxes": torch.Tensor(
                    [[400, 0, 563, 1024], [200, 200, 400, 400], [400, 400, 563, 1024]]
                ).to(DEVICE),
                "labels": torch.Tensor([24, 24, 24]).to(DEVICE).to(torch.int64),
            }
        ],
        bound=SAMPLE_BOUND,
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
        # colors=["#0080FF" for _ in range(len(labels))],
        # font=f"{ASSETS_ROOT}/16020_FUTURAM.ttf",
        # font_size=24,
    )
    result = result / 255

    save_image(attacked_img, f"{SAVE_ROOT}/attacked.png")
    save_image(result, f"{SAVE_ROOT}/attacked_result.png")
    save_image((attacked_img - img) ** 2, f"{SAVE_ROOT}/diff.png")


if __name__ == "__main__":
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.9
    )
    main(model)
