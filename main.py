from torchvision.io.image import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.utils import draw_bounding_boxes, save_image
from util import LABEL_DICT

if __name__ == "__main__":
    source_img = read_image("/home/ordinaryhuman/sample.png")
    img = (source_img.clone() / 255).unsqueeze(0)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=0.9
    )
    model.eval()
    outputs = model(img)
    for i, output in enumerate(outputs):
        boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
        labels = [
            LABEL_DICT[label] + str(score)[:4]
            for label, score in zip(labels.tolist(), scores.tolist())
        ]
        result = draw_bounding_boxes(source_img, boxes.detach(), labels) / 255
        save_image(result, f"/home/ordinaryhuman/result_{i}.png")
    # (320, 60), (430, 110)
