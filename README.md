# Adversarial-Composition
Compose desired image with data such that will cause pretrained models misbehave.

## FasterRCNN
- Model in eval
    [
        {
            "boxes": tensor(
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
                device="cuda:0",
            ),
            "labels": tensor([1, 3, 3, 3, 1, 1, 28, 1, 1, 1, 1, 31], device="cuda:0"),
            "scores": tensor(
                [
                    0.9997,
                    0.9987,
                    0.9964,
                    0.9496,
                    0.9340,
                    0.9179,
                    0.9155,
                    0.8535,
                    0.8017,
                    0.7775,
                    0.6573,
                    0.6002,
                ],
                device="cuda:0",
            ),
        }
    ]

- Model in train
    {
        "loss_classifier": torch.Tensor(0.2334, device="cuda:0"), # 分類
        "loss_box_reg": torch.Tensor(0.2486, device="cuda:0"), # 框
        "loss_objectness": torch.Tensor(0.1513, device="cuda:0"), # 有東西的機率
        "loss_rpn_box_reg": torch.Tensor(0.0271, device="cuda:0"), # 
    }