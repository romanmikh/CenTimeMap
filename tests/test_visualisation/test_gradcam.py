# currently crashes pytest due to hard-coded path

# pylint: disable=no-member, no-name-in-module

import torch
from cv2 import imread
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.vgg import vgg16
from torchvision.models.vision_transformer import vit_b_16, vit_l_32

from src.visualisation.cam import CamInterp

model_name = "resnet50"
weight = list(torch.hub.load("pytorch/vision", "get_model_weights", name=model_name))  # type: ignore
weight = weight[0]
preprocess = weight.transforms(antialias=True)  # type: ignore
cats = weight.meta["categories"]  # type: ignore
tgt_class = cats.index("tench")
MODEL = resnet50(weights=weight)
# https://media.npr.org/assets/img/2014/11/11/lung-scan-7069159ae64a5dee3c77301d97af91f4cb3918a1.jpg
X = imread("/home/rocky/Downloads/lung_dummy.png") / 255.0
X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0).float()
X = preprocess(X)


def test_gradcam_target_layer():
    interp = CamInterp(model=MODEL, cam_interp_method="kpcacam")
    target_layer = interp.get_target_layer()

    assert isinstance(
        target_layer, (nn.Conv2d, nn.Conv3d, nn.LayerNorm)
    ), "Invalid target layer"


def test_gradcam_output():
    interp = CamInterp(model=MODEL, cam_interp_method="kpcacam")
    output, pred = interp.interpret(X)

    pred = pred.squeeze(0).argmax().item()
    pred = cats[pred]
    print(f"Predicted class: {pred}")

    assert output.shape == (1, 224, 224), "Invalid output shape"
    rgb_img = X.squeeze(0).permute(1, 2, 0).numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    vis = show_cam_on_image(rgb_img, output[0], use_rgb=True)
    plt.imshow(vis)
    plt.savefig("gradcam_output.png")
    # plt.show()

if __name__ == "__main__":
    test_gradcam_target_layer()
    test_gradcam_output()
