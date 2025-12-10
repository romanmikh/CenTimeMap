# test_gradcam_3D.py
import torch
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models.resnet import resnet50
import torch.nn as nn

from src.dataset.dummy_dataset import DummyCTDataset
from src.visualisation.cam import CamInterp

from torchvision.models import resnet50, ResNet50_Weights

weights     = ResNet50_Weights.DEFAULT
preprocess  = weights.transforms(antialias=True)
MODEL       = resnet50(weights=weights)
CATEGORIES  = weights.meta["categories"]

def get_mid_slice():
    """get a centred 2-D slice from a synthetic 3-D CT """
    vol = DummyCTDataset(n_samples=1, random_spheres=False, use_spheres=True, feat_frac=1.0)[0]["img"][0]  # (Z,Y,X)
    sl  = vol[vol.shape[0] // 2]                                               # (Y,X)
    sl  = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)                       # normialise
    return sl.repeat(3, 1, 1)                                                  # (3,Y,X)


X = preprocess(get_mid_slice().unsqueeze(0))                                   # (1,3,224,224)


def test_gradcam_3d_target_layer():
    target = CamInterp(model=MODEL, cam_interp_method="gradcam").get_target_layer()
    assert isinstance(target, (nn.Conv2d, nn.Conv3d, nn.LayerNorm))


def test_gradcam_3d_output():
    cam    = CamInterp(model=MODEL, cam_interp_method="gradcam")
    heat, pred = cam.interpret(X)                                              # heat: (1,224,224)
    assert heat.shape == (1, 224, 224)

    cls = CATEGORIES[pred.squeeze(0).argmax().item()]
    print("Predicted class:", cls)

    rgb  = X.squeeze(0).permute(1, 2, 0).numpy()
    rgb = rgb - rgb.min()
    rgb = (rgb / (rgb.max() + 1e-6)).astype("float32")

    plt.figure(); plt.imshow(rgb); plt.axis("off")
    # plt.savefig("test_cam3D_dummy_slice.png")


    vis = show_cam_on_image(rgb, heat[0], use_rgb=True)
    plt.imshow(vis); plt.axis("off")
    plt.savefig("results/transformer_viz/resnet_grads/test_cam_resnet_grads_slice.png")

if __name__ == "__main__":
    test_gradcam_3d_target_layer()
    test_gradcam_3d_output()
