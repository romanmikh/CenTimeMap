# CenTimeMap

CenTime is a deep learning framework for predicting life expectancy of patients with diseases such as IPF from CT scans, based on transformer architecture, first presented in ... . Here we present an improved model that naturally incorporates visualisation of the mortality of regions of affected tissue, additionally achieving a better performance on real data.

## Installation & Usage

```sh
git clone git@github.com:romanmikh/centimemap.git centimemap && cd centimemap
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                     # Python 3.12.3
# pytest
python3 -m src.main --method centime --head interp --data dummy1 --use-lungmask --load-ckpt # register at wandb & add credentials
```

The software is test on ... server consisting of:
xxx GiB RAM
xxx GPUs
...

The software is run on ... Linux and CUDAXXX.

To preprocess and mask the images, we use the following software:

- [lungmask](https://github.com/JoHof/lungmask)
  ...

Pretrained parameters: https://drive.google.com/drive/to_be_added
