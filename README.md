# ViT-IC: Vision Transformer for Image Clustering

![diagram](diagram.png)

Implementation of a self-supervised contrastive learning framework with a Vision Transformer backbone for clustering images. You can read the paper [here](https://ryanlow.me/RyanLow_ViTIC.pdf).

There are only two different types of commands to run, they should look like the follow examples:

```
python train.py config/cifar10.yaml
python eval.py config/cifar10.yaml
```

Change experiment parameters in a `.yaml` file. Examples can be found in the `/config` directory.