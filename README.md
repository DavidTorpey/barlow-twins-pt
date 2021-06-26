# Barlow Twins

This repository contains a PyTorch implementation of the [Barlow Twins](https://arxiv.org/abs/2103.03230)
self-supervised learning architecture.

There are a couple of differences from the paper. Firstly, in this implementation
the pre-training is performed on CIFAR10 instead of ImageNet, as this implementation
is meant to serve simply as a reference. Secondly, the Adam optimiser is used instead
of LARS. A couple of other minor changes (e.g. in types of augmentations) are also
present. However, these are simply enough amendments/changes to make in your own
extensions or implementations.

I also hard-core the projector network size/architecture to a 2-layer MLP. You
can extend the code to make the size dynamic based on config/input.

## Configuration

The config is stored in `src/config.py`. You can abstract this in your own way
should you desire (e.g. YAML, JSON, etc.).

## Running the Code

Simply run:

```bash
python main.py
```