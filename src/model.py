import torch
from torch import nn
from torchvision import models

from config import LAMBDA
from utils import off_diagonal


class BarlowTwins(nn.Module):
    """Implementation of the Barlow Twins self-supervised learning architecture

    References:
        https://arxiv.org/abs/2103.03230
    """
    def __init__(self):
        super(BarlowTwins, self).__init__()

        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128)
        )

        self.bn = nn.BatchNorm1d(128, affine=False)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """A forward pass through the Barlow Twins architecture

        A forward pass consists of computing the latent vectors and projection
        vectors for both batches of random views. Then the batches of projection
        vectors are independently standardised (zero-mean and unit variance). This
        implementation uses a BatchNorm layer for this, however, one could instead
        simply do x <- (x - mu) / sigma. The cross-correlation matrix is then
        computed from this normalised vectors, and the loss is computed using the
        on-diagonal and off-diagonal elements.

        The on-diagonal loss attempts to maximise the correlation, and thus we
        minimise 1-correlation. Similarly, we minimise off-diagonal correlation
        values since we want these vectors to be dissimilar.
        """
        h1 = self.encoder(view1).squeeze(-1).squeeze(-1)
        z1 = self.projector(h1)

        h2 = self.encoder(view2).squeeze(-1).squeeze(-1)
        z2 = self.projector(h2)

        num_vectors = len(z1)

        cross_correlation_matrix = torch.matmul(self.bn(z1).T, self.bn(z2)) / num_vectors

        on_diag_loss = torch.diagonal(cross_correlation_matrix).add_(-1).pow_(2).sum()

        off_diag_loss = off_diagonal(cross_correlation_matrix).pow_(2).sum()

        full_loss = on_diag_loss + LAMBDA * off_diag_loss

        return full_loss
