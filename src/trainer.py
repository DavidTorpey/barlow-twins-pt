import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from model import BarlowTwins
from config import EPOCHS, DEVICE


class Trainer:
    """Simple trainer class to abstract away training details of the self-supervised
    pre-training from user"""

    def __init__(self, model: BarlowTwins, optimiser: optim.Optimizer):
        self.model = model.to(DEVICE)
        self.optimiser = optimiser

    def train(self, train_loader: DataLoader):
        n_train_batches = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))

        for epoch in range(EPOCHS):
            print(f'Starting epoch {epoch + 1} of {EPOCHS}')
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch + 1}/{EPOCHS}: Processing batch {batch_idx + 1} of {n_train_batches}')

                batch = (e.to(DEVICE) for e in batch)
                view1, view2 = batch

                self.optimiser.zero_grad()

                loss = self.model(view1, view2)

                loss.backward()
                self.optimiser.step()

        torch.save(self.model, 'model.pth')
