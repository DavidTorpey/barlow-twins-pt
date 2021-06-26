from torch import optim

from data import get_cifar10_loader
from model import BarlowTwins
from config import LR, WEIGHT_DECAY
from trainer import Trainer


def main():
    train_loader = get_cifar10_loader()

    barlow_twins = BarlowTwins()

    optimiser = optim.Adam(barlow_twins.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    trainer = Trainer(barlow_twins, optimiser)

    trainer.train(train_loader)


if __name__ == '__main__':
    main()
