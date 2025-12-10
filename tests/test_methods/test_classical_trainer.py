# pylint: disable=no-member

import pdb
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from src.methods import ClassicalTrainer
from src.utils.train_utils import seed_everything
from tests.test_utils import CustomDataset, TestModel
from src.utils.train_utils import get_device


def test_classical_trainer():
    seed_everything(42)
    tmax = 100
    model = TestModel(10, tmax)
    model.forward = lambda x, y=None: F.softmax(model.model(x), dim=1) # ensures +ve values
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None
    device=get_device()

    x = torch.randn(10, 10)
    x = (x - x.mean()) / x.std()
    target = torch.randint(0, tmax, (10,))
    events = torch.randint(0, 2, (10,))

    train_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=True
    )
    val_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=False
    )
    # pdb.set_trace()
    trainer = ClassicalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        tmax=tmax,
    )
    trainer.train()

if __name__ == "__main__":
    test_classical_trainer()
