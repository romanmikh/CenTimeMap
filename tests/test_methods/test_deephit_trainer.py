# Successfully overfits

# pylint: disable=no-member
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.methods import DeepHitTrainer
from src.utils.train_utils import seed_everything
from tests.test_utils import CustomDataset, TestModel
from src.utils.train_utils import get_device


def test_deephit_trainer():
    seed_everything(42)
    model = TestModel(10, 100)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None
    device=get_device()
    x = torch.randn(10, 10)
    x = (x - x.mean()) / x.std()
    target = torch.randint(0, 100, (10,))
    events = torch.randint(0, 2, (10,))
    # events[:] = 1
    train_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=True
    )
    val_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=False
    )
    trainer = DeepHitTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        tmax=100,
        ranking=False,
    )
    trainer.train()

if __name__ == "__main__":
    test_deephit_trainer()
