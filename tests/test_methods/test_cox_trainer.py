# Successfully overfits
# MAE censored: nan (cox model does not handle censored data)

# pylint: disable=no-member
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.methods import CoxTrainer
from src.utils.train_utils import seed_everything
from tests.test_utils import CustomDataset, TestModel
from src.utils.train_utils import get_device


def test_cox_trainer():
    seed_everything(123)
    model = TestModel(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None
    device=get_device()
    x = torch.randn(10, 10) # 10 scans with 10 features each, not 2D scans
    x = (x - x.mean()) / x.std()
    target = torch.randint(0, 100, (10,))
    events = torch.randint(0, 2, (10,))
    events[:] = 1
    train_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=True
    )
    val_loader = DataLoader(
        CustomDataset(x, target, events), batch_size=10, shuffle=False
    )
    trainer = CoxTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        tmax=100,
    )
    trainer.train()

if __name__ == "__main__":
    test_cox_trainer()