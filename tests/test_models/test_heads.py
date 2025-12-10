import torch
from src.utils.train_utils import seed_everything
from numpy.testing import assert_allclose
from src.models.heads import StandardHead, InterpretabilityHead

STANDARD_OUTPUT = torch.tensor([[0.206693, 0.204693, 0.201309, 0.196611, 0.190694],
                                [0.207011, 0.204847, 0.201303, 0.196451, 0.190389]])
INTERPRET_OUTPUT = torch.tensor([[0.207069, 0.204871, 0.201298, 0.196421, 0.190341],
                                 [0.207299, 0.204983, 0.201294, 0.196306, 0.190119]])

def test_standard_head():
    seed_everything(42)
    batch_size, num_patches, dim, tmax, var = 2, 8, 512, 5, 144.0
    x = torch.randn(batch_size, num_patches, dim)
    x = x.view(batch_size, 2, 2, 2, dim)
    
    head = StandardHead(DIM_IN=dim, tmax=tmax, var=var)
    output = head(x)
    
    assert output.shape == (batch_size, tmax)
    for i, pred in enumerate(output):
        assert_allclose(pred.detach().numpy(), STANDARD_OUTPUT[i].numpy(), rtol=1e-4)

def test_interpretability_head():
    seed_everything(42)
    batch_size, num_patches, dim, tmax, var = 2, 8, 512, 5, 144.0
    x = torch.randn(batch_size, num_patches, dim)
    x = x.view(batch_size, 2, 2, 2, dim)
    
    head = InterpretabilityHead(dim=dim, tmax=tmax, var=var)
    output = head(x)
    
    assert output.shape == (batch_size, tmax)
    for i, pred in enumerate(output):
        assert_allclose(pred.detach().numpy(), INTERPRET_OUTPUT[i].numpy(), rtol=1e-4)

if __name__ == "__main__":
    test_standard_head()
    test_interpretability_head()
