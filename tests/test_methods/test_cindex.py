# All assertions passed

# pylint: disable=no-member
import torch
from sksurv.metrics import concordance_index_censored

from src.metrics import CIndex


def test_cindex():
    """
    Test the CIndex metric.
    """
    cindex = CIndex()
    preds = torch.Tensor([0.0, 1.0, 2.0])
    target = torch.Tensor([1.0, 2.0, 2.0])
    event = torch.Tensor([0, 0, 1]).bool()
    cindex.update(preds, target, event)
    tgt_answer = concordance_index_censored(
        event.cpu().numpy(), target.cpu().numpy(), preds.cpu().numpy()
    )[0]
    assert cindex.compute() == tgt_answer

    # test with multiple updates
    cindex = CIndex()
    preds1 = torch.Tensor([0.0, 1.0, 2.0])
    target1 = torch.Tensor([1.0, 2.0, 2.0])
    event1 = torch.Tensor([0, 0, 1]).bool()
    cindex.update(preds1, target1, event1)
    preds2 = torch.Tensor([1.0, 2.0, 3.0])
    target2 = torch.Tensor([2.0, 3.0, 3.0])
    event2 = torch.Tensor([0, 1, 1]).bool()
    cindex.update(preds2, target2, event2)
    tgt_answer = concordance_index_censored(
        torch.cat([event1, event2]).cpu().numpy(),
        torch.cat([target1, target2]).cpu().numpy(),
        torch.cat([preds1, preds2]).cpu().numpy(),
    )[0]
    assert cindex.compute() == tgt_answer

    cindex = CIndex()
    preds1 = torch.Tensor([0.0, 1.0, 2.0])
    target1 = torch.Tensor([1.0, 2.0, 2.0])
    event1 = torch.Tensor([0, 0, 1]).bool()
    cindex.update(preds1, target1, event1)
    preds2 = torch.Tensor([1.0, 2.0, 3.0])
    target2 = torch.Tensor([2.0, 3.0, 3.0])
    event2 = torch.Tensor([0, 1, 1]).bool()
    cindex.update(preds2, target2, event2)
    preds3 = torch.Tensor([2.0, 3.0, 4.0])
    target3 = torch.Tensor([3.0, 4.0, 4.0])
    event3 = torch.Tensor([1, 1, 1]).bool()
    cindex.update(preds3, target3, event3)
    tgt_answer = concordance_index_censored(
        torch.cat([event1, event2, event3]).cpu().numpy(),
        torch.cat([target1, target2, target3]).cpu().numpy(),
        torch.cat([preds1, preds2, preds3]).cpu().numpy(),
    )[0]
    assert cindex.compute() == tgt_answer
    print("All assertions passed")

if __name__ == "__main__":
    test_cindex()