# pylint: disable=no-member
import torch
from torch.testing import assert_close

from src.methods.mb import CoxMemoryBank


def test_coxmb_trainer():
    """
    Test the CoxMB trainer.
    """
    preds1 = torch.randn(10, 1)
    preds2 = torch.randn(10, 1)

    times1 = torch.randint(0, 100, (10, 1))
    times2 = torch.randint(0, 100, (10, 1))

    events1 = torch.randint(0, 2, (10, 1))
    events2 = torch.randint(0, 2, (10, 1))

    mb = CoxMemoryBank(k=1.0, total_samples=20)
    mb.update(preds1, events1, times1)
    mb.update(preds2, events2, times2)

    preds, times, events = mb.get_memory_bank()
    assert_close(preds, torch.cat((preds1, preds2)))
    assert_close(times, torch.cat((times1, times2)))
    assert_close(events, torch.cat((events1, events2)))

    mb = CoxMemoryBank(k=0.5, total_samples=20)

    mb.update(preds1, events1, times1)
    mb.update(preds2, events2, times2)

    preds, times, events = mb.get_memory_bank()
    assert_close(preds, torch.cat((preds1, preds2))[-10:])
    assert_close(times, torch.cat((times1, times2))[-10:])
    assert_close(events, torch.cat((events1, events2))[-10:])

    # Test reset
    mb.reset()
    assert mb.current_size == 0
    assert mb.preds.size(0) == 10
    assert mb.times.size(0) == 10
    assert mb.events.size(0) == 10

    # Test update overflow
    mb = CoxMemoryBank(k=0.75, total_samples=20)
    mb.update(preds1, events1, times1)
    mb.update(preds2, events2, times2)
    mb.update(preds1, events1, times1)
    preds, times, events = mb.get_memory_bank()
    assert_close(preds, torch.cat((preds1, preds2, preds1))[-15:])
    assert_close(times, torch.cat((times1, times2, times1))[-15:])
    assert_close(events, torch.cat((events1, events2, events1))[-15:])

if __name__ == "__main__":
    test_coxmb_trainer()
    print("MB tests passed.")