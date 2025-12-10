# pylint: disable=no-member

import torch
from numpy.testing import assert_array_equal

from src.metrics import CoxAccumulator
from tests.test_utils import get_median_survival_cox


def test_cox_accumulator():
    accumulator = CoxAccumulator(tmax=100)
    tr_preds1 = torch.rand(10)
    tr_events1 = torch.randint(0, 2, (10,))
    tr_times1 = torch.randint(0, 100, (10,))
    ts_preds1 = torch.rand(10)

    tr_preds2 = torch.rand(10)
    tr_events2 = torch.randint(0, 2, (10,))
    tr_times2 = torch.randint(0, 100, (10,))
    ts_preds2 = torch.rand(10)

    # during training
    accumulator.update(tr_preds=tr_preds1, tr_events=tr_events1, tr_times=tr_times1)
    accumulator.update(tr_preds=tr_preds2, tr_events=tr_events2, tr_times=tr_times2)
    accumulator.fit_breslow()
    preds = accumulator.compute(training=True)
    expected_preds = get_median_survival_cox(
        torch.cat([tr_preds1, tr_preds2]).cpu().numpy(),
        torch.cat([tr_events1, tr_events2]).cpu().numpy(),
        torch.cat([tr_times1, tr_times2]).cpu().numpy(),
        torch.cat([tr_preds1, tr_preds2]).cpu().numpy(),
        tmax=100,
    )
    assert_array_equal(preds, torch.Tensor(expected_preds))

    # during testing
    accumulator.update(ts_preds=ts_preds1)
    accumulator.update(ts_preds=ts_preds2)
    preds = accumulator.compute(training=False)
    expected_preds = get_median_survival_cox(
        torch.cat([tr_preds1, tr_preds2]).cpu().numpy(),
        torch.cat([tr_events1, tr_events2]).cpu().numpy(),
        torch.cat([tr_times1, tr_times2]).cpu().numpy(),
        torch.cat([ts_preds1, ts_preds2]).cpu().numpy(),
        tmax=100,
    )
    assert_array_equal(preds, torch.Tensor(expected_preds))

if __name__ == "__main__":
    test_cox_accumulator()
    print("CoxAccumulator tests passed.")