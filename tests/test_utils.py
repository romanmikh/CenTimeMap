import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator
from torch import nn
from torch.utils.data import Dataset


def get_median_survival_cox(
    tr_preds: np.ndarray,
    tr_events: np.ndarray,
    tr_times: np.ndarray,
    ts_preds: np.ndarray,
    tmax: int = 156,
) -> np.ndarray:
    """
    Compute the median survival time from the predicted risk scores using Cox model.

    Args:
        tr_preds (np.ndarray): Training set predicted risk scores. Shape: (n_samples,).
        tr_events (np.ndarray): Training set binary event labels. Shape: (n_samples,).
        tr_times (np.ndarray): Training set time-to-event labels. Shape: (n_samples,).
                                Event time if event is True, censoring time otherwise.
        ts_preds (np.ndarray): Test set predicted risk scores. Shape: (n_samples,).
        tmax (int, optional): Maximum time to consider. Defaults to 156 (tmax in OSIC
          dataset). Used as the upper bound of the time range, as the default behavior
            of the Breslow estimator is to output infinity if the survival value does
              not drop below 0.5.

    Returns:
        np.ndarray: Predicted median survival time for each sample in the test set.
    """

    breslow = BreslowEstimator().fit(tr_preds, tr_events, tr_times)
    min_time, max_time = tr_times.min(), tr_times.max()
    times = np.arange(min_time, max_time)

    sample_surv_fn = breslow.get_survival_function(ts_preds)
    intermediate_preds = np.asarray([[fn(t) for t in times] for fn in sample_surv_fn])

    preds = []
    for pred in intermediate_preds:
        median_time_idx = np.where(pred <= 0.5)[0]
        preds.append(median_time_idx[0] if len(median_time_idx) > 0 else tmax)

    return np.array(preds)


class CustomDataset(Dataset):
    def __init__(self, x, target, events):
        self.x = x
        self.times = target
        self.events = events

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "img": self.x[idx],
            "time": self.times[idx],
            "event": self.events[idx],
        }


class TestModel(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(TestModel, self).__init__()
        self.input_dim = input_dim  # to capture them while saving args
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, y=None):
        return self.model(x)
