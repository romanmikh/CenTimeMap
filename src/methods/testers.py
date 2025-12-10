# # pylint: disable=arguments-differ, super-init-not-called

# import os
# from abc import abstractmethod
# from typing import Optional, Tuple, Union, no_type_check

# import torch
# import yaml
# from torch import nn
# from torch.utils.data import DataLoader, SequentialSampler

# from src.methods import DeepHitTrainer
# from src.methods.centime_trainer import CenTimeTrainer
# from src.metrics import MAE, RAE, CIndex, CoxAccumulator
# from tests import models_dict


# class BaseTester:
#     """Base class for all testers."""

#     def __init__(
#         self,
#         ckpt_path: str,
#         loader: DataLoader,
#         device: Union[str, torch.device],
#     ):
#         assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

#         self.device = device
#         self.loader = loader
#         self.ckpt_path = ckpt_path
#         exp_path = os.path.dirname(ckpt_path)
#         with open(os.path.join(exp_path, "args.yaml"), "r", encoding="utf-8") as f:
#             self.args = yaml.safe_load(f)

#         self.model, self.accumulator = self.get_model()

#         self.cindex = CIndex().to(self.device)
#         self.mae_uncensored = MAE(mode="uncensored").to(self.device)
#         self.mae_censored = MAE(mode="censored").to(self.device)
#         self.rae_uncensored = RAE(mode="uncensored").to(self.device)
#         self.rae_censored = RAE(mode="censored").to(self.device)

#     def get_model(self) -> Tuple[nn.Module, Optional[CoxAccumulator]]:
#         """Construct the model and load it from the checkpoint."""
#         model = models_dict[self.args["model"]](**self.args)
#         ckpt = torch.load(self.ckpt_path, map_location=self.device)
#         model.load_state_dict(ckpt["model"])
#         accumulator = ckpt.get("accumulator", None)
#         model.to(self.device)
#         return model, accumulator

#     @abstractmethod
#     def test(self) -> Tuple[float, float, float, float, float]:
#         """
#         Test the model.

#         Returns:
#             C-index, MAE uncensored, MAE censored, RAE uncensored, RAE censored.
#         """


# class CoxTester(BaseTester):
#     """Tester for Cox loss."""

#     @no_type_check
#     def get_metrics(self) -> Tuple[float, float, float, float, float]:
#         """
#         Get the metrics for the model.

#         Returns:
#             C-index, MAE uncensored, MAE censored, RAE uncensored, RAE censored.
#         """
#         assert isinstance(
#             self.loader.sampler, SequentialSampler
#         ), "Shuffle must be False in the data loader for accurate results"

#         preds = self.accumulator.compute(training=False).to(self.device)
#         tr_times = self.loader.dataset.targets.to(self.device)  # type: ignore
#         tr_events = self.loader.dataset.events.to(self.device)  # type: ignore

#         self.rae_censored.update(preds, tr_times, tr_events)
#         self.rae_uncensored.update(preds, tr_times, tr_events)
#         self.mae_censored.update(preds, tr_times, tr_events)
#         self.mae_uncensored.update(preds, tr_times, tr_events)

#         cindex = self.cindex.compute()
#         self.cindex.reset()
#         mae_nc = self.mae_uncensored.compute()
#         self.mae_uncensored.reset()
#         mae_c = self.mae_censored.compute()
#         self.mae_censored.reset()
#         rae_nc = self.rae_uncensored.compute()
#         self.rae_uncensored.reset()
#         rae_c = self.rae_censored.compute()
#         self.rae_censored.reset()

#         self.accumulator.reset()  # type: ignore

#         return cindex.item(), mae_nc.item(), mae_c.item(), rae_nc.item(), rae_c.item()

#     def test(self) -> Tuple[float, float, float, float, float]:
#         """
#         Test the model.

#         Returns:
#             C-index, MAE uncensored, MAE censored, RAE uncensored, RAE censored.
#         """
#         self.model.eval()
#         for batch in self.loader:
#             batch = {k: v.to(self.device) for k, v in batch.items()}
#             img = batch["img"]
#             clinical_data = (
#                 batch["clinical_data"] if self.args["clinical_data"] else None
#             )
#             output = self.model(img, clinical_data)
#             self.cindex.update(output, batch["time"], batch["event"])
#             self.accumulator.update(ts_preds=output)  # type: ignore

#         cindex, mae_nc, mae_c, rae_nc, rae_c = self.get_metrics()

#         return cindex, mae_nc, mae_c, rae_nc, rae_c


# class DeepHitTester(BaseTester, DeepHitTrainer):
#     """Tester for DeepHit."""

#     def test(self) -> Tuple[float, float, float, float, float]:
#         """
#         Test the model.

#         Returns:
#             C-index, MAE uncensored, MAE censored, RAE uncensored, RAE censored.
#         """
#         self.model.eval()
#         for batch in self.loader:
#             batch = {k: v.to(self.device) for k, v in batch.items()}
#             img = batch["img"]
#             clinical_data = (
#                 batch["clinical_data"] if self.args["clinical_data"] else None
#             )
#             output = self.model(img, clinical_data).softmax(dim=1)
#             pred_time = self.get_predicted_time(output)
#             self.mae_uncensored.update(pred_time, batch["time"], batch["event"])
#             self.mae_censored.update(pred_time, batch["time"], batch["event"])
#             self.cindex.update(-pred_time, batch["time"], batch["event"])
#             self.rae_uncensored.update(pred_time, batch["time"], batch["event"])
#             self.rae_censored.update(pred_time, batch["time"], batch["event"])

#         mae_nc = self.mae_uncensored.compute().item()
#         self.mae_uncensored.reset()
#         mae_c = self.mae_censored.compute().item()
#         self.mae_censored.reset()
#         cindex = self.cindex.compute().item()
#         self.cindex.reset()
#         rae_nc = self.rae_uncensored.compute().item()
#         self.rae_uncensored.reset()
#         rae_c = self.rae_censored.compute().item()
#         self.rae_censored.reset()

#         return cindex, mae_nc, mae_c, rae_nc, rae_c


# class DistTester(BaseTester, CenTimeTrainer):
#     """Tester for CenTime and Classical losses."""

#     def __init__(self, *args, **kwargs):
#         BaseTester.__init__(self, *args, **kwargs)
#         self.variance = self.args["variance"]
#         self.tmax = self.args["tmax"]
#         self.distribution = self.args["distribution"]

#     def test(self) -> Tuple[float, float, float, float, float]:
#         """
#         Test the model.

#         Returns:
#             C-index, MAE uncensored, MAE censored, RAE uncensored, RAE censored.
#         """
#         self.model.eval()
#         for batch in self.loader:
#             batch = {k: v.to(self.device) for k, v in batch.items()}
#             img = batch["img"]
#             clinical_data = (
#                 batch["clinical_data"] if self.args["clinical_data"] else None
#             )
#             output = self.model(img, clinical_data)
#             pred_time = self.get_predicted_time(output)
#             self.mae_uncensored.update(pred_time, batch["time"], batch["event"])
#             self.mae_censored.update(pred_time, batch["time"], batch["event"])
#             self.cindex.update(-pred_time, batch["time"], batch["event"])
#             self.rae_uncensored.update(pred_time, batch["time"], batch["event"])
#             self.rae_censored.update(pred_time, batch["time"], batch["event"])

#         mae_nc = self.mae_uncensored.compute().item()
#         self.mae_uncensored.reset()
#         mae_c = self.mae_censored.compute().item()
#         self.mae_censored.reset()
#         cindex = self.cindex.compute().item()
#         self.cindex.reset()
#         rae_nc = self.rae_uncensored.compute().item()
#         self.rae_uncensored.reset()
#         rae_c = self.rae_censored.compute().item()
#         self.rae_censored.reset()

#         return cindex, mae_nc, mae_c, rae_nc, rae_c
