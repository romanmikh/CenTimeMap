"""
PyTorch Dataset class for the dataset. A placeholder. Users should implement their own
 dataset class.

Author: Ahmed H. Shahin
Date: 31/08/2023
"""

import json
import multiprocessing as mp
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
from lungmask import mask as lm_mask, LMInferer

import src.dataset.custom_transforms as tr


def load_scan(path):
    """
    Load a scan from an h5 file.

    Args:
        path: Path to the h5 file.

    Returns:
        scan: Scan.
    """
    file = h5py.File(path, "r")
    # scan = np.array(file.get("img"))
    
    # if downsampled is a key in the h5 file, use it instead
    if "downsampled" in file.keys():
        scan = np.array(file.get("downsampled"))
    else:
        scan = np.array(file.get("img"))
        
    return scan


class OSICDataset(Dataset):
    """
    Load the dataset and apply the transformations.

    Args:
        data_path (str): path to the data directory.
        clinical_data_path (str): path to the clinical data.
        split (str): split of the dataset (train, val, test).
        segment (bool): whether to (lung) segment the images or not.
        fold (int): fold of the dataset (1, 2, 3, 4, 5).
        num_patients (int): number of samples to load. -1 means all samples.
        p_aug (float): probability of applying the augmentation.
        p_uncens (float): proportion of uncensored samples to load.
        clinical_normalizer (object): clinical data normalizer for continuous variables.
          None means we are loading the training set and we need to fit the normalizer.
        clinical_encoder (object): clinical data encoder for categorical variables.
          None means we are loading the training set and we need to fit the encoder.
        load_imgs (bool): whether to load the images or not. This is useful when we
          only want to load the clinical data.
    """

    def __init__(
        self,
        data_path,
        clinical_data_path,
        split="train",
        segment=False,
        fold=1,
        num_patients=-1,
        p_uncens=1.0,
        use_lung_mask=False,
        use_clinical_data=False,
        clinical_normalizer=None,
        clinical_encoder=None,
        load_imgs=True,
        tmax=None,
        n_dummy_states=0,
    ):
        super().__init__()
        assert split in ["train", "val", "test", "test_external"]
        if use_clinical_data:
            if split == "train":
                assert clinical_normalizer is None and clinical_encoder is None
            else:
                assert clinical_normalizer is not None and clinical_encoder is not None
        if split == "test_external":
            assert num_patients == -1
            assert (
                fold is None
            ), "fold must be None during external testing. It is only used for cross-validation."
            fold = 1
        else:
            assert fold in range(1, 6), "fold must be in [1, 2, 3, 4, 5]."

        self.clinical_normalizer = clinical_normalizer
        self.clinical_encoder = clinical_encoder
        self.load_imgs = load_imgs
        self.split = split
        self.use_clinical_data = use_clinical_data
        self.n_dummy_states = n_dummy_states
        self.use_lung_mask = use_lung_mask

        fold = str(fold) if fold is not None else None
        segment = str(bool(segment))

        df_clinical = self.load_clinical_data(
            clinical_data_path, split, fold, num_patients, p_uncens, tmax
        )

        self.extract_clinical_data(df_clinical)

        if self.load_imgs:
            # we converted the images to h5 files to save space
            self.img_paths = [os.path.join(data_path, f"{pid_sid}_preprocessed.h5") for pid_sid in self.pid_sids]

        self.transforms = self.get_transforms(
            split,
            use_clinical_data,
        )
        
        if self.use_lung_mask:
            self.inferer = LMInferer(batch_size=10, tqdm_disable=True, force_cpu=False)

            

    def get_transforms(
        self,
        split,
        use_clinical_data=False,
    ):
        """
        Get the transformations to apply to the data.
        """
        trs = transforms.Compose(
            [
                tr.ToTensor(),
                tr.Windowing(),
            ]
        )

        if use_clinical_data:
            trs.transforms.insert(len(trs.transforms), tr.ImputeMissingData(split=split, params_path="imputation_model.pkl.npz"))
            trs.transforms.insert(len(trs.transforms), tr.NormalizeClinicalData(normalizer=self.clinical_normalizer, encoder=self.clinical_encoder))
        
        if not self.load_imgs:
            del trs.transforms[1]

        # if split == "train":
        #     trs.transforms.insert(len(trs.transforms), tr.RandomTranslateTorch(prob=p_aug))

        return trs

    def extract_clinical_data(self, df_clinical):
        """
        Extract the clinical data from the dataframe.

        Args:
            df_clinical (pandas.DataFrame): clinical data.
        """
        self.pid_sids = df_clinical["pid_sid"].values
        self.time_to_event = df_clinical["LAST PATIENT INFORMATION (transplant is censoring)"].values
        self.event = df_clinical["STATES (DEAD=1/ALIVE=0) (transplant is censoring)"].values
        self.cat = df_clinical["EVENT_TIME_CAT"].values
        cont_feats = ["AGE", "FVC PREDICTED", "DLCO"]
        disc_feats = [
            "SEX(male=1,female=0)",
            "SMOKING HISTORY(active-smoker=2,ex-smoker=1,never-smoker=0)",
            "ANTIFIBROTIC",
        ]
        clinical_data = df_clinical[cont_feats + disc_feats]
        if self.split == "train":
            (
                self.clinical_normalizer,
                self.clinical_encoder,
            ) = self.normalize_clinical_data(clinical_data, disc_feats, cont_feats)
        self.clinical_data = clinical_data.values

    @staticmethod
    def load_clinical_data(data_path, split, fold, num_patients, p_uncens, tmax):
        """
        Load the clinical data.

        Args:
            data_path (str): path to the clinical data.
            split (str): split of the dataset (train, val, test).
            fold (int): fold of the dataset (1, 2, 3, 4, 5).
            num_patients (int): number of patients to load. -1 means all patients.
            p_uncens (float): proportion of uncensored samples to load.

        Returns:
            df_clinical (pandas.DataFrame): clinical data.
        """
        df_clinical = pd.read_csv(data_path)
        df_clinical = df_clinical.loc[df_clinical[f"survival_split_{fold}"] == split]
        # convert months to weeks
        # add 1 to avoid 0 time_to_event. t can't be 0,
        df_clinical["LAST PATIENT INFORMATION (transplant is censoring)"] = (df_clinical["LAST PATIENT INFORMATION (transplant is censoring)"] / 4.3).round().astype(int) + 1
        df_cens = df_clinical.loc[df_clinical["STATES (DEAD=1/ALIVE=0) (transplant is censoring)"] == 0].reset_index(drop=True)
        df_cens.loc[df_cens["LAST PATIENT INFORMATION (transplant is censoring)"] > tmax, "LAST PATIENT INFORMATION (transplant is censoring)"] = tmax - 1
        df_uncens = df_clinical.loc[df_clinical["STATES (DEAD=1/ALIVE=0) (transplant is censoring)"] == 1].reset_index(drop=True)
        assert df_uncens.loc[df_uncens["LAST PATIENT INFORMATION (transplant is censoring)"] > tmax].shape[0] == 0, "Uncensored patients have survival time greater than tmax"
        df_uncens = df_uncens[: int(p_uncens * len(df_uncens))]
        if num_patients != -1:
            df_cens = df_cens[: num_patients // 2]
            df_uncens = df_uncens[: num_patients // 2]
        # for oversampling from less frequent event times
        # df_uncens["EVENT_TIME_CAT"] = pd.cut(df_uncens["LAST PATIENT INFORMATION (transplant is censoring)"], bins=[0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], labels=False)
        df_uncens["EVENT_TIME_CAT"] = pd.cut(df_uncens["LAST PATIENT INFORMATION (transplant is censoring)"], bins=[0, 48, 96, 120], labels=False)
        df_cens["EVENT_TIME_CAT"] = -1
        df_clinical = pd.concat([df_cens, df_uncens]).reset_index(drop=True)
        df_clinical['EVENT_TIME_CAT'] = df_clinical.apply(lambda row: 'censored' if row["STATES (DEAD=1/ALIVE=0) (transplant is censoring)"] == 0 else f'uncensored_{row["EVENT_TIME_CAT"]}', axis=1)

        return df_clinical

    @staticmethod
    def normalize_clinical_data(clinical_data, disc_feats, cont_feats):
        """
        Normalizes the clinical data:
            - Continuous featurs: by subtracting the mean and dividing by the standard deviation.
            - Categorical features: by one-hot encoding.

        Args:
            clinical_data (pandas.DataFrame): clinical data.
            disc_feats (list): list of discrete features.
            cont_feats (list): list of continuous features.

        Returns:
            normalizer (object): normalizer for continuous features.
            encoder (object): encoder for discrete features.
        """
        # normalize continuous features
        cont_data = clinical_data[cont_feats].values
        normalizer = StandardScaler()
        cont_data = normalizer.fit(cont_data)

        # encode discrete features
        disc_data = clinical_data[disc_feats].values
        disc_data = disc_data[
            np.isnan(disc_data).sum(axis=1) == 0
        ]  # remove rows with missing values before encoding
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit_transform(disc_data)

        return normalizer, encoder

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): index of the item to get.

        Returns:
            sample (dict): dictionary containing the sample.
        """
        pid_sid = self.pid_sids[idx]
        time_to_event = np.array([self.time_to_event[idx]])
        event = np.array([self.event[idx]])
        cat = self.cat[idx]

        sample = {
            "pid_sid": pid_sid,
            # "time_to_event": time_to_event,
            "time": time_to_event,
            "event": event,
            # "cat": cat,
        }

        if self.use_clinical_data:
            clinical_data = self.clinical_data[idx]
            sample["clinical_data"] = clinical_data

        if self.load_imgs:
            img = load_scan(self.img_paths[idx]).astype(np.float32)
            sample["img"] = img
            
        if self.use_lung_mask:
            # HU_RANGE = 1624.0
            # HU_LB = -1024.0
            # ct_hu = sample["img"] * HU_RANGE + HU_LB
            seg_labels = self.inferer.apply(sample["img"]).astype(np.uint8)
            lung_mask = seg_labels > 0
            # print(f"lung mask shape: {lung_mask.shape}")
            lung_mask = np.transpose(lung_mask, (0, 2, 1)) # switch the second and third dim of mask
            lung_mask_t = lung_mask[::-1].copy() # flip the mask along the first dim
            sample["lung_mask"] = lung_mask_t

        sample = self.transforms(sample)

        return sample


class OSICDatasetRAM(OSICDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.parallel_load_imgs()

    def load_scan(self, path):
        file = h5py.File(path, "r")
        scan = np.array(file.get("img"))
        return scan, path # return the path as well to reorder the imgs

    def parallel_load_imgs(self):
        """
        Load the images in parallel. This requires sufficient RAM memory but makes
          training faster.
        """
        print("loading scans into memory...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            data = pool.map(self.load_scan, self.img_paths)
        unordered_imgs = [i[0] for i in data]
        unordered_paths = [i[1] for i in data]

        # reorder the imgs and paths to match the order of the pid_sids
        self.imgs = [None] * len(self.img_paths)
        for i, path in enumerate(self.img_paths):
            idx = unordered_paths.index(path)
            self.imgs[i] = unordered_imgs[idx]
        print("done")

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): index of the item to get.

        Returns:
            sample (dict): dictionary containing the sample.
        """
        pid_sid = self.pid_sids[idx]
        time_to_event = np.array([self.time_to_event[idx]])
        event = np.array([self.event[idx]])
        cat = self.cat[idx]

        sample = {
            "pid_sid": pid_sid,
            "time_to_event": time_to_event,
            "event": event,
            "cat": cat,
        }

        if self.use_clinical_data:
            clinical_data = self.clinical_data[idx]
            sample["clinical_data"] = clinical_data

        if self.load_imgs:
            # img = self.imgs[idx]
            img = load_scan(os.path.join("/dev/shm/osic", os.path.basename(self.img_paths[idx]))).astype(np.float32)
            sample["img"] = img

        sample = self.transforms(sample)

        return sample
