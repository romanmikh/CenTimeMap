import torch
from torch.utils.data import Dataset
from src.utils.settings import *


def dummy_dataset_manager(n_samples: int, complexity: int):
    """
    Returns a DummyCTDataset with the desired complexity level:

    1: single bright centered spheres, survival time scales with sphere size. Random noise otherwise
        Expected: model highlights sphere
        Demonstrates: model can learn simple features by survival times

    2: as 1, but with random number (1-3) of bright spheres at random locations and of random sizes
        Expected: model highlights multiple spheres
        Demonstrates: model can learn feature presence, but not by their fixed location

    3: many small bright bubbles only (near edge of image), simulating cysts in IPF
        Expected: model highlights bubbles
        Demonstrates: model can learn to focus on small relevant features

    4: as 1, but sphere is of average brightness vs rest of image (placed in corner so it's visible)
        Expected: model highlights sphere
        Demonstrates: model can learn features by shape & placement, not only by excessive brightness. Real fibrosis is not excessively bright in CT scans.

    5: as 2 & 3, but spheres present for short survival time samples only & random bubbles in ALL samples
        Expected: model highlights spheres, ignores bubbles
        Demonstrates: model can learn to focus on relevant features & ignore tempting confounders of similar appearance

    6: as 5, but with bright sphere always present in corner.
        Expected: model highlights spheres, ignores bubbles & corner sphere
        Demonstrates: model can learn to ignore tempting confounders of similar appearance & fixed location.
        Model should be able to highlight IPF-causing features, ignore similar-looking non-IPF-causing features & ignore omnipresent structures (trachea, heart) if irrelevant.
    """
    if complexity == 1:
        return DummyCTDataset(
            n_samples=n_samples,
            central_sphere=True,
            central_high_brightness=True,
        )
    elif complexity == 2:
        return DummyCTDataset(
            n_samples=n_samples,
            random_spheres=True,
            rand_high_brightness=True,
        )
    elif complexity == 3:
        return DummyCTDataset(
            n_samples=n_samples,
            honeycombing=True,
            honey_high_brightness=True,
        )
    elif complexity == 4:
        return DummyCTDataset(
            n_samples=n_samples,
            corner_sphere=True,
            corner_high_brightness=False,
        )
    elif complexity == 5:
        return DummyCTDataset(
            n_samples=n_samples,
            random_spheres=True,
            rand_omnipresent=False,
            rand_high_brightness=True,
            honeycombing=True,
            honey_omnipresent=True,
            honey_high_brightness=True,
        )
    elif complexity == 6:
        return DummyCTDataset(
            n_samples=n_samples,
            random_spheres=True,
            rand_omnipresent=False,
            rand_high_brightness=True,
            honeycombing=True,
            honey_omnipresent=True,
            honey_high_brightness=True,
            corner_sphere=True,
            corner_omnipresent=True,
            corner_high_brightness=True,
        )
    else:
        raise ValueError("complexity must be in 1-6")


class DummyCTDataset(Dataset):
    """Synthetic CTs with a synthetic features in sphere_frac of the volumes."""

    def __init__(
        self,
        n_samples: int = 10,
        bg_noise: float = 0.2,
        feat_frac: float = 0.5,
        avg_brightness: float = 0.0,
        high_brightness: float = 20.0,
        central_sphere: bool = False,
        central_radius: int = 25,
        central_omnipresent: bool = False,
        central_high_brightness: bool = True,
        corner_sphere: bool = False,
        corner_radius: int = 25,
        corner_omnipresent: bool = False,
        corner_high_brightness: bool = False,
        random_spheres: bool = False,
        rand_count_lo: int = 2,
        rand_count_hi: int = 3,
        rand_radius_lo: int = 10,
        rand_radius_hi: int = 20,
        rand_omnipresent: bool = False,
        rand_high_brightness: bool = True,
        honeycombing: bool = False,
        honey_omnipresent: bool = False,
        honey_count_lo: int = 20,
        honey_count_hi: int = 30,
        honey_radius_lo: int = 5,
        honey_radius_hi: int = 8,
        honey_high_brightness: bool = True,
    ):
        super().__init__()
        self.imgs = torch.zeros(n_samples, 1, CT_SIZE_W, CT_SIZE_H, CT_SIZE_D)
        self.masks = torch.zeros(n_samples, 1, CT_SIZE_W, CT_SIZE_H, CT_SIZE_D)
        self.times = torch.empty(n_samples)
        self.events = torch.empty(n_samples)

        zz, yy, xx = torch.meshgrid(
            torch.arange(CT_SIZE_W),
            torch.arange(CT_SIZE_H),
            torch.arange(CT_SIZE_D),
            indexing="ij",
        )

        def add_fixed_central_sphere(mask_dst, mask_out, radius=central_radius):
            centre = CT_SIZE_W // 2
            brightness = high_brightness if central_high_brightness else avg_brightness
            mask = (xx - centre) ** 2 + (yy - centre) ** 2 + (
                zz - centre
            ) ** 2 < radius**2
            mask_dst[mask] = brightness
            mask_out[mask] = 1

        def add_fixed_corner_sphere(mask_dst, mask_out, radius=corner_radius):
            centre = CT_SIZE_W
            brightness = high_brightness if corner_high_brightness else avg_brightness
            mask = (xx - centre) ** 2 + (yy - centre) ** 2 + (
                zz - centre
            ) ** 2 < radius**2
            mask_dst[mask] = brightness
            mask_out[mask] = 1  # TODO 1 vs 0 if omnipresent?

        def add_random_spheres(mask_dst, mask_out):
            num = torch.randint(rand_count_lo, rand_count_hi, ()).item()
            for i in range(num):
                mid = (rand_radius_lo + rand_radius_hi) // 2
                if i % 2 == 0:
                    rad = torch.randint(rand_radius_lo, mid, ()).item()
                else:
                    rad = torch.randint(mid, rand_radius_hi, ()).item()
                cx, cy, cz = [
                    torch.randint(rad, CT_SIZE_W - rad, ()).item() for _ in range(3)
                ]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 < rad**2
                mask_dst[mask] = (
                    high_brightness if rand_high_brightness else avg_brightness
                )
                mask_out[mask] = 1

        def add_honeycombing(mask_dst, mask_out):
            """Honeycombing pattern: concentric layers of small, touching hollow bubbles"""
            n_layers = torch.randint(2, 5, ()).item()

            # iterate outward -> inward (1-voxel strips)
            for layer in range(1, n_layers + 1):
                # mask for the current “shell” one voxel thick
                shell = (
                    (xx == layer)
                    | (xx == CT_SIZE_W - 1 - layer)
                    | (yy == layer)
                    | (yy == CT_SIZE_H - 1 - layer)
                    | (zz == layer)
                    | (zz == CT_SIZE_D - 1 - layer)
                )
                # candidate centres along this shell
                cand = torch.stack(torch.where(shell), dim=1)
                if cand.numel() == 0:
                    continue
                n_cysts = torch.randint(honey_count_lo, honey_count_hi, ()).item()
                centres = cand[torch.randperm(cand.shape[0])[:n_cysts]]
                for cx, cy, cz in centres:
                    rad = torch.randint(
                        honey_radius_lo, honey_radius_hi, ()
                    ).item()  # cyst radius 2-3 vox
                    # cavity (air in cyst) left empty - do nothing
                    r2 = (
                        (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
                    )  # 1-voxel wall around cavity
                    wall = (r2 <= (rad + 1) ** 2) & (r2 >= rad**2)
                    mask_dst[wall] = (
                        high_brightness if honey_high_brightness else avg_brightness
                    )
                    mask_out[wall] = 1

        for i in range(n_samples):
            radius = 0
            self.imgs[i] = torch.randn_like(self.imgs[i]) * bg_noise  # noise

            if central_omnipresent:
                add_fixed_central_sphere(self.imgs[i, 0], self.masks[i, 0])
            if corner_omnipresent:
                add_fixed_corner_sphere(self.imgs[i, 0], self.masks[i, 0])
            if rand_omnipresent:
                add_random_spheres(self.imgs[i, 0], self.masks[i, 0])
            if honey_omnipresent:
                add_honeycombing(self.imgs[i, 0], self.masks[i, 0])

            if i < int(n_samples * feat_frac):
                if central_sphere and not central_omnipresent:
                    radius = central_radius + torch.randint(-5, 5, ())
                    add_fixed_central_sphere(
                        self.imgs[i, 0], self.masks[i, 0], radius=radius
                    )
                if corner_sphere and not corner_omnipresent:
                    radius = corner_radius + torch.randint(-10, 9, ())
                    add_fixed_corner_sphere(
                        self.imgs[i, 0], self.masks[i, 0], radius=radius
                    )
                if random_spheres and not rand_omnipresent:
                    add_random_spheres(self.imgs[i, 0], self.masks[i, 0])
                if honeycombing and not honey_omnipresent:
                    add_honeycombing(self.imgs[i, 0], self.masks[i, 0])

                surv_time = 35 - radius
                self.times[i] = torch.randint(
                    surv_time - 1, surv_time, ()
                ).float()  # short survival, censored
                self.events[i] = 1.0
            else:  # noise only CT scan
                self.times[i] = torch.randint(
                    99, 100, ()
                ).float()  # long survival, uncensored
                self.events[i] = 0.0
            # print(f"Sample radius: {radius}, time: {self.times[i].item()}, event: {self.events[i].item()}")

    def __getitem__(self, idx):
        return {
            "img": self.imgs[idx],
            "time": self.times[idx],
            "event": self.events[idx],
            "mask": self.masks[idx][0],  # shape: (Z, Y, X)
        }

    def __len__(self):
        return len(self.imgs)
