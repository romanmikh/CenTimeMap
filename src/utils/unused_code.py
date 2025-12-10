""" alternative MLPs tested in InterpretabilityHead - worse performance"""

# self.mlp = nn.Sequential(  # worst results
#     nn.Linear(DIM_IN, self.hidden),
#     nn.LeakyReLU(0.1),
#     nn.LayerNorm(self.hidden),
#     nn.Linear(self.hidden, self.hidden),
#     nn.LeakyReLU(0.1),
#     nn.LayerNorm(self.hidden),
#     nn.Linear(self.hidden, 1),
# )
# self.mlp = nn.Sequential(  # sub-par but reasonable results 
#     nn.LayerNorm(DIM_IN),
#     nn.Linear(DIM_IN, self.hidden, bias=False),
#     nn.GELU(),
#     nn.Linear(self.hidden, 1)
# )
# attempted to seaprate loc (viz) from risk (weighting) logic - worse results
# self.mlp_loc = nn.Sequential(
#     nn.LayerNorm(dim),
#     nn.Linear(dim, self.hidden),
#     nn.GELU(),
#     nn.Linear(self.hidden, 1)
# )
# self.mlp_risk = nn.Sequential(
#     nn.LayerNorm(dim),
#     nn.Linear(dim, self.hidden),
#     nn.GELU(),
#     nn.Linear(self.hidden, 1)
# )


""" attention pooling module tested instead of MLP in head but worse performance"""
# class AttentionPooling(nn.Module):
#     def __init__(self, dim_in):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(1, dim_in))  # learnable query
#         self.key_proj = nn.Linear(dim_in, dim_in)
#         self.value_proj = nn.Linear(dim_in, 1)

#     def forward(self, x):  # x: (B, N, D)
#         q = self.query.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, 1, D)
#         k = self.key_proj(x)  # (B, N, D)
#         v = self.value_proj(x).squeeze(-1)  # (B, N)
#         attn = torch.softmax((q @ k.transpose(1, 2)) / (k.size(-1) ** 0.5), dim=-1)  # (B, 1, N)
#         weighted_v = (attn * v.unsqueeze(1)).sum(dim=-1)  # (B,)
#         return weighted_v


""" dataset wrapper to add clinical data - not used currently"""
# class WithClinical(Dataset):
#     """Adds clinical data to a base dataset"""
#     def __init__(self, base_ds: Dataset, clinical_csv: str):
#         self.base_ds = base_ds
#         self.n = len(self.base_ds)

#         df = pd.read_csv(clinical_csv)
#         # TODO adapt this to real clinical data (currently numerical fields only)
#         cols = [f"field{i}" for i in range(1, 31)]
#         self.clin = torch.tensor(df[cols].values, dtype=torch.float32)

#     def __len__(self):
#         return self.n

#     def __getitem__(self, idx):
#         item = self.base_ds[idx]                # base dataset as before
#         item["clinical_data"] = self.clin[idx]  # clinical data
#         return item