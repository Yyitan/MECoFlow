import torch
from pathlib import Path
for p in ["output/research/piflow_mit/features.pt", "output/research/warmflow_mit_gaussian_iso_seed0/classifier_logits_residual_0.50.pt"]:
    d = torch.load(p, map_location="cpu", weights_only=False)
    if "val" in d:
        print(p, d["val"]["logits"].shape)
    else:
        print(p, d.keys())
