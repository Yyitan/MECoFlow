import sys

import torch


for path in sys.argv[1:]:
    gate_logits = torch.load(path, map_location="cpu", weights_only=True)[
        "sler_gate_logits"
    ]
    print(
        path,
        "logits=",
        [round(value, 6) for value in gate_logits.tolist()],
        "gates=",
        [round(value, 6) for value in torch.sigmoid(gate_logits).tolist()],
    )
