import torch
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


def main():
    args = parser.parse_args()
    load_args(args.config, args)
    set_seed(args.seed)

    dataset = CompositionDataset(
        args.dataset_path,
        phase="train",
        split="compositional-split-natural",
    )
    model, optimizer = configure_model(args, dataset)
    model.cuda().train()
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
    batch = [item.cuda() if not isinstance(item, tuple) else item for item in batch]

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss = model(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite SLER loss: {loss}")
    loss.backward()

    print("losses", model.last_loss_dict)
    composer_grad = model.sler_composer[-1].weight.grad
    if composer_grad is None or not torch.isfinite(composer_grad).all():
        raise RuntimeError("SLER composer did not receive a finite gradient")
    if composer_grad.abs().sum().item() == 0:
        raise RuntimeError("SLER composer gradient is exactly zero")

    print("SLER_SMOKE_OK")
    print("composer_grad_l1", composer_grad.abs().sum().item())
    print("max_memory_mb", torch.cuda.max_memory_allocated() / 1024 ** 2)


if __name__ == "__main__":
    main()
