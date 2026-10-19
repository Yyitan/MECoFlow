import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from flags import parser
from model.configure_model import configure_model
from utils import load_args, set_seed


def main():
    args = parser.parse_args()
    load_args(args.config, args)
    if getattr(args, "use_sler", False):
        raise RuntimeError("legacy scoring check requires use_sler=false")
    set_seed(args.seed)

    dataset = CompositionDataset(
        args.dataset_path,
        phase="train",
        split="compositional-split-natural",
    )
    model, _ = configure_model(args, dataset)
    model.cuda().eval()
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    image = batch[0].cuda()

    with torch.no_grad():
        token_com, token_att, token_obj = model.csp.construct_token_tensors(model.train_pairs)
        com_text, att_text, obj_text = model.get_text_features(token_com, token_att, token_obj)
        att, obj, com, glb = model.image_encoder(image)
        legacy = (
            model.temp_logit * F.normalize(att, p=2, dim=-1) @ att_text.T,
            model.temp_logit * F.normalize(obj, p=2, dim=-1) @ obj_text.T,
            model.temp_logit * F.normalize(com, p=2, dim=-1) @ com_text.T,
            model.temp_logit * F.normalize(glb, p=2, dim=-1) @ com_text.T,
        )
        refactored = model.compute_branch_logits(
            image, com_text, att_text, obj_text, return_routes=False
        )[:4]

    differences = [float((old - new).abs().max()) for old, new in zip(legacy, refactored)]
    if max(differences) > 1e-6:
        raise RuntimeError(f"legacy scoring changed: {differences}")
    print("LEGACY_SCORING_OK", differences)


if __name__ == "__main__":
    main()
