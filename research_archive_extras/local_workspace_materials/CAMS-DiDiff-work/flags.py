import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='CAMS/configs/utzappos.yml', help='Path of the config file') 
parser.add_argument('--dataset_path', default='ut-zap50k', help='Path of the dataset')
parser.add_argument('--splitname', default='compositional-split-natural', help="Dataset split")
parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-5) #1e-5
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")

parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--threshold', default=None,help="Apply a specific threshold at test time for the hard masking")
parser.add_argument('--threshold_trials', type=int, default=50,help="how many threshold values to try")
parser.add_argument('--use_diffusion_refine', action='store_true', default=False, help='enable candidate diffusion refinement')
parser.add_argument('--diff_steps', type=int, default=4, help='number of refinement steps')
parser.add_argument('--diff_train_topk', type=int, default=64, help='top-k train candidates refined by diffusion head')
parser.add_argument('--diff_eval_topk', type=int, default=128, help='top-k eval candidates refined by diffusion head')
parser.add_argument('--diff_hidden_dim', type=int, default=512, help='hidden size of diffusion refinement head')
parser.add_argument('--diff_time_dim', type=int, default=128, help='time embedding size of diffusion refinement head')
parser.add_argument('--diff_dropout', type=float, default=0.1, help='dropout of diffusion refinement head')
parser.add_argument('--diff_step_size', type=float, default=1.0, help='residual update step size of diffusion refinement')
parser.add_argument('--diff_loss_weight', type=float, default=0.2, help='training loss weight for diffusion refinement')
parser.add_argument('--diff_zero_mean_delta', action='store_true', default=False, help='make candidate deltas zero-mean before applying refinement')
parser.add_argument('--diff_use_gate', action='store_true', default=False, help='enable per-pair residual gate in diffusion refinement')
parser.add_argument('--diff_mix', type=float, default=1.0, help='mixing weight for refined logits; 0 keeps CAMS logits, 1 uses full refinement')
parser.add_argument('--diff_delta_center', action='store_true', default=False, help='center refined-logit delta before mixing it into CAMS logits')
