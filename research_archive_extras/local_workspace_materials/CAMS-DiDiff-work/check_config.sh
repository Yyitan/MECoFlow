cd ./lyt/czsl/CAMS-DiDiff
CUDA_VISIBLE_DEVICES=1 python3 -c "from flags import parser; from utils import load_args; args=parser.parse_args(['--config','configs/mit-states-didiff.yml']); load_args(args.config,args); print(args.use_diffusion_refine, args.diff_steps, args.diff_train_topk, args.diff_eval_topk, args.save_path)"
