torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_dual_out --seeds=0-63 --batch=64 \
    --network ffhq-training-runs/00041-ffhq-256x256-uncond-ddpmpp-edm-gpus8-batch32-fp32/network-snapshot-000750.pkl \
    --img_resolution 128