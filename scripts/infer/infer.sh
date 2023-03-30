torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_dual_out --seeds=0-63 --batch=64 \
    --network ffhq-training-runs/00048-ffhq-64x64_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-000627.pkl \
    --img_resolution 128