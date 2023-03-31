torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-63 --batch=64 \
    --network ffhq-training-runs/00057-ffhq-128x128_ffhq-96x96_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-023206.pkl \
    --img_resolution 34