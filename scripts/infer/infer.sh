torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=cifar_64_def_out --seeds=0-49999 --batch=64 \
    --network training-runs/00062-cifar10-32x32-cond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-000000.pkl \
    --img_resolution 32