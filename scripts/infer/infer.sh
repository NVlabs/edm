torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=large_cifar_64_dual_out --seeds=0-63 --batch=64 \
    --network cifar-training-runs/00000-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch128-fp16/network-snapshot-007507.pkl \
    --img_resolution 64