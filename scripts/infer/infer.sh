torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=cifar_out/64 --seeds=0-10 --batch=64 \
    --network cifar-training-runs/00000-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch128-fp16/network-snapshot-006256.pkl \
    --img_resolution 64