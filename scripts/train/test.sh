torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch 16 --mode="dual"