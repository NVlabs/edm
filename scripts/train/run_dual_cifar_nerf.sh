torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --mode="dual" --batch 256 --n_res_blocks 12 --random_fourier