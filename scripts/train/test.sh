torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs \
    --data=datasets/ffhq-256x256.zip --arch=ddpmpp --mode="dual" --fp16 True --batch 1