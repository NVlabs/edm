torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \
    --data=datasets/ffhq-256x256.zip --arch=ddpmpp --mode="dual" --batch 16 \
    --fp16 True