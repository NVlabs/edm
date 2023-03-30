torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    --data=datasets/ffhq-256x256.zip --arch=ddpmpp --mode="dual" --batch 32 \
    --model_config_path model_configs/ffhq.yml \
    --lr 1e-3 --tick 25 --snap 10