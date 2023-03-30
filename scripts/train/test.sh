torchrun --standalone --nproc_per_node=1 train.py --outdir=ffhq-training-runs \
    --data=datasets/ffhq-256x256.zip --arch=ddpmpp --mode="dual" --batch 8 \
    --model_config_path model_configs/cifar.yml \
    --lr 1e-5 --tick 1 --snap 50000 \
    -v