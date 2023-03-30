torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d datasets/ffhq-64x64.zip -d datasets/ffhq-32x32.zip \
     --arch=ddpmpp --mode="def" --batch 256 \
    --model_config_path model_configs/cifar.yml \
    --lr 1e-3 --tick 25 --snap 25