torchrun --standalone --nproc_per_node=1 train.py --outdir=test-training-runs \
    -d datasets/ffhq-64x64.zip -d datasets/ffhq-32x32.zip \
    --arch=ddpmpp --mode="dual" --batch 8 \
    --model_config_path model_configs/cifar.yml \
    --lr 1e-5 --tick 1 --snap 50000 \
    -v