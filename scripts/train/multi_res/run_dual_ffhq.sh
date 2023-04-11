torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    -d /mnt/nvme/home/alex/repos/diffusion/edm/datasets/ffhq-64x64.zip -d /mnt/nvme/home/alex/repos/diffusion/edm/datasets/ffhq-48x48.zip -d /mnt/nvme/home/alex/repos/diffusion/edm/datasets/ffhq-32x32.zip \
     --arch=ddpmpp --mode="dual" --batch 256 \
    --model_config_path model_configs/cifar.yml \
    --lr 1e-3 --tick 25 --snap 25