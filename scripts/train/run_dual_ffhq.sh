torchrun --standalone --nproc_per_node=8 train.py --outdir=ffhq-training-runs \
    --data=datasets/ffhq-256x256.zip --arch=ddpmpp --mode="dual" --batch 128 \
    --fp16 True --model_config_path model_configs/cifar.yml \
    --lr 1e-4