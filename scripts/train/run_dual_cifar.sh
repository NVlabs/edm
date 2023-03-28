torchrun --standalone --nproc_per_node=8 train.py --outdir=cifar-training-runs \
    --data=datasets/cifar10-32x32.zip --arch=ddpmpp --mode="dual" --batch 128 \
    --fp16 True --model_config_path model_configs/cifar.yml