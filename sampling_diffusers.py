from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32")

print(pipeline.unet)