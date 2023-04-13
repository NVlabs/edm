#torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-63 --batch=64 \
#    --network ffhq-training-runs/00057-ffhq-128x128_ffhq-96x96_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-023206.pkl \
#    --img_resolution 34

import torch
from torch.nn.functional import pad

def resize_to_shape(x_ft,dim1,dim2):
    d1,d2 = x_ft.shape
    if d1 < dim1:
        x_ft = pad(x_ft,(
                        0, dim1- d1,
                        0, dim2 - d2))
        print(x_ft)
    else : 
        x_ft = x_ft[:dim1, :dim2]
    return x_ft

n  = 64 

a = torch.rand((4,4))
a = torch.fft.rfft2(a)
print(a.shape)
print(resize_to_shape(a,32,17).shape)
