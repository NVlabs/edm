import tensorly as tl
from tensorly.plugins import use_opt_einsum
import torch
tl.set_backend('pytorch')

use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...
    
    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


torch.cuda.empty_cache()

in_channels, out_channels, modes1, modes2 = 256, 256, 32, 17
scale = 1 / (in_channels * out_channels)
rank, factorization = 0.1, "cp"
weight_shape = (in_channels, out_channels, modes1, modes2, 2)
cp_weights = FactorizedTensor.new(weight_shape, rank=rank, factorization=factorization).cuda()
cp_weights.normal_(0, scale)


x = torch.randn(1, in_channels, modes1, modes2, 2).cuda()
import time
start = time.time()
out = _contract_cp(x, cp_weights)
end = time.time()
print("cp factorized time forward time: ", end - start)
print(out.shape)
t = torch.cuda.get_device_properties(0).total_memory / 1e9
r = torch.cuda.memory_reserved(0) / 1e9
a = torch.cuda.memory_allocated(0) / 1e9
print("Total gpu mem: ", t)
print("Reserved gpu mem: ", r)
print("Allocated gpu mem: ", a)

torch.cuda.empty_cache()

factorization = "tucker"
tucker_weights = FactorizedTensor.new(weight_shape, rank=rank, factorization=factorization).cuda()
tucker_weights.normal_(0, scale)

import time
start = time.time()
out = _contract_tucker(x, tucker_weights)
end = time.time()
print("tucker factorized time forward time: ", end - start)
print(out.shape)
t = torch.cuda.get_device_properties(0).total_memory / 1e9
r = torch.cuda.memory_reserved(0) / 1e9
a = torch.cuda.memory_allocated(0) / 1e9
print("Total gpu mem: ", t)
print("Reserved gpu mem: ", r)
print("Allocated gpu mem: ", a)

torch.cuda.empty_cache()

def compl_mul2d(input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights)


weights = torch.nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)).cuda()
start = time.time()
out = compl_mul2d(x, weights)
end = time.time()
print("default impl forward time: ", end - start)
print(out.shape)
t = torch.cuda.get_device_properties(0).total_memory / 1e9
r = torch.cuda.memory_reserved(0) / 1e9
a = torch.cuda.memory_allocated(0) / 1e9
print("Total gpu mem: ", t)
print("Reserved gpu mem: ", r)
print("Allocated gpu mem: ", a)