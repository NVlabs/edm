import torch
from pathlib import Path


def get_linear_sampler_kwargs(sampler_kwargs, device):
    linear_sampler_kwargs = {
        "skip_method": sampler_kwargs.pop("skip_method"),
        "sigma_min": sampler_kwargs.pop("sigma_skip"),
    }
    ds_params_dir = Path(sampler_kwargs.pop("ds_params_dir")) # TODO: not needed in the Isotropic case
    for param_path in ds_params_dir.iterdir():
        param_name = str(param_path).split("/")[-1].split(".")[0]
        linear_sampler_kwargs[param_name] = \
            torch.load(param_path, map_location=device)
   
    sampler_kwargs["sigma_max"] = linear_sampler_kwargs["sigma_min"]
    return sampler_kwargs

def linear_sampler(*args, **kwargs):
    skip_method = kwargs.pop("skip_method")
    if skip_method == "gaussian":
        return gaussian_linear_sampler(*args)
    else:
        raise ValueError

def gaussian_linear_sampler(x_T, sigma_max, sigma_min, mu, U, lambdas):
    I = torch.eye(U.shape[0], device=U.device)
    add1 = (I - U @ U.T) * sigma_min / sigma_max
    
    coefs = (sigma_min**2 + lambdas) / (sigma_min**2 + lambdas).sqrt()
    add2 = (U * coefs[None]) @ U.T

    return mu + (add1 + add2) @ (x_T  - mu)
    