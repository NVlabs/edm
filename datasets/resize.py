import torch
import zipfile
import PIL.Image
import numpy as np
from torchvision.transforms import PILToTensor, ToPILImage, Resize
import torch.nn.functional as F
import pathlib
import os
from tqdm import tqdm
import argparse
import multiprocessing as mp


def downscale(zip_file_name, resize, file_names, out_folder):
    file = zipfile.ZipFile(zip_file_name)
    for i, file_name in tqdm(enumerate(file_names)):
        try:
            im_file = file.open(file_name, "r")
            image = PIL.Image.open(im_file)
            image = resize(image)
            if i == 0:
                image.save("temp.png")
            image.save(os.path.join(out_folder, os.path.basename(file_name)))
        except PIL.UnidentifiedImageError as e:
            print(f"Error with file: {file_name}")
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ffhq", type=str)
    parser.add_argument("--size", default=32, type=int)
    parser.add_argument("--num_procs", default=24, type=int)
    parser.add_argument("--input_file_name", default="ffhq-256x256.zip", type=str)
    parser.add_argument("--output_folder_name", default=None, type=str)
    args = parser.parse_args()
    file_name = args.input_file_name
    p_to_t = PILToTensor()
    t_to_p = ToPILImage()
    size = args.size
    resize = Resize(size)
    out_folder = "{}_{}x{}".format(args.dataset_name, size, size) if args.output_folder_name is None else args.output_folder_name
    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
    print("Resizing {} to {}...".format(file_name, args.size))
    print(f"Putting results in {out_folder}")

    file = zipfile.ZipFile(file_name)
    file_names = list(filter(lambda f: ".png" in f, file.namelist()))
    batch_size = len(file_names) // args.num_procs + args.num_procs
    file_name_batches = [file_names[i * batch_size : (i+1) * batch_size] for i in range(args.num_procs)]

    procs = []
    for file_names in file_name_batches:
        p = mp.Process(target=downscale, args=[file_name, resize, file_names, out_folder])
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print("Num out files: {}".format(len(os.listdir(out_folder))))