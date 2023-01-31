import numpy as np
import os
import PIL.Image
from tqdm import tqdm

workdir = "out"
cnt = 0
imdir = "im_out"
os.makedirs(imdir, exist_ok=True)
for file in tqdm(os.listdir(workdir)):
    print("Loading {}...".format(os.path.join(workdir, file)))
    ims = np.load(os.path.join(workdir, file))
    for im in ims:
        image_path = os.path.join(imdir, "img_{}.png".format(cnt))
        PIL.Image.fromarray(im, 'RGB').save(image_path)
        cnt += 1
print("Total ims processed: {}".format(cnt))
