import requests
import torch

import numpy as np
from torch.utils.data import Dataset
from osgeo import gdal
from skimage import io
from PIL import Image
from tqdm import tqdm


def download_file(url, path):
    local_filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 4096
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with requests.get(url, stream=True) as r:
        with open(path + local_filename, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    progress_bar.close()
    return local_filename


def onehot(i, num_classes):
    v = [0] * int(num_classes)
    v[i] = 1
    return v


def read_geotiff(path):
    # Directly read the tiff data skimage and gdal. Somehow dtype=uint8.
    # Import as_gray=False to avoid float64 conversion.
    img = io.imread(path, as_gray=False, plugin="gdal")
    cs = gdal.Open(path)

    return img, cs


class CTX_Image(Dataset):
    """CTX dataset."""

    def __init__(self, path, window_size=200, transform=None, cutout=None):
        self.transform = transform
        self.path = path
        self.image_full, self.cs = read_geotiff(path)
        self.window_size = window_size

        # Crop image according to values in crop
        if cutout is not None:
            self.cutout(cutout)

        # Get shapes of "new" full image
        self.image_size_full = np.shape(self.image_full)

        self.num_tiles_full = np.ceil(
            np.array(self.image_size_full) / self.window_size
        ).astype("int")

        wd = self.image_size_full[0]
        hd = self.image_size_full[1]
        # create new image of desired size and color (blue) for padding
        ww, hh = window_size * self.num_tiles_full
        # hh = window_size * self.num_tiles_full[1]

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - hd) // 2

        # copy img image into center of result image
        self.padded_full = np.zeros(
            tuple((self.num_tiles_full * self.window_size).astype("int")),
            dtype=np.uint8,
        )
        self.padded_full[xx : xx + wd, yy : yy + hd] = self.image_full

        # self.padded_full[:self.image_size_full[0], :self.image_size_full[1]] = self.image_full

        step_size_full = 1
        idx_tiles_full_a = np.rint(
            np.arange(0, self.num_tiles_full[0] * self.window_size, step_size_full)
        ).astype("int")
        idx_tiles_full_b = np.rint(
            np.arange(0, self.num_tiles_full[1] * self.window_size, step_size_full)
        ).astype("int")

        self.idx_tiles_full_a = idx_tiles_full_a[
            idx_tiles_full_a + self.window_size
            < self.num_tiles_full[0] * self.window_size
        ]
        self.idx_tiles_full_b = idx_tiles_full_b[
            idx_tiles_full_b + self.window_size
            < self.num_tiles_full[1] * self.window_size
        ]

        self.num_full = np.array(
            [self.idx_tiles_full_a.__len__(), self.idx_tiles_full_b.__len__()]
        )
        self.out_shape = (
            self.idx_tiles_full_a.__len__(),
            self.idx_tiles_full_b.__len__(),
        )

    def __len__(self):
        return np.prod(self.num_full)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_in_res = idx
        idx_aa, idx_bb = np.unravel_index(idx_in_res, self.num_full)
        idx_a = self.idx_tiles_full_a[idx_aa]
        idx_b = self.idx_tiles_full_b[idx_bb]
        image = self.padded_full[
            idx_a : idx_a + self.window_size, idx_b : idx_b + self.window_size
        ]
        center_pixel = image[self.window_size // 2, self.window_size // 2]
        image = np.dstack([image] * 3)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, center_pixel

    def get_image(self):
        return self.image_full

    def cutout(self, crop):
        self.crop_image(crop)

    def crop_image(self, crop):
        self.image_full = self.image_full[crop[1] : crop[3], crop[0] : crop[2]]
