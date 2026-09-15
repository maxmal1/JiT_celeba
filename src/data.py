import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

ROOT = "/home/ubuntu/data/CelebA"


def build_memmap(root, size, path):
    tf = transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize(size),
        transforms.PILToTensor(),
    ])
    ds = datasets.CelebA(root, split="all", transform=tf, download=True)
    loader = DataLoader(ds, batch_size=512, num_workers=8)

    tmp = path + ".tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(ds), size, size, 3))
    i = 0
    for x, _ in tqdm(loader, desc=f"building {os.path.basename(path)}"):
        out[i:i + len(x)] = x.permute(0, 2, 3, 1).numpy()
        i += len(x)
    out.flush()
    del out
    os.replace(tmp, path)


class CelebA(Dataset):
    def __init__(self, size, root=ROOT, save_dir=ROOT):
        path = os.path.join(save_dir, f"celeba_{size}.npy")
        if not os.path.exists(path):
            build_memmap(root, size, path)
        self.data = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx]))
        return x.movedim(-1, -3)


def normalize(x):
    return x.float() / 127.5 - 1
