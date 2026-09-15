"""
python -m src.export train --n 50000 --out ~/data/CelebA/fid_train
python -m src.export val --out ~/data/CelebA/fid_val

python -m src.export run_dir/run --n 10000
"""

import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from data import CelebA
from denoiser import Denoiser


def make_out_dir(path):
    os.makedirs(path, exist_ok=True)
    assert not os.listdir(path), f"{path} is not empty; stale images would skew FID"


def save_pngs(batch, out_dir, start):
    for i, img in enumerate(batch.permute(0, 2, 3, 1).numpy()):
        Image.fromarray(img).save(os.path.join(out_dir, f"{start + i:06d}.png"))


def export_real(split, n, out_dir, size, n_val, seed):
    ds = CelebA(size)
    n_train = len(ds) - n_val
    if split == "train":
        idx = np.sort(np.random.default_rng(seed).choice(n_train, n, replace=False))
    else:
        idx = np.arange(n_train, len(ds))
    for s in tqdm(range(0, len(idx), 1000), desc=split):
        save_pngs(ds[idx[s:s + 1000]], out_dir, s)


@torch.no_grad()
def export_run(run_dir, n, out_dir, batch_size, seed):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    denoiser = Denoiser(config, "cuda", config["param"])
    denoiser.load_state_dict(torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")["denoiser"])
    denoiser = denoiser.cuda().eval()

    torch.manual_seed(seed)
    for s in tqdm(range(0, n, batch_size), desc=os.path.basename(run_dir)):
        # always generate a full batch so torch.compile doesn't recompile for the last one
        z = denoiser.generate(batch_size)[:n - s]
        save_pngs(((z.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu(), out_dir, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="'train' or 'val' for real images, otherwise a run dir")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--out", help="output dir; defaults to <run_dir>/fid_samples for runs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.target in ("train", "val"):
        assert args.out, "--out is required for real images"
        make_out_dir(args.out)
        export_real(args.target, args.n, args.out, args.size, args.n_val, args.seed)
    else:
        out = args.out or os.path.join(args.target, "fid_samples")
        make_out_dir(out)
        export_run(args.target, args.n, out, args.batch_size, args.seed)
