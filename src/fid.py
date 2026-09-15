"""
python -m src.fid runs_x runs_v runs_eps
"""
import argparse
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.io import decode_image
from tqdm.auto import tqdm


def add_folder(fid, folder, real, batch_size):
    paths = sorted(Path(folder).glob("*.png"))
    assert paths, f"no pngs in {folder}"
    for i in tqdm(range(0, len(paths), batch_size), desc=str(folder)):
        fid.update(torch.stack([decode_image(str(p)) for p in paths[i:i + batch_size]]).cuda(), real=real)
    return len(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run dirs (or image folders with --subdir '')")
    parser.add_argument("--real", default=str(Path.home() / "data/CelebA/fid_train"))
    parser.add_argument("--subdir", default="fid_samples", help="sample folder inside each run dir")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).cuda()
    n_real = add_folder(fid, args.real, True, args.batch_size)
    for run in args.runs:
        fid.reset()
        n = add_folder(fid, Path(run) / args.subdir, False, args.batch_size)
        print(f"{run}: FID-{n // 1000}K = {fid.compute().item():.2f} (vs {n_real} real)")
