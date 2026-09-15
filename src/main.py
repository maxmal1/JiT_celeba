import argparse
import csv
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from data import CelebA, normalize
from denoiser import Denoiser


def get_batch(ds, idx, device):
    # used for val only; pinned memory makes the host->gpu copy much faster
    return normalize(ds[idx].pin_memory().to(device, non_blocking=True))


@torch.no_grad()
def val_loss(denoiser, ds, n_train, batch_size, device):
    # fixed seed so t and eps are the same every eval; fork_rng leaves the training rng untouched
    with torch.random.fork_rng(devices=[torch.device(device)]):
        torch.manual_seed(0)
        losses = [denoiser(get_batch(ds, np.arange(i, min(i + batch_size, len(ds))), device)).item()
                  for i in range(n_train, len(ds), batch_size)]
    return float(np.mean(losses))


@torch.no_grad()
def save_samples(denoiser, path, n, device):
    with torch.random.fork_rng(devices=[torch.device(device)]):
        torch.manual_seed(0)
        z = denoiser.generate(n)
    grid = make_grid((z.clamp(-1, 1) + 1) / 2, nrow=int(n ** 0.5))
    save_image(grid, path)
    return grid


def make_run_dir(out_dir, base, resume):
    # new runs get the next _vN; --resume picks the latest existing version
    prefix = f"{base}_v"
    names = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    versions = sorted(int(n[len(prefix):]) for n in names if n.startswith(prefix) and n[len(prefix):].isdigit())
    if resume:
        assert versions, f"no existing run to resume for {base} in {out_dir}"
        v = versions[-1]
    else:
        v = versions[-1] + 1 if versions else 1
    return os.path.join(out_dir, f"{prefix}{v}")


def main(config, param, resume=False):
    assert param in ["x", "v", "eps"], f"{param} must be one of x, v, eps"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    run_dir = make_run_dir(config["out_dir"], f"{config['name']}_{param}", resume)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**config, "param": param}, f)

    ds = CelebA(config["img_size"])
    n_train = len(ds) - config["n_val"]
    # sample train indices with replacement from an effectively endless sampler: no epoch boundaries
    sampler = RandomSampler(range(n_train), replacement=True, num_samples=10**9)
    loader = iter(DataLoader(ds, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"],
                             pin_memory=True, prefetch_factor=4, persistent_workers=True))

    denoiser = Denoiser(config, device, param).to(device)
    opt = torch.optim.AdamW(
        denoiser.model.parameters(), 
        lr=config["lr"], 
        betas=(0.9, 0.95), 
        weight_decay=config["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / config["warmup_steps"]))
    scaler = torch.amp.GradScaler()

    start = 0
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        denoiser.load_state_dict(ckpt["denoiser"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start = ckpt["step"]
        print(f"resumed from step {start}")

    # overwrite when starting from step 0 so a resumed run without a checkpoint doesn't duplicate the header
    log_file = open(os.path.join(run_dir, "loss.csv"), "a" if start else "w", newline="")
    log = csv.writer(log_file)
    if start == 0:
        log.writerow(["step", "train_loss", "val_loss"])
    # event files live in the run dir; purge_step drops points logged after the last checkpoint of a crashed run
    # (the checkpoint step itself was logged before saving, so keep it)
    writer = SummaryWriter(run_dir, purge_step=start + 1 if start else None, flush_secs=30)

    running = torch.zeros((), device=device)
    pbar = tqdm(range(start, config["steps"]), initial=start, total=config["steps"], desc=os.path.basename(run_dir))
    for step in pbar:
        x = normalize(next(loader).to(device, non_blocking=True))
        loss = denoiser(x)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(denoiser.model.parameters(), config["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        denoiser.update_ema()
        running += loss.detach()

        step += 1
        if step % config["log_every"] == 0:
            train = running.item() / config["log_every"]
            running.zero_()
            val = val_loss(denoiser, ds, n_train, config["batch_size"], device) if step % config["val_every"] == 0 else ""
            log.writerow([step, train, val])
            log_file.flush()
            writer.add_scalar("loss/train", train, step)
            if val != "":
                writer.add_scalar("loss/val", val, step)
            pbar.set_postfix(train=f"{train:.4f}", val=f"{val:.4f}" if val != "" else "-")

        if step % config["sample_every"] == 0 or step == config["steps"]:
            grid = save_samples(denoiser, os.path.join(run_dir, f"samples_{step:06d}.png"), config["n_samples"], device)
            writer.add_image("samples", grid, step)

        if step % config["save_every"] == 0 or step == config["steps"]:
            torch.save({"denoiser": denoiser.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step}, ckpt_path)

    log_file.close()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--param", type = str, help = "Parameter to evaluate. Can be one of x, v, eps")
    parser.add_argument("--resume", action="store_true", help="Resume the latest version of this name/param")

    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.param, args.resume)
