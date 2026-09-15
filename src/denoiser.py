# --------------------------------------------------------
# References:
# JiT: https://github.com/LTH14/JiT
# --------------------------------------------------------
import torch
import torch.nn as nn

from copy import deepcopy

from model import JiT


def to_v(param, pred, z_t, t, t_min):
    # helper fcn get pred to v_pred for v_loss
    if param == 'x':
        x_theta = pred
        v_theta = (x_theta - z_t) / (1 - t).clamp(min=t_min)
    if param == 'eps':
        eps_theta = pred
        v_theta = (z_t - eps_theta) / t.clamp(min=t_min)
    if param == 'v':
        v_theta = pred
    return v_theta


class Denoiser(nn.Module):
    def __init__(self, config, device, param='x'):
        super().__init__()

        self.model = JiT(**config['model_args'])

        self.param = param

        self.img_size = config["img_size"]

        self.ema_decay = config["ema_decay"]
        self.ema = deepcopy(self.model).requires_grad_(False)

        self.method = config["sampling_method"]
        self.steps = config["num_sampling_steps"]
        self.mu = config["mu"]
        self.sigma = config["sigma"]
        self.noise_scale = config["noise_scale"]
        self.t_min = config["t_min"]

        self.device = device

    def forward(self, x):
        B, C, H, W = x.shape
        t = torch.sigmoid(torch.randn(B, device=x.device) * self.sigma + self.mu).view(-1, 1, 1, 1)
        eps = self.noise_scale * torch.randn_like(x)
        z_t = t * x + (1 - t) * eps

        # our dataset has no labels. Make "fake labels"
        fake_labels = torch.zeros(B, dtype=torch.long, device=self.device)
        # only the network runs in fp16; noise mixing, to_v and the loss stay fp32
        with torch.autocast(x.device.type, dtype=torch.float16):
            pred = self.model(z_t, t.flatten(), fake_labels)
        v_true = x - eps
        v_pred = to_v(self.param, pred.float(), z_t, t, self.t_min)

        loss = (v_true - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    @torch.no_grad()
    def generate(self, B):
        fake_labels = torch.zeros(B, dtype=torch.long, device=self.device)

        z = self.noise_scale * torch.randn(B, 3, self.img_size, self.img_size, device=self.device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=self.device).view(-1, *([1] * z.ndim)).expand(-1, B, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, fake_labels)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], fake_labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, fake_labels):
        with torch.autocast(z.device.type, dtype=torch.float16):
            pred = self.ema(z, t.flatten(), fake_labels)
        v_pred = to_v(self.param, pred.float(), z, t, self.t_min)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        for e, p in zip(self.ema.parameters(), self.model.parameters()):
            e.lerp_(p, 1 - self.ema_decay)