import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch.nn.functional as F
from typing import Callable, Sequence, Type
from math import pi
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm
from diffusion.diff_utils import exists, groupby, default
from a_unet.apex import (
    AttentionItem,
    CrossAttentionItem,
    InjectChannelsItem,
    ModulationItem,
    ResnetItem,
    SkipCat,
    SkipModulate,
    XBlock,
    XUNet,
)

from a_unet import (
    TimeConditioningPlugin,
    ClassifierFreeGuidancePlugin,
    Module,
    T5Embedder,
)

""" Distributions """


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin


""" Diffusion Methods """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class Diffusion(nn.Module):
    """Interface used by different diffusion methods"""

    pass


class VDiffusion(Diffusion):
    def __init__(
            self, net: nn.Module, sigma_distribution: Distribution = UniformDistribution(), loss_fn: Any = F.mse_loss
    ):
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.loss_fn = loss_fn
        self.input_projection = Conv1d_with_init(2, 1, 1)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(self, x: Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device

        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise

        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        return self.loss_fn(v_pred, v_target)


""" Schedules """


class Schedule(nn.Module):
    """Interface used by different sampling schedules"""

    def forward(self, num_steps: int, device: torch.device) -> Tensor:
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __init__(self, start: float = 1.0, end: float = 0.0):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, num_steps: int, device: Any) -> Tensor:
        return torch.linspace(self.start, self.end, num_steps, device=device)


""" Samplers """


class Sampler(nn.Module):
    pass


class VSampler(Sampler):
    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, schedule: Schedule = LinearSchedule()):
        super().__init__()
        self.net = net
        self.schedule = schedule
        self.input_projection = Conv1d_with_init(2, 1, 1)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def forward(  # type: ignore
            self, x_noisy: Tensor, num_steps: int, show_progress: bool = False, **kwargs
    ) -> Tensor:
        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i + 1, 0]:.2f})")

        return x_noisy


class DiffusionModel(nn.Module):
    def __init__(
            self,
            net_t: Callable,
            diffusion_t: Callable = VDiffusion,
            sampler_t: Callable = VSampler,
            loss_fn: Callable = torch.nn.functional.mse_loss,
            dim: int = 1,
            **kwargs,
    ):
        super().__init__()
        diffusion_kwargs, kwargs = groupby("diffusion_", kwargs)
        sampler_kwargs, kwargs = groupby("sampler_", kwargs)

        self.net = net_t(dim=dim, **kwargs)
        self.diffusion = diffusion_t(net=self.net, loss_fn=loss_fn, **diffusion_kwargs)
        self.sampler = sampler_t(net=self.net, **sampler_kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.diffusion(*args, **kwargs)

    def sample(self, *args, **kwargs) -> Tensor:
        return self.sampler(*args, **kwargs)


def TextConditioningPlugin11(
        net_t: Type[nn.Module], embedder: Optional[nn.Module] = None
) -> Callable[..., nn.Module]:
    """Adds text conditioning"""
    embedder = embedder if exists(embedder) else T5Embedder()
    # msg = "TextConditioningPlugin embedder requires embedding_features attribute"
    # assert hasattr(embedder, "embedding_features"), msg
    features: int = embedder.embedding_features  # type: ignore

    def Net(embedding_features: int = features, **kwargs) -> nn.Module:
        # msg = f"TextConditioningPlugin requires embedding_features={features}"
        # assert embedding_features == features, msg
        net = net_t(embedding_features=embedding_features, **kwargs)  # type: ignore

        def forward(
                x: Tensor, text: Sequence[str], embedding: Optional[Tensor] = None, **kwargs
        ):
            text_embedding = embedding
            # if exists(embedding):
            #     text_embedding = torch.cat([text_embedding, embedding], dim=1)
            return net(x, embedding=text_embedding, **kwargs)

        return Module([embedder, net], forward)  # type: ignore

    return Net


def UNetV0(
        dim: int,
        in_channels: int,
        channels: Sequence[int],
        factors: Sequence[int],
        items: Sequence[int],
        attentions: Optional[Sequence[int]] = None,
        cross_attentions: Optional[Sequence[int]] = None,
        context_channels: Optional[Sequence[int]] = None,
        attention_features: Optional[int] = None,
        attention_heads: Optional[int] = None,
        embedding_features: Optional[int] = None,
        resnet_groups: int = 8,
        use_modulation: bool = True,
        modulation_features: int = 1024,
        embedding_max_length: Optional[int] = None,
        use_time_conditioning: bool = True,
        use_embedding_cfg: bool = False,
        use_text_conditioning: bool = False,
        out_channels: Optional[int] = None,
):
    # Set defaults and check lengths
    num_layers = len(channels)
    attentions = default(attentions, [0] * num_layers)
    cross_attentions = default(cross_attentions, [0] * num_layers)
    context_channels = default(context_channels, [0] * num_layers)
    xs = (channels, factors, items, attentions, cross_attentions, context_channels)
    assert all(len(x) == num_layers for x in xs)  # type: ignore

    # Define UNet type
    UNetV0 = XUNet

    if use_embedding_cfg:
        msg = "use_embedding_cfg requires embedding_max_length"
        # assert exists(embedding_max_length), msg
        # UNetV0 = LanguageShapeletGuidancePlugin(UNetV0, embedding_max_length)
        assert exists(embedding_max_length), msg
        UNetV0 = ClassifierFreeGuidancePlugin(UNetV0, embedding_max_length)

    if use_text_conditioning:
        UNetV0 = TextConditioningPlugin11(UNetV0)

    if use_time_conditioning:
        assert use_modulation, "use_time_conditioning requires use_modulation=True"
        UNetV0 = TimeConditioningPlugin(UNetV0)

    # Build
    return UNetV0(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                context_channels=ctx_channels,
                items=(
                              [ResnetItem]
                              + [ModulationItem] * use_modulation
                              + [InjectChannelsItem] * (ctx_channels > 0)
                              + [AttentionItem] * att
                              + [CrossAttentionItem] * cross
                      )
                      * items,
            )
            for channels, factor, items, att, cross, ctx_channels in zip(*xs)  # type: ignore # noqa
        ],
        skip_t=SkipModulate if use_modulation else SkipCat,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
    )
