"""Typed configuration for the 1D heat diffusion experiments.

Using dataclasses (instead of a plain nested dict, as in the original notebook)
gives autocomplete/type-checking in an IDE and a single source of truth for
defaults. Every sub-config can be overridden independently, e.g.::

    from heat_diffusion1D.config import Config, FDMConfig
    cfg = Config(alpha=0.05, fdm=FDMConfig(nx=201, scheme="implicit"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ICParams:
    """Parameters for the available initial-condition shapes."""
    amplitude: float = 1.0
    n_mode: int = 1        # used by "sine": f(x) = amplitude * sin(n_mode * pi * x / L)
    center: float = 0.5    # used by "gaussian"
    width: float = 0.08    # used by "gaussian"
    x0: float = 0.4        # used by "step"
    x1: float = 0.6        # used by "step"


@dataclass
class FDMConfig:
    nx: int = 101
    nt: int = 4000
    scheme: Literal["explicit", "implicit"] = "explicit"


@dataclass
class PINNConfig:
    """Physics-informed network: trained on the PDE residual + IC + BC only.
    No solution data is used — this is the 'ML learns the physics' path."""
    hidden_layers: int = 4
    neurons: int = 32
    activation: Literal["tanh", "silu", "sigmoid"] = "tanh"
    n_collocation: int = 4000
    n_ic: int = 300
    n_bc: int = 300
    epochs: int = 6000
    lr: float = 1e-3
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 10.0
    seed: int = 0


@dataclass
class DataDrivenConfig:
    """Purely supervised baseline: trained on (x, t, u) samples of the
    analytical solution, with no PDE knowledge at all."""
    hidden_layers: int = 4
    neurons: int = 32
    activation: Literal["tanh", "silu", "sigmoid"] = "tanh"
    n_train: int = 300
    noise_std: float = 0.0
    t_max_train: Optional[float] = None   # None = sample labels across the full [0, T]
    epochs: int = 4000
    lr: float = 1e-3
    seed: int = 0


@dataclass
class Config:
    # --- Physics ---
    L: float = 1.0
    alpha: float = 0.01
    T: float = 2.0
    bc_left: float = 0.0
    bc_right: float = 0.0

    # --- Initial condition ---
    ic_type: Literal["sine", "gaussian", "step"] = "sine"
    ic_params: ICParams = field(default_factory=ICParams)

    # --- Analytical (Fourier sine series) ---
    n_modes: int = 100

    # --- Sub-configs for each solution method ---
    fdm: FDMConfig = field(default_factory=FDMConfig)
    pinn: PINNConfig = field(default_factory=PINNConfig)
    data_driven: DataDrivenConfig = field(default_factory=DataDrivenConfig)


def default_config() -> Config:
    """Convenience factory, mirrors the defaults used in the exploratory notebook."""
    return Config()