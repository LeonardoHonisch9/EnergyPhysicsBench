"""1D heat diffusion: analytical, finite-difference, and neural-network solvers."""

from .config import Config, DataDrivenConfig, FDMConfig, ICParams, PINNConfig, default_config
from .analytical import analytical_pointwise, analytical_solution, fourier_coefficients
from .fdm import FDMResult, fdm_solve
from .models import HeatNet
from .training import TrainResult, train_data_driven, train_pinn
from .evaluate import ErrorMetrics, error_metrics, evaluate_net
from .initial_conditions import ic_function, ic_np, u_steady

__version__ = "0.1.0"

__all__ = [
    "Config", "DataDrivenConfig", "FDMConfig", "ICParams", "PINNConfig", "default_config",
    "analytical_pointwise", "analytical_solution", "fourier_coefficients",
    "FDMResult", "fdm_solve",
    "HeatNet",
    "TrainResult", "train_data_driven", "train_pinn",
    "ErrorMetrics", "error_metrics", "evaluate_net",
    "ic_function", "ic_np", "u_steady",
]