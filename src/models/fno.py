"""
Fourier Neural Operator (FNO) Model Module
==========================================

This module implements the FNO architecture for 2D temperature and phase field prediction.
It typically includes the SpectralConv2d layers and the overall FNO2d model definition.
"""

import torch.nn as nn
from neuralop.models import FNO


class InductionFNO(nn.Module):
    """
    Wrapper around neuralop.models.FNO for Induction Hardening tasks.

    Args:
        n_modes (tuple): Number of Fourier modes to keep in (x, y) directions.
        hidden_channels (int): Width of the hidden layers.
        in_channels (int): Number of input channels (e.g., x, y, T_initial).
        out_channels (int): Number of output channels (e.g., T_final, Phase).
        n_layers (int): Number of Fourier Layers. Default is 4.

    """

    def __init__(
        self,
        n_modes: tuple[int, int] = (16, 16),
        hidden_channels: int = 32,
        in_channels: int = 3,
        out_channels: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()
        # Initialize the FNO model from neuraloperator library
        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

    def forward(self, x):
        """
        Forward pass handling dimension permutation.

        Args:
            x: Input tensor of shape [Batch, X, Y, In_Channels]

        Returns:
            Output tensor of shape [Batch, X, Y, Out_Channels]

        """
        # 1. Permute to [Batch, Channels, X, Y] for FNO library
        x = x.permute(0, 3, 1, 2)

        # 2. Forward pass
        out = self.model(x)

        # 3. Permute back to [Batch, X, Y, Channels] for consistency
        return out.permute(0, 2, 3, 1)
