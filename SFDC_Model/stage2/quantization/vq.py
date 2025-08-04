# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

from dataclasses import dataclass, field
import math
import typing as tp

import torch
from torch import nn

from .core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
        if you want to know more information about RVQ, you can read the soundstream paper (https://arxiv.org/abs/2107.03312)
        Residual vector quantizer cascades N_q layers of VQ. 
        the algorithm is described as follows:
        **********************************************************************************
        Input: y = enc(x) the output of the encoder, vector quantizers Q_i for i = 1...N_q
        Output: the quantized y^hat
        
        y^hat <- 0 
        residual <- y
        for i=1 to N_q do
            y^hat += Q_i(residual)
            residual -= Q_i(residual)
        return y^hat

        **********************************************************************************
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        # bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
        # n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        quantized, codes, commit_loss = self.vq(x, self.n_q)
        # bw = torch.tensor(n_q * bw_per_q).to(x)
        return quantized, codes, torch.mean(commit_loss)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码时固定使用全部量化层"""
        return self.vq.encode(x, n_q=self.n_q)
   

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes) # vq.decode output -> quantized_out
        return quantized
