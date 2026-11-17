"""Orientation-map encodings used by project 017.

The direct Euler-to-RGB mode is included for paper reproduction.  The
sin/cos mode is safer around angular wrap-around and is the recommended mode
for the first experiment once raw Euler maps are available.

``mode='sincos'`` still treats every angle as a full turn. ``mode='euler_gb'``
is the cubic reduced-zone encoding: phi1 has period 360 degrees, phi2 has
period 90 degrees, and Phi is either sin/cos or a linear map. The boundary
channel is optional and is not invented from the angles.
"""
from __future__ import annotations

import math

import torch


def euler_to_channels(
    euler: torch.Tensor,
    mode: str = "sincos",
    phi_encoding: str = "sincos",
    boundary: torch.Tensor | None = None,
    phi_span_degrees: float = 90.0,
) -> torch.Tensor:
    """Convert BCHW Euler angles in radians to normalized image channels."""
    if euler.ndim != 4 or euler.shape[1] != 3:
        raise ValueError("Expected Euler tensor with shape Bx3xHxW")
    if mode == "rgb":
        return (euler.remainder(2 * torch.pi) / (2 * torch.pi)) * 2 - 1
    if mode == "sincos":
        return torch.cat((torch.sin(euler), torch.cos(euler)), dim=1)
    if mode == "euler_gb":
        channels = _reduced_zone_channels(euler, phi_encoding, phi_span_degrees)
        if boundary is None:
            return channels
        mask = boundary
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != euler.shape[0] or mask.shape[-2:] != euler.shape[-2:]:
            raise ValueError("Boundary mask must match the Euler batch and spatial size.")
        return torch.cat((channels, mask.to(dtype=channels.dtype, device=channels.device)), dim=1)
    raise ValueError("mode must be 'rgb', 'sincos', or 'euler_gb'")


def _unit_pair(sine: torch.Tensor, cosine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.sqrt(sine.square() + cosine.square()).clamp_min(1.0e-8)
    return sine / norm, cosine / norm


def _reduced_zone_channels(
    euler: torch.Tensor,
    phi_encoding: str,
    phi_span_degrees: float,
) -> torch.Tensor:
    phi1 = euler[:, 0:1]
    cap_phi = euler[:, 1:2]
    phi2 = euler[:, 2:3]
    sine_1, cosine_1 = torch.sin(phi1), torch.cos(phi1)
    sine_2, cosine_2 = torch.sin(4.0 * phi2), torch.cos(4.0 * phi2)
    if phi_encoding == "sincos":
        sine_phi, cosine_phi = torch.sin(cap_phi), torch.cos(cap_phi)
        return torch.cat((sine_1, cosine_1, sine_phi, cosine_phi, sine_2, cosine_2), dim=1)
    if phi_encoding == "linear":
        if phi_span_degrees <= 0:
            raise ValueError("phi_span_degrees must be positive.")
        linear = cap_phi / math.radians(phi_span_degrees) * 2.0 - 1.0
        return torch.cat((sine_1, cosine_1, linear, sine_2, cosine_2), dim=1)
    raise ValueError("phi_encoding must be 'sincos' or 'linear'")


def normalized_euler_gb(prediction: torch.Tensor, phi_encoding: str = "sincos") -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize predicted sin/cos pairs before an angle is recovered.

    The last channel is the raw grain-boundary logit or target and is not normalized.
    """
    if prediction.ndim != 4:
        raise ValueError("Expected BxCxHxW Euler+GB channels.")
    if phi_encoding == "sincos":
        orientation_count = 6
    elif phi_encoding == "linear":
        orientation_count = 5
    else:
        raise ValueError("phi_encoding must be 'sincos' or 'linear'")
    if prediction.shape[1] != orientation_count + 1:
        raise ValueError(
            f"Expected {orientation_count + 1} channels for phi_encoding={phi_encoding}."
        )
    pieces = []
    if phi_encoding == "sincos":
        pairs = ((0, 1), (2, 3), (4, 5))
    else:
        pairs = ((0, 1), (3, 4))
        pieces.append(prediction[:, 2:3])
    normalized = []
    for sine_index, cosine_index in pairs:
        sine, cosine = _unit_pair(prediction[:, sine_index:sine_index + 1], prediction[:, cosine_index:cosine_index + 1])
        normalized.append(sine)
        normalized.append(cosine)
    if phi_encoding == "linear":
        orientation = torch.cat((normalized[0], normalized[1], pieces[0], normalized[2], normalized[3]), dim=1)
    else:
        orientation = torch.cat(normalized, dim=1)
    return orientation, prediction[:, -1:]


def decode_euler_channels(
    channels: torch.Tensor,
    phi_encoding: str = "sincos",
    phi_span_degrees: float = 90.0,
) -> torch.Tensor:
    """Invert orientation channels to radians, shape Bx3xHxW.

    A boundary channel, when present, is ignored. phi2 uses a 90 degree period,
    so 90 degrees comes back as 0 degrees.
    """
    if channels.ndim != 4:
        raise ValueError("Expected BxCxHxW channels.")
    orientation_count = 6 if phi_encoding == "sincos" else 5 if phi_encoding == "linear" else None
    if orientation_count is None:
        raise ValueError("phi_encoding must be 'sincos' or 'linear'")
    if channels.shape[1] == orientation_count:
        padded = torch.cat((channels, torch.zeros_like(channels[:, :1])), dim=1)
    elif channels.shape[1] == orientation_count + 1:
        padded = channels
    else:
        raise ValueError(f"Expected {orientation_count} or {orientation_count + 1} channels.")
    orientation, _boundary = normalized_euler_gb(padded, phi_encoding)
    if phi_encoding == "sincos":
        phi1 = torch.atan2(orientation[:, 0], orientation[:, 1]) % (2.0 * math.pi)
        cap_phi = torch.atan2(orientation[:, 2], orientation[:, 3]).clamp(0.0, math.radians(phi_span_degrees))
        phi2 = (torch.atan2(orientation[:, 4], orientation[:, 5]) % (2.0 * math.pi)) / 4.0
    else:
        phi1 = torch.atan2(orientation[:, 0], orientation[:, 1]) % (2.0 * math.pi)
        cap_phi = (orientation[:, 2] + 1.0) * 0.5 * math.radians(phi_span_degrees)
        phi2 = (torch.atan2(orientation[:, 3], orientation[:, 4]) % (2.0 * math.pi)) / 4.0
    phi1 = torch.where(phi1 >= (2.0 * math.pi - 1.0e-8), torch.zeros_like(phi1), phi1)
    phi2 = torch.where(phi2 >= (math.pi / 2.0 - 1.0e-8), torch.zeros_like(phi2), phi2)
    return torch.stack((phi1, cap_phi, phi2), dim=1)
