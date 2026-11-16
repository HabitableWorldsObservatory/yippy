"""Static photometric aperture rules on per-pixel source-flux Images.

Source positions are (x, y) in the Image's lambda/D frame. ``weights`` returns
fractional coverage on that Image's grid. Planet throughput uses the same
subpixel weights with band-limited PSF samples. Stellar leakage is the absolute
sum of the native stellar per-pixel fractions times the native weights.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Circular:
    """Circle of radius ``radius_lod`` about the source's refined local peak."""

    radius_lod: float

    def __post_init__(self):
        """Validate the static rule parameter."""
        if not math.isfinite(self.radius_lod) or self.radius_lod <= 0:
            raise ValueError("radius_lod must be finite and positive")

    def weights(self, psf, source_position, *, oversample=10):
        """Fractional native-pixel coverage, using exact circle-square overlaps."""
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


@dataclass(frozen=True)
class HalfMaximum:
    """Peak-connected region above ``fraction`` of the refined local peak."""

    fraction: float = 0.5

    def __post_init__(self):
        """Validate the static rule parameter."""
        if not math.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError("fraction must be between zero and one")

    def weights(self, psf, source_position, *, oversample=10):
        """Fractional native-pixel coverage from the band-limited level set."""
        from .performance import _rule_weights

        return _rule_weights(self, psf, source_position, oversample)


@dataclass(frozen=True)
class Matched:
    """PSF weighting, distinct from a geometric aperture.

    w=p/max(p), tau=sum(w*p), A=sum(w)**2/sum(w**2)*pixel_area.
    A is the noise-equivalent area, not the integral of the weights.
    A zero template has zero weights, throughput and effective area.
    Reference: Kasdin & Braems (2006), ApJ 646, 1260,
    https://doi.org/10.1086/505017 (PSF matched-filter photometry).
    This peak normalization is yippy's explicit weighting convention.
    """

    def weights(self, psf, source_position, *, oversample=10):
        """Peak-normalized template on the supplied grid (no mask or recentering)."""
        from .performance import _validate_photometry_image

        _validate_photometry_image(psf)
        data = jnp.asarray(psf.data)
        peak = jnp.max(data)
        return jnp.where(peak > 0, data / jnp.where(peak > 0, peak, 1), 0)


ApertureRule = Circular | HalfMaximum | Matched
