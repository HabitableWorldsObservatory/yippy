"""Dense scene rendering with a validity mask for each source position."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp


class DenseSceneOperator(eqx.Module):
    """Native full or mirror-quadrant PSFs and their source admission mask."""

    psfs: jax.Array
    valid: jax.Array

    def __init__(self, psfs, valid):
        """Drop refused and nonfinite planes before any flux contraction."""
        psfs, valid = jnp.asarray(psfs), jnp.asarray(valid)
        if psfs.ndim != 4 or valid.shape != psfs.shape[:2] or valid.dtype != bool:
            raise ValueError("expected PSFs [sy,sx,y,x] and boolean validity [sy,sx]")
        self.valid = valid & jnp.all(jnp.isfinite(psfs), axis=(-1, -2))
        self.psfs = jnp.where(self.valid[..., None, None], psfs, 0.0)

    def render(self, flux, grid):
        """Return the native detector array and whether every lit source is valid."""
        if flux.shape != grid.shape:
            raise ValueError("source flux must match the native grid")
        if self.valid.shape == grid.shape:
            valid = jnp.all((flux == 0) | (jnp.isfinite(flux) & self.valid))
            image = jnp.einsum("ij,ijyx->yx", flux, self.psfs)
        else:
            ny, nx = grid.shape
            if not grid.is_geometric or self.valid.shape != (
                (ny + 1) // 2,
                (nx + 1) // 2,
            ):
                raise ValueError(
                    "quarter PSFs must match the positive quadrant of a geometric grid"
                )
            # Shared zero-offset axes belong to the positive halves on odd grids.
            sx, sy = math.ceil(grid.center_x), math.ceil(grid.center_y)
            padx, pady = nx % 2, ny % 2
            quadrants = jnp.stack(
                (
                    flux[sy:, sx:],
                    jnp.pad(flux[sy:, :sx][:, ::-1], ((0, 0), (padx, 0))),
                    jnp.pad(flux[:sy, :sx][::-1, ::-1], ((pady, 0), (padx, 0))),
                    jnp.pad(flux[:sy, sx:][::-1, :], ((pady, 0), (0, 0))),
                )
            )
            valid = jnp.all((quadrants == 0) | (jnp.isfinite(quadrants) & self.valid))
            parts = jnp.einsum("qij,ijyx->qyx", quadrants, self.psfs)
            image = (
                parts[0] + parts[1, :, ::-1] + parts[2, ::-1, ::-1] + parts[3, ::-1, :]
            )
        return jnp.where(valid, image, jnp.nan), valid
