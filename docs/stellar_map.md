# The Stellar Intensity Map

A YIP stores the light of the occulted star in `stellar_intens.fits`: one map per
tabulated stellar diameter, on a square grid of cells (0.25 lambda0/D in the
public packages). This page states what each stored value is, how yippy puts
the map on a detector grid, and how accurate that read is.

## What a stored value is

The format paper fixes the normalization: the values "are equal to the stellar
count rate in a given pixel divided by the total stellar count rate entering the
coronagraph", so that without a coronagraph the map sums to one, and "all maps
must be at least Nyquist sampled" (Stark and Krist 2019, *Standardized
Coronagraph Parameters for Input into Yield Calculations*). It does not say how
the value of one pixel is computed.

The code that writes the public packages settles that. In the `cds_pipeline`
repository (`cds_pipeline/yield_data_package.py`, `make_stellar_diameter_maps`)
each map is the intensity of a Fraunhofer propagation evaluated at the cell
centers, multiplied by the cell area:

```text
img = compute_image_with_stellar_diameter(design, stellar_diameter, **sampling)
images.append(img['output_field'] * grid.weights)
```

A stored value is therefore a **point sample of the intensity at the cell
center, times the cell area**: a midpoint estimate of the cell integral, not the
integral itself. The files carry no `SAMPKIND` keyword, and yippy decodes an
absent one as `"point"`: `stellar_map(None, diameter_lod)` returns the samples
as a density per square grid unit.

The two definitions agree on totals and disagree per pixel. The intensity is
band-limited, so the sum of the samples times the cell area equals the integral
of the intensity. A single cell integral, however, is the sample smoothed by the
cell; on the public AAVC map the stored sample differs from it by about 2
percent at the median and 13 percent at the 95th percentile of the dark-hole
pixels.

### Band limit

An intensity formed through a pupil of diameter D has no spatial frequency above
D/lambda cycles per radian. The producer averages wavelengths from 0.95 to 1.05
lambda0 on one lambda0/D grid, so the cutoff of a stored map is 1/0.95 cycles
per lambda0/D, the cutoff of its bluest wavelength. The table samples at 4 per
lambda0/D, so its Nyquist frequency is 2 cycles per lambda0/D and the map is
sampled about twice as finely as it needs to be.

## How yippy reads the map

A detector pixel integrates the intensity over its footprint. `stellar_map` with
a `grid` therefore returns, for every pixel, the **integral of a reconstruction
of the samples over the pixel footprint**, at every pitch, including the native
one. There is no change of method when the pixels become as large as the cells.

The reconstruction is the interpolating cubic spline with the not-a-knot end
condition (`cubic_spline` in optixstuff), and the read is the exact integral of
that spline over each pixel. The choice follows from the map:

- A band-limited (sinc) reconstruction is exact for a table that holds the whole
  image, and it is what the off-axis PSFs use. The stellar map, however, ends
  inside a bright field about seven decades above the dark hole, and the sinc
  tails of that cut ring through the hole and below its floor.
- Linear interpolation is positive but smooths: its pixel integral at the table
  pitch is a 1/8, 3/4, 1/8 filter along each axis, which moves dark-hole pixels
  by several percent.
- The cubic spline reproduces cubic polynomials exactly and uses only the table
  at its edge. Its coefficients depend on every sample, with a weight that falls
  by a factor of about 0.268 per cell.

### Positivity floor

The spline has negative lobes. Each pixel is therefore the larger of the spline
read and one tenth of the linear footprint read of the same samples, which has
no negative weight. The result is positive wherever the table is, and, as the
maximum of two continuous reads, continuous in pointing, wavelength and stellar
diameter, with a kink where the floor starts to bind. On the public AAVC map the
floor binds nowhere between 1 and 24 lambda/D: only at the central null of the
largest tabulated stars (inside 0.3 lambda/D) and within four cells of the table
edge, at pitches finer than the table. `Coronagraph.stellar_map` applies the
floor; to read a native map from `StellarIntens` yourself, use
`yippy.sampling.stellar_on_grid`, because `optixstuff.coronagraph.on_grid` alone
does not floor.

### Support

A pixel is supported when its four corners lie two cells inside the table. Pass
the native density measure to `coverage` for this read.

This example needs a prepared coronagraph (`coro`) and a detector grid (`grid`).

```python notest
from optixstuff.grids import Measure, coverage

supported = coverage(
    coro.stellar_grid,
    grid,
    reconstruction="cubic_spline",
    measure=Measure.density("grid"),
)
```

On the native grid this leaves out the two outer rings. The two cells mark the
support boundary and are not an accuracy bound: the spline has no finite reach.

## Accuracy

- **Per pixel.** On a band-limited speckle field with power up to the band edge,
  the read errs by about 0.05 percent at the median and 0.3 percent at the 95th
  percentile in the dark hole at the table pitch, and by 0.03 to 0.07 percent at
  the median at pitches from 0.125 to 0.75 lambda/D. The read that returns the
  stored sample errs by 1.3 percent and 7 percent there.
- **In apertures.** On the public AAVC map, the sums over apertures of radius
  0.7 lambda/D on rings from 4 to 15 lambda/D, at pitches from 0.2275 to 0.275
  lambda/D, agree within 0.1 percent with two independent reads of the same
  samples: the order-5 spline footprint, and the band-limited footprint of the
  map tapered outside the rings (`tests/test_stellar_read.py`, tolerance 0.3
  percent). A linear read misses them by 5.8 to 7.4 percent and a Keys cubic
  by up to 0.6 percent.

## Limits

- There is no truth for a real map. Reconstructions of the public map disagree
  per pixel at the level above, and rebuilding the odd samples from the even
  ones (a table at half the sampling) errs by about 5 percent per pixel with the
  spline, 10 with the Keys cubic and 16 with linear.
- The stated accuracy holds for fields that are band-limited at the table pitch.
  At a hard edge the spline overshoots by up to 8 percent of the step.
- Within six cells of the table edge a pixel depends on the end condition of the
  spline: changing it moves a pixel by up to 13 percent 2 to 3 cells from the
  edge at a pitch of 0.125 lambda/D, and by under 0.1 percent beyond 8 cells.
  The edge of the public table is at 32 lambda/D, in the bright field outside
  the working angles.
- Because the spline has no finite reach, it is not for maps with zero-filled
  masks: zeroing the public map beyond a straight mask edge moves the spline
  read by up to 2.7 percent 2 to 3 cells from the edge, 0.8 percent at 3 to 4
  cells and 0.18 percent at 4 to 6 cells.
- The read is axis-aligned: a rotated or sheared detector grid raises
  `NotImplementedError`.
- The sky transmission map is not integrated: it is intensive, and it is sampled
  at the pixel centers. Against the pixel mean, the center sample differs by at
  most about 0.5 percent in an aperture of radius 0.7 lambda/D on the public
  packages, and by much more in single pixels on a steep occulter edge or a hard
  field stop.
- Each read of the star onto a grid costs two pairs of matrix products (the
  spline and the floor).
