# Contrast Floors

A YIP records the contrast of an idealized design. Real hardware cannot
hold that contrast: wavefront stability, polarization, and coating
errors set a practical limit. `Coronagraph` accepts a `contrast_floor`
so you can cap a design at the contrast the instrument is actually
expected to reach.

```python
from yippy import Coronagraph, fetch_yip

# "This coronagraph cannot do better than 4e-10."
coro = Coronagraph(fetch_yip("eac1_aavc_2d"), contrast_floor=4e-10)
```

`contrast_floor` is a lower bound on raw contrast, in raw-contrast
units. Smaller contrast means better starlight suppression, so the
floor is applied as `max(raw_contrast, contrast_floor)`: any separation
where the design beats the floor is reported at the floor instead.

The same keyword exists on `EqxCoronagraph` and behaves identically.

## What the floor applies to

The floor is applied in two places, so it holds whether you read the
stored performance curve or evaluate the interpolant:

1. **At load time.** The raw-contrast curve is floored before the
   log-space spline is fit, so the contrast interpolant itself never
   dips below the floor.
2. **At call time.** `raw_contrast(separation)` clamps the interpolated
   value again, which keeps spline undershoot between knots from
   sneaking under the floor.

Everything that reads raw contrast therefore inherits the floor. For
the shipped `eac1_aavc_2d` YIP, whose design contrast runs from roughly
2e-11 to 2e-10 across the dark zone, a 4e-10 floor dominates at every
separation:

```python
coro.raw_contrast([3.0, 5.0, 10.0, 20.0])
# array([4.e-10, 4.e-10, 4.e-10, 4.e-10])
```

`noise_floor_exosims` is built on `raw_contrast`, so it picks the floor
up as well. Note that it also takes its own `contrast_floor` argument
(default `1e-10`), applied on top of whatever the coronagraph carries.
The effective floor is the larger of the two. Pass `contrast_floor=0`
to that method if you want only the coronagraph-level limit.

## What the floor does not apply to

The floor is a statement about contrast, and only quantities in
contrast units are clamped. It does not touch:

- `core_mean_intensity`, and therefore `noise_floor_ayo`, which work in
  per-pixel intensity units rather than per-aperture contrast.
- `core_throughput`, `core_area`, and `occulter_transmission`.
- The off-axis PSFs, stellar intensity maps, and sky transmission maps.

If you are working in the AYO or pyEDITH per-pixel convention, a
contrast-unit floor is not the right knob, and the two noise-floor
conventions will disagree once a floor is set. See
{doc}`/examples/06_Noise_Floor_Conventions` for how the conventions
relate.

## Inspecting the setting

`contrast_floor` is stored on the coronagraph and is `None` when no
floor was requested:

```python
>>> coro.contrast_floor
4e-10
```

It is a construction-time argument. Setting it after the fact changes
`raw_contrast` evaluation but leaves the already-fit interpolant
unfloored, so build a new `Coronagraph` when you want to compare
floors.
