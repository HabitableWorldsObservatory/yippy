# Changelog

## [4.0.0](https://github.com/HabitableWorldsObservatory/yippy/compare/v3.0.0...v4.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **sinc:** refuse float32 product metadata with PrecisionError
* **yippy:** close table, delivery, sinc and scene reads at the prepared model's precision
* **yippy:** rebase queries in float64 before narrowing, refuse float32 coefficient overflow, keep scene tangents at dark cells
* **coronagraph:** a package whose stellar_intens.fits MINLAM/MAXLAM disagree with offax_psf.fits is refused, and a package with MINLAM below MAXLAM is band-averaged and is placed once per observing band at its centre.
* **stellar:** stellar maps off the knots differ from earlier releases: the public AAVC map by up to 3.8e-6 relative (knots bitwise unchanged) and Optimal Order 6 by up to 0.4 percent at zero diameter. Performance caches written by earlier releases are recomputed. Subnormal positive stellar samples are refused at preparation, and export_exosims refuses nonpositive or nonfinite leakage.
* **jax:** drop the phasor arguments from the shift helpers
* **export:** write each exported file's own pixel pitch in the declared units
* **stellar:** keep zero samples and refuse lit flux outside the scene grid
* **sampling:** stellar_map on a detector grid returns new values at every pitch, including the native one: each pixel is the integral of the not-a-knot cubic spline of the stored samples over its footprint, floored at 0.1 of the linear footprint read. STELLAR_RECONSTRUCTION is cubic_spline (was linear), so the supported region shrinks by a second outer ring of the native grid (margin 2 cells, was 0.5). Requires the optixstuff release with the cubic_spline footprint read.
* **performance:** the performance record holds one leakage profile per tabulated stellar diameter, PERFVER is 8 and the cache key adds the diameter axis; evaluate of leakage and raw_contrast needs the stellar diameter.
* **sampling:** drop the outside argument of fraction_image; products are zero past their table
* **scene:** the scene operator's source grid extends to the off-axis node hull, so disk sources beyond the stamp are rendered
* **scene:** an operator returns a finite image and the cells it covers, validity is the union, and masked operators have a public constructor
* **read:** one source domain per backend, the node hull; a source outside it is flagged and zero, and the 1D read keeps its wing past the stamp
* **maps:** stop reflecting the stellar and sky maps past the table, so unmeasured pixels are zero
* **performance:** the default leakage_rule for a 2D product is now aperture_ring_mean (the aperture averaged over position angle) instead of aperture_mean (the +x axis only); pass leakage_rule='aperture_mean' for the old read. azimuthal_ring_mean is deprecated in favor of aperture_ring_mean.
* **scene:** sum a tuple of scene operators and refuse oversized operators on every route through hwoutils.memory
* **scene:** render scenes with BlendSceneOperator and refuse operators above max_bytes
* underscore the legacy modules, stop exporting logger, drop the legacy notebooks
* **export:** drop the Coronagraph export wrappers, spell out separation_*_lod, require ppf
* **datasets:** remove fetch_sinc_starter until the starter delivery is published
* **coronagraph:** follow the trimmed slot and stellar_diameter_lod naming
* **coronagraph:** drop the curve queries and finite-star leakage operators from Coronagraph
* **sinc:** reduce yippy.sinc to 15 names and fold the memory limit helpers into MemoryBudget
* **delivery:** move the sinc delivery format, routing and tools to yippy.delivery

### Features

* **coronagraph:** declare band-averaged packages from the off-axis spectral limits ([c64822c](https://github.com/HabitableWorldsObservatory/yippy/commit/c64822c97a69a3b45e435932e2a376773753bdfb))
* **coronagraph:** drop the curve queries and finite-star leakage operators from Coronagraph ([4897381](https://github.com/HabitableWorldsObservatory/yippy/commit/48973813f7aa2d8a7c9caa92bbfe363466b36360))
* **datasets:** remove fetch_sinc_starter until the starter delivery is published ([196f6fb](https://github.com/HabitableWorldsObservatory/yippy/commit/196f6fb0934d3e3ab7878d15989be685c6865049))
* **export:** drop the Coronagraph export wrappers, spell out separation_*_lod, require ppf ([4b17b78](https://github.com/HabitableWorldsObservatory/yippy/commit/4b17b7864156bc805c7a17127e934ec5d62c44fa))
* **performance:** attach the uniform-sky profile to every prepared performance record ([4aaa193](https://github.com/HabitableWorldsObservatory/yippy/commit/4aaa1932c05cb542f1434371d5d4138fd11b7d0a))
* **performance:** curves record carries leakage over the stellar diameter axis ([9504dee](https://github.com/HabitableWorldsObservatory/yippy/commit/9504dee9cfd73c939fea691f102d2cf97a9979a5))
* **performance:** read a 2D product's star leakage as the exact-coverage aperture averaged over azimuth by default ([25a1ab8](https://github.com/HabitableWorldsObservatory/yippy/commit/25a1ab8f98173698c4238cebffcfff3ce63fc5fd))
* **sampling:** drop the outside argument of fraction_image; products are zero past their table ([2e9abba](https://github.com/HabitableWorldsObservatory/yippy/commit/2e9abba4b12ef64d722dec9c3cae12310d52b96e))
* **sampling:** read the stellar map by its floored cubic spline footprint ([237616f](https://github.com/HabitableWorldsObservatory/yippy/commit/237616f47f239506d4b25641c1a0c950303b8b61))
* **scene:** an operator returns a finite image and the cells it covers, validity is the union, and masked operators have a public constructor ([f082a2b](https://github.com/HabitableWorldsObservatory/yippy/commit/f082a2bf76a7c41596c591b8b7f952b7363c3de4))
* **scene:** render scenes with BlendSceneOperator and refuse operators above max_bytes ([70ce265](https://github.com/HabitableWorldsObservatory/yippy/commit/70ce2654b1c0f7e5de61f32bc74939a46e6f6328))
* **scene:** sum a tuple of scene operators and refuse oversized operators on every route through hwoutils.memory ([0242317](https://github.com/HabitableWorldsObservatory/yippy/commit/02423173f439256897acd70f1eb54b194f714849))
* **scene:** the scene operator's source grid extends to the off-axis node hull, so disk sources beyond the stamp are rendered ([6ec4d26](https://github.com/HabitableWorldsObservatory/yippy/commit/6ec4d26c8e74d592470dfc8213a158186329a25f))
* underscore the legacy modules, stop exporting logger, drop the legacy notebooks ([def790b](https://github.com/HabitableWorldsObservatory/yippy/commit/def790ba6228778414c3b5aed73b8b7d3b258f40))


### Bug Fixes

* **delivery:** blend offsets and stamps at the query precision ([4ca41b8](https://github.com/HabitableWorldsObservatory/yippy/commit/4ca41b8bdc8e3d8bc14dc6220f843cb917476767))
* **export:** write each exported file's own pixel pitch in the declared units ([8ff3cf1](https://github.com/HabitableWorldsObservatory/yippy/commit/8ff3cf1086bc709e6c23e40f26ec4c55d5b93c75))
* **maps:** stop reflecting the stellar and sky maps past the table, so unmeasured pixels are zero ([6a423b2](https://github.com/HabitableWorldsObservatory/yippy/commit/6a423b2d6711860f0de7bad104a611f146870a5d))
* **offaxis:** zero each blend neighbor outside its translated stored window ([422d693](https://github.com/HabitableWorldsObservatory/yippy/commit/422d6934331c237db3705f840e0bb90ba8824e10))
* **precision:** read host offsets at the source dtype and enforce the precision guard ([92eb844](https://github.com/HabitableWorldsObservatory/yippy/commit/92eb8449f22823e1e7f05436b6c321f616437088))
* **read:** blend 2D nodes bilinearly in radius and azimuth so a throughput ramp is followed, not stepped ([c376d6f](https://github.com/HabitableWorldsObservatory/yippy/commit/c376d6fda2c3414c1e3ffbffeac4b556e87db4c1))
* **read:** keep the blend reader differentiable at nodes and the origin ([cc0cf33](https://github.com/HabitableWorldsObservatory/yippy/commit/cc0cf33d7aaf9006838714e2bd8ca0af4057c7bf))
* **read:** one source domain per backend, the node hull; a source outside it is flagged and zero, and the 1D read keeps its wing past the stamp ([1963ca9](https://github.com/HabitableWorldsObservatory/yippy/commit/1963ca9c107a5f653b94a8ffceeac380f44433dd))
* **sinc:** contract prefiltered read one axis at a time to avoid an XLA CPU miscompile ([7868d3b](https://github.com/HabitableWorldsObservatory/yippy/commit/7868d3b2f55060b18a0f00c1c12faf9535a8114d))
* **sinc:** refuse float32 product metadata with PrecisionError ([0b544b9](https://github.com/HabitableWorldsObservatory/yippy/commit/0b544b9ad0b52bf187450cbd9cd3d8c4eee3ccd9))
* **stellar:** bound the stellar interpolation to its bracketing samples and refuse dark exports ([3863875](https://github.com/HabitableWorldsObservatory/yippy/commit/38638752a4d531743c3cebf53c9f7744a12008ee))
* **stellar:** declare linear reconstruction for stellar maps so the dark hole cannot ring negative ([78eb38a](https://github.com/HabitableWorldsObservatory/yippy/commit/78eb38aa203cbd788e7c2ba3e5705940cbc83b7d))
* **stellar:** keep linear reconstruction for stellar maps, since log biases the dark-hole flux low ([96d84dc](https://github.com/HabitableWorldsObservatory/yippy/commit/96d84dc61c0412e72c518d791e8cc47c35f9a5fb))
* **stellar:** keep zero samples and refuse lit flux outside the scene grid ([a62af68](https://github.com/HabitableWorldsObservatory/yippy/commit/a62af68186105502ff456f2bc7c03a10a2fa1446))
* **stellar:** reconstruct stellar maps in log intensity so the dark hole stays positive ([0b99457](https://github.com/HabitableWorldsObservatory/yippy/commit/0b994576088cb46b4c31673fa1233135769563f3))
* **yippy:** close table, delivery, sinc and scene reads at the prepared model's precision ([eda94b8](https://github.com/HabitableWorldsObservatory/yippy/commit/eda94b8bc7bebee64d5a843b895f37f4e0086bf1))
* **yippy:** key cached performance records on the record algorithm version ([a082e19](https://github.com/HabitableWorldsObservatory/yippy/commit/a082e198f852732f732d723985aa011411775d60))
* **yippy:** rebase queries in float64 before narrowing, refuse float32 coefficient overflow, keep scene tangents at dark cells ([6dbab1c](https://github.com/HabitableWorldsObservatory/yippy/commit/6dbab1cc798d3a612952b2224e7521cb3de931c0))


### Code Refactoring

* **coronagraph:** follow the trimmed slot and stellar_diameter_lod naming ([12313bb](https://github.com/HabitableWorldsObservatory/yippy/commit/12313bbad7a792eb33fdf3c16a4a3f316716e8bb))
* **delivery:** move the sinc delivery format, routing and tools to yippy.delivery ([43513d6](https://github.com/HabitableWorldsObservatory/yippy/commit/43513d6b4c0ed344a5bc0b382dfc507a03239500))
* **jax:** drop the phasor arguments from the shift helpers ([68bdacf](https://github.com/HabitableWorldsObservatory/yippy/commit/68bdacfe311d4f03de51bd0f0604f214d36132a7))
* **sinc:** reduce yippy.sinc to 15 names and fold the memory limit helpers into MemoryBudget ([c197e62](https://github.com/HabitableWorldsObservatory/yippy/commit/c197e62936af538a111220face233fe9fa7c54df))

## [3.0.0](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.12.0...v3.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* declare point, integral and mean sampling for deliveries and YIP products
* implement imaging, bind scene operators with validity and require sinchole
* open YIPs as immutable packages with one writer, an archive descriptor and a synthetic test writer
* drop Matched from performance records and scalar exports
* record weighted-aperture variance moments and integrate hard-edged leakage over the planet region
* build off-axis PSF grids from the off-axis header and refuse packages whose grids differ
* one frozen Coronagraph implementing the optixstuff contract
* call sinchole admission and reader selection instead of copied rules
* delete Coronagraph query methods; the optixstuff contract is the query front end
* compute methods return data only, plotting moves to yippy.viz, and dead helpers and rule literals are gone
* off-axis coordinates come from the shared grid, and the deprecated Python synthesis path and transform copies are gone
* the circular aperture centre is a named field that defaults to the source position
* add aperture rules and a single performance-curve record with one evaluator
* products carry shared pixel grids built from the header, with the axis at the pixel-centre coordinate

### Features

* add aperture rules and a single performance-curve record with one evaluator ([e2355b2](https://github.com/HabitableWorldsObservatory/yippy/commit/e2355b24033bebaffd75301f7a4b87fa50d1b680))
* add Coronagraph.with_rules to prepare other aperture rules on a loaded coronagraph ([40f3a46](https://github.com/HabitableWorldsObservatory/yippy/commit/40f3a463b4ff6c923a31a5c93170dfa6f989937a))
* coronagraph PSFs and stellar maps declare band-limited reconstruction ([aa86c9e](https://github.com/HabitableWorldsObservatory/yippy/commit/aa86c9e426d66dab683eed3c6df9b3202a25fb75))
* declare point, integral and mean sampling for deliveries and YIP products ([57725da](https://github.com/HabitableWorldsObservatory/yippy/commit/57725da81a5f57f2554113ab0a4473332357d626))
* export DEFAULT_APERTURE_RULE, the 0.7 lambda/D circle used when no rule is named ([abe8b16](https://github.com/HabitableWorldsObservatory/yippy/commit/abe8b16b490b86c283b0c036c1af77474a88ae28))
* implement imaging, bind scene operators with validity and require sinchole ([9aaa7d7](https://github.com/HabitableWorldsObservatory/yippy/commit/9aaa7d77073890f0d4ed4c6874b9fd0681f75516))
* implement the prepared and stellar_frame coronagraph contract methods ([703c8e7](https://github.com/HabitableWorldsObservatory/yippy/commit/703c8e7e4301ab47cbdcc96b57bc5ad7af3ad266))
* one frozen Coronagraph implementing the optixstuff contract ([c27a456](https://github.com/HabitableWorldsObservatory/yippy/commit/c27a4561a5f027a5fbbe026b9dba0a4d4b211498))
* open YIPs as immutable packages with one writer, an archive descriptor and a synthetic test writer ([e85778a](https://github.com/HabitableWorldsObservatory/yippy/commit/e85778af2670f88ea572ac37bc2fb6eecaf90c78))
* **performance:** prepare linear leakage weights for finite-star queries ([57e701f](https://github.com/HabitableWorldsObservatory/yippy/commit/57e701f4ab2e4cd2053fe5978b55384218fb6b44))
* products carry shared pixel grids built from the header, with the axis at the pixel-centre coordinate ([119bbb0](https://github.com/HabitableWorldsObservatory/yippy/commit/119bbb0ce56a7c4e9ae7866da98899a2cf9c105d))
* stellar and sky maps declare that they continue past the tabulated field ([7abf3da](https://github.com/HabitableWorldsObservatory/yippy/commit/7abf3dac81a78d09fee7a1e7296dc26a0e5026cd))
* the circular aperture centre is a named field that defaults to the source position ([510b08d](https://github.com/HabitableWorldsObservatory/yippy/commit/510b08d646ce6c374b6f021cd9337d221bed3313))


### Bug Fixes

* build off-axis PSF grids from the off-axis header and refuse packages whose grids differ ([cbd1bf9](https://github.com/HabitableWorldsObservatory/yippy/commit/cbd1bf9e3ba27551419a59b4b92301d4edbb86be))
* compute the circle-square overlap angle with atan2 so a disk tiles to float64 rounding ([8526410](https://github.com/HabitableWorldsObservatory/yippy/commit/852641022fae83f4df1c03154794420f3ce08187))
* **coronagraph:** warn when a PSF datacube is built from a 1D YIP ([0fae8d6](https://github.com/HabitableWorldsObservatory/yippy/commit/0fae8d61818fe2675c2fb83a7aae32af8522eb14))
* record weighted-aperture variance moments and integrate hard-edged leakage over the planet region ([d02e8b9](https://github.com/HabitableWorldsObservatory/yippy/commit/d02e8b962d8328d9d2a9850f82966a1ab496fb8c))


### Code Refactoring

* call sinchole admission and reader selection instead of copied rules ([d4e78e6](https://github.com/HabitableWorldsObservatory/yippy/commit/d4e78e6c4602f240d52ec7e190006986ed01d665))
* compute methods return data only, plotting moves to yippy.viz, and dead helpers and rule literals are gone ([a2f6f30](https://github.com/HabitableWorldsObservatory/yippy/commit/a2f6f30b0ee96a78d67495a1380c86a7a4f1f5d3))
* delete Coronagraph query methods; the optixstuff contract is the query front end ([6dfec7a](https://github.com/HabitableWorldsObservatory/yippy/commit/6dfec7a207d68b3f366bc3d88074b1fde686c6c7))
* drop Matched from performance records and scalar exports ([0a0b7e7](https://github.com/HabitableWorldsObservatory/yippy/commit/0a0b7e70cfb96840e646379a42376f28aef75972))
* off-axis coordinates come from the shared grid, and the deprecated Python synthesis path and transform copies are gone ([5b980a7](https://github.com/HabitableWorldsObservatory/yippy/commit/5b980a7249be00735be061b28b6219992747667b))

## [2.12.0](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.11.0...v2.12.0) (2026-11-06)


### Features

* optional sinchole backend for versioned off-axis PSF deliveries ([471826b](https://github.com/HabitableWorldsObservatory/yippy/commit/471826b8d6065f8b4a8fc80e2e1c8f3fab44e688))
* **sinc:** add yippy-sinc validate and plan, starter dataset entry ([4d0063c](https://github.com/HabitableWorldsObservatory/yippy/commit/4d0063ce5a4922eb74999c4ee6fd8bb52bca068a))


### Bug Fixes

* **datasets:** raise a clear error while the starter sinc delivery is unpublished ([61ac7fd](https://github.com/HabitableWorldsObservatory/yippy/commit/61ac7fd97fa361f1553dd233bd73493b209c1dc0))
* exclude checks outside a narrower caller domain instead of rejecting them ([72af04b](https://github.com/HabitableWorldsObservatory/yippy/commit/72af04be9a29b111d59f90345c7728bca3257787))
* reject invalid evidence and conflicting premises before any blend fallback ([5d4f108](https://github.com/HabitableWorldsObservatory/yippy/commit/5d4f108a712d7ba491a14b35b75cbeb1ce0ce5ea))
* validate deliveries before routing and enforce sinc fallback rules ([9d4da84](https://github.com/HabitableWorldsObservatory/yippy/commit/9d4da842b9f0b55e76ee7a25fbb4a0c9459659ee))

## [2.11.0](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.6...v2.11.0) (2026-11-05)


### Features

* **performance:** interpolate performance curves between YIP offsets with PCHIP by default ([107f0cc](https://github.com/HabitableWorldsObservatory/yippy/commit/107f0cc45b0f40bf16b0e844b6aabe771b21a2e8))
* **performance:** measure the truncation core on a 10x subpixel grid ([0081ad5](https://github.com/HabitableWorldsObservatory/yippy/commit/0081ad505682aa5f10a5ca4fe54cf9dc3df79f91))
* **performance:** warn when a PSF is sampled coarser than Nyquist ([07f096c](https://github.com/HabitableWorldsObservatory/yippy/commit/07f096cc1f1a08c9a9debe37f1a0e4c077519826))
* **yippy:** add an opt-in quintic spatial circle reader ([75e3d47](https://github.com/HabitableWorldsObservatory/yippy/commit/75e3d47f92925328e6be8ec3d88df2dff5666eeb))


### Bug Fixes

* **api:** export performance functions and support fixed centers ([3ab075b](https://github.com/HabitableWorldsObservatory/yippy/commit/3ab075b5ce564327a0ce6e2c12bd2b7641f3fbbd))
* **coronagraph:** center separation_map on the optical axis ([be8f470](https://github.com/HabitableWorldsObservatory/yippy/commit/be8f4704b0f065b6e7e4682dd72ea4f354a7e405))
* **coronagraph:** write the PSF datacube cache atomically and rebuild a truncated cache instead of failing on every load ([7bba536](https://github.com/HabitableWorldsObservatory/yippy/commit/7bba536a1d859f2f1e66e588787692102a965ad1))
* **datasets:** find the YIP folder among the unpacked files even when the unzip folder holds clutter such as a Finder .DS_Store ([b675625](https://github.com/HabitableWorldsObservatory/yippy/commit/b6756250ab5c8455228a98f5739003483b5c2acb))
* **eqx_coronagraph:** sample two circles per diameter in core intensity ([33b032f](https://github.com/HabitableWorldsObservatory/yippy/commit/33b032f8f94ba91a76032d1ffa7ac09a5898d19e))
* **eqx:** mask stellar diameters outside tabulated bounds ([bc38ee9](https://github.com/HabitableWorldsObservatory/yippy/commit/bc38ee96915a707cdd495521a641e0bc2de66e5f))
* **eqx:** retain array separation shapes in intensity profiles ([65fdfbb](https://github.com/HabitableWorldsObservatory/yippy/commit/65fdfbbe36f998a1354fbb20b3a1e4e77b3c59a6))
* **performance:** cache the truncation core and keep it off the edge ([5f4dbfc](https://github.com/HabitableWorldsObservatory/yippy/commit/5f4dbfc6af29029d2e22557468ca4c0a02a3c602))
* **performance:** core mean intensity from the stellar map on circles ([a3f2a0d](https://github.com/HabitableWorldsObservatory/yippy/commit/a3f2a0d153d7af1745937dedabf30f20866a38c4))
* **performance:** give zero-flux offsets the finite contrast sentinel ([b11e5c5](https://github.com/HabitableWorldsObservatory/yippy/commit/b11e5c580bc57afa1f74c89365f7b728f0654b86))
* **performance:** integrate global cores with fractional weights ([45648ad](https://github.com/HabitableWorldsObservatory/yippy/commit/45648ade0443e88b253d1a397acd83ab17b66ec5))
* **performance:** measure circular-aperture throughput on the band-limited subpixel grid ([a49d89c](https://github.com/HabitableWorldsObservatory/yippy/commit/a49d89cce779e35f4f6cc7f65a96a56ebae1ddb6))
* **performance:** measure the core mean intensity profile about the optical axis ([9e676cb](https://github.com/HabitableWorldsObservatory/yippy/commit/9e676cbff92a12bf7351521c50c77c7602c45f2f))
* **performance:** preserve peak restriction at fractional edges ([27f0c65](https://github.com/HabitableWorldsObservatory/yippy/commit/27f0c650276703b7d1e363ef7f6ac9e523a705bf))
* **performance:** sample valid angular arcs at image corners ([77ce3f3](https://github.com/HabitableWorldsObservatory/yippy/commit/77ce3f3b20bb98a1d78236c020cee9dd8eabc0d2))
* **performance:** tabulate the core area every 0.25 lambda/D ([ac74867](https://github.com/HabitableWorldsObservatory/yippy/commit/ac74867c7cd61a74345d61282ef027ff9e1e9f3a))
* **performance:** tabulate the core curves at the YIP offsets by default ([3fc8081](https://github.com/HabitableWorldsObservatory/yippy/commit/3fc8081571b0ad66ae33f63f7ca0bc4db80a9a36))
* **performance:** take core throughput from the subpixel core ([be231de](https://github.com/HabitableWorldsObservatory/yippy/commit/be231dee9054c18e42b713ecc2eecf22623213b4))
* **performance:** weight aperture edges and constrain source peaks ([160b7f3](https://github.com/HabitableWorldsObservatory/yippy/commit/160b7f3810bf2efed6d554abc4a0c8d93838d305))
* **yippy:** declare tested dependency floors ([4856eda](https://github.com/HabitableWorldsObservatory/yippy/commit/4856eda1a7ccf0058cb7af68ddd8a67ade17cef8))
* **yippy:** preserve cache precision and repair strict CI checks ([2d5b35b](https://github.com/HabitableWorldsObservatory/yippy/commit/2d5b35b45959f21e3fd20c68946c4ecfc9551ac5))
* **yippy:** stabilize import sorting before version generation ([a95faf1](https://github.com/HabitableWorldsObservatory/yippy/commit/a95faf11c257959a57e817af43e8ee32413332ea))

## [2.10.6](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.5...v2.10.6) (2026-09-15)


### Bug Fixes

* **cache:** key performance and datacube caches by complete calculation identity ([704c28d](https://github.com/HabitableWorldsObservatory/yippy/commit/704c28d126a2f731444dbcc77d68acc7a604be94))

## [2.10.5](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.4...v2.10.5) (2026-09-15)


### Bug Fixes

* require hwoutils&gt;=1.10.0 for corrected rotation/shear math ([8b07b81](https://github.com/HabitableWorldsObservatory/yippy/commit/8b07b81f91003466c0f0700bc8200f3f06202b2d))

## [2.10.4](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.3...v2.10.4) (2026-09-14)


### Bug Fixes

* **transforms:** correct off-axis centering, rotation, and PSF rebinning via hwoutils ([9b0ab7b](https://github.com/HabitableWorldsObservatory/yippy/commit/9b0ab7b60217a6a94bd04ab3fd4ae42882df874e))

## [2.10.3](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.2...v2.10.3) (2026-08-31)


### Bug Fixes

* **cache:** invalidate performance/datacube caches on source YIP or downsample changes ([54f695e](https://github.com/HabitableWorldsObservatory/yippy/commit/54f695e5aed4bde232ce044c697f9002e09a3365))

## [2.10.2](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.1...v2.10.2) (2026-08-21)


### Bug Fixes

* Require hwoutils&gt;=1.6.1 for the radial_profile keyword rename ([96df49d](https://github.com/HabitableWorldsObservatory/yippy/commit/96df49de457c81f4250787043773c5205d821233))

## [2.10.1](https://github.com/HabitableWorldsObservatory/yippy/compare/v2.10.0...v2.10.1) (2026-07-30)


### Miscellaneous Chores

* release 2.10.1 ([1026e0d](https://github.com/HabitableWorldsObservatory/yippy/commit/1026e0d205b5d6e4c65da6ca50d2bc760dd0b8cd))

## [2.10.0](https://github.com/HWO-Project/yippy/compare/v2.9.1...v2.10.0) (2026-06-23)


### Features

* Add citation file ([e0a2893](https://github.com/HWO-Project/yippy/commit/e0a2893a750e5c0cb3b4343253e95856f904629b))

## [2.9.1](https://github.com/HWO-Project/yippy/compare/v2.9.0...v2.9.1) (2026-06-02)


### Bug Fixes

* Migrate to HWO-Project repo details ([485c99b](https://github.com/HWO-Project/yippy/commit/485c99b3ab8530da62d8bd8dec3ed3b7a245730b))

## [2.9.0](https://github.com/CoreySpohn/yippy/compare/v2.8.3...v2.9.0) (2026-05-30)


### Features

* **precision:** add float_dtype/dtype_tag helper following jax_enable_x64 ([2852d8a](https://github.com/CoreySpohn/yippy/commit/2852d8a35aa8d8c0c8d17ee0d0d246445a9ec746))
* **precision:** datacube builder follows jax_enable_x64 and dtype-keys its cache ([5845b5a](https://github.com/CoreySpohn/yippy/commit/5845b5a63371c8e00b91db8a47f53edab351e961))
* **precision:** EqxCoronagraph follows jax_enable_x64 via native canonicalization ([07cbbce](https://github.com/CoreySpohn/yippy/commit/07cbbcebaee8b3fc22f21fb214a7a6e33d91aad0))

## [2.8.3](https://github.com/CoreySpohn/yippy/compare/v2.8.2...v2.8.3) (2026-05-25)


### Bug Fixes

* Update to match API changes and naming standards ([678171d](https://github.com/CoreySpohn/yippy/commit/678171d4e1dfa7ee855ae418d2c83ded71ce7ba9))

## [2.8.2](https://github.com/CoreySpohn/yippy/compare/v2.8.1...v2.8.2) (2026-05-25)


### Bug Fixes

* Route psf datacube through numpy instead of jnp to avoid concatenate corruption in H100 ([dc11d62](https://github.com/CoreySpohn/yippy/commit/dc11d62ee0442bcf58da85674fac9758c5500447))

## [2.8.1](https://github.com/CoreySpohn/yippy/compare/v2.8.0...v2.8.1) (2026-05-19)


### Bug Fixes

* Fixing bug in the yip catalog table that messed up the readthedocs build ([5fa3651](https://github.com/CoreySpohn/yippy/commit/5fa36512c637bcaf4caaffc51cc0260b453017b4))

## [2.8.0](https://github.com/CoreySpohn/yippy/compare/v2.7.3...v2.8.0) (2026-05-19)


### Features

* Adding usort offaxis OVC data and removing hard coded structure ([eceefa3](https://github.com/CoreySpohn/yippy/commit/eceefa308193a218ead5aeebcf47e9b40ba1f32c))

## [2.7.3](https://github.com/CoreySpohn/yippy/compare/v2.7.2...v2.7.3) (2026-05-18)


### Bug Fixes

* Move YIP hosting from Zenodo to a data release ([d811a5e](https://github.com/CoreySpohn/yippy/commit/d811a5ef6381ed9c2bfdd0585c24cae1f8e7a23e))

## [2.7.2](https://github.com/CoreySpohn/yippy/compare/v2.7.1...v2.7.2) (2026-05-15)


### Bug Fixes

* Properly pass the psf truncation ratio to the performance metrics function on cache hit ([ab40225](https://github.com/CoreySpohn/yippy/commit/ab402259dcf288205d690edcd3763d90888827e7))

## [2.7.1](https://github.com/CoreySpohn/yippy/compare/v2.7.0...v2.7.1) (2026-05-15)


### Bug Fixes

* Resolve zenodo archive script error ([676eb18](https://github.com/CoreySpohn/yippy/commit/676eb184da7e0baeece5015655e83ab24ffe987d))

## [2.7.0](https://github.com/CoreySpohn/yippy/compare/v2.6.2...v2.7.0) (2026-05-15)


### Features

* Custom cache directory support ([c636b1f](https://github.com/CoreySpohn/yippy/commit/c636b1fae877b944aa6c67b42e80f8165234feb9))

## [2.6.2](https://github.com/CoreySpohn/yippy/compare/v2.6.1...v2.6.2) (2026-05-13)


### Bug Fixes

* Add zenodo info and update badge style ([09149ae](https://github.com/CoreySpohn/yippy/commit/09149aefa3e106eb34d1ad28e8f7123d9824bbc9))

## [2.6.1](https://github.com/CoreySpohn/yippy/compare/v2.6.0...v2.6.1) (2026-05-13)


### Miscellaneous Chores

* release 2.6.1 ([5b8c144](https://github.com/CoreySpohn/yippy/commit/5b8c14487a227f7fdb5d5034f4d40f2db0570901))

## [2.6.0](https://github.com/CoreySpohn/yippy/compare/v2.5.1...v2.6.0) (2026-05-13)


### Features

* Update the YIP loading documentation ([6607b3c](https://github.com/CoreySpohn/yippy/commit/6607b3c7a355846090feb446a77c44e19551811b))

## [2.5.1](https://github.com/CoreySpohn/yippy/compare/v2.5.0...v2.5.1) (2026-05-12)


### Bug Fixes

* **datasets:** add sampling filter to disambiguate same-family 1D/2D YIPs ([33dbaed](https://github.com/CoreySpohn/yippy/commit/33dbaed84cddf548ff19998d9a420cdf25ccc769))

## [2.5.0](https://github.com/CoreySpohn/yippy/compare/v2.4.0...v2.5.0) (2026-05-12)


### Features

* **datasets:** add 1d/2d sampling suffix; include 2D AAVC YIP ([f36f50d](https://github.com/CoreySpohn/yippy/commit/f36f50d4c3ba23bcd59c525e3dadcf32e1f37b98))
* **datasets:** add designer attribution field to CATALOG ([990e75d](https://github.com/CoreySpohn/yippy/commit/990e75d4d591b922fd68021d71713459367b037f))
* **datasets:** add Zenodo packaging script, populate catalog, docs ([1d4524d](https://github.com/CoreySpohn/yippy/commit/1d4524d26c62006d0fd6699f9cb835df8087c443))
* **datasets:** replace fetch_coronagraph with query-style YIP catalog ([f8aa2fc](https://github.com/CoreySpohn/yippy/commit/f8aa2fc73ff1a141bdbd2f72494d8535d5c79387))
* **datasets:** scaffold YIP catalog and remove fetch_coronagraph ([d822759](https://github.com/CoreySpohn/yippy/commit/d8227594c26a5c59c393812e93126e023bd9e524))
* **datasets:** wire up Zenodo DOI for v1 YIP archive ([d903c89](https://github.com/CoreySpohn/yippy/commit/d903c89e95b88ce12d7aabf7aebd8e5e65840cdb))
* Switch YIP hosting to Zenodo instead of hosting in the repo ([7e8a3e0](https://github.com/CoreySpohn/yippy/commit/7e8a3e0860c857017236d4414cdcac90991728da))

## [2.4.0](https://github.com/CoreySpohn/yippy/compare/v2.3.2...v2.4.0) (2026-04-29)


### Features

* Update interpolation method to calculate distance in r/theta coordinates instead of cartesian to better model the azimuthal variation of the PSFs ([98bb0f2](https://github.com/CoreySpohn/yippy/commit/98bb0f22cdd856973e151a5b921ccd260f5e1b04))


### Bug Fixes

* Update performance metric computation to better account for the edge of the image ([a90303f](https://github.com/CoreySpohn/yippy/commit/a90303f36a8f0472ebd6dba51ca395e573a8b88e))

## [2.3.2](https://github.com/CoreySpohn/yippy/compare/v2.3.1...v2.3.2) (2026-04-20)


### Bug Fixes

* Force float type for new jax version compatibility ([4c277df](https://github.com/CoreySpohn/yippy/commit/4c277dfa3b1b86dbafd2ade1798800be1ae1583d))
* Make the float types explicit for even more arrays to fix FITS file errors ([c7c2e61](https://github.com/CoreySpohn/yippy/commit/c7c2e61467f89d6d84ecf641a7e1a047700a7a59))

## [2.3.1](https://github.com/CoreySpohn/yippy/compare/v2.3.0...v2.3.1) (2026-03-30)


### Bug Fixes

* Add support for additional Coronagraph keywords within EqxCoronagraph ([4f1b998](https://github.com/CoreySpohn/yippy/commit/4f1b99879654b85c44afc617b187e280096cb22c))

## [2.3.0](https://github.com/CoreySpohn/yippy/compare/v2.2.2...v2.3.0) (2026-03-30)


### Features

* Add GPU PSF datacube function ([a35a224](https://github.com/CoreySpohn/yippy/commit/a35a224c83ebff24aa536ec3ab1151acd7023ba0))

## [2.2.2](https://github.com/CoreySpohn/yippy/compare/v2.2.1...v2.2.2) (2026-03-04)


### Bug Fixes

* Fully switch to using lod_unit.lod explicitly instead of relying on the astropy.units registry entry for lod, offload more to hwoutils ([951133e](https://github.com/CoreySpohn/yippy/commit/951133e233196d8c10659ab6731b89ce73aba71f))

## [2.2.1](https://github.com/CoreySpohn/yippy/compare/v2.2.0...v2.2.1) (2026-02-28)


### Bug Fixes

* Removing incorrect function in fft methods ([8c7909e](https://github.com/CoreySpohn/yippy/commit/8c7909e22c93c90defff1f41f095b413bb1c717f))

## [2.2.0](https://github.com/CoreySpohn/yippy/compare/v2.1.0...v2.2.0) (2026-02-27)


### Features

* Improve caching system ([080fc27](https://github.com/CoreySpohn/yippy/commit/080fc271eaf412021cc5c108d26d11c0b2f89f9c))

## [2.1.0](https://github.com/CoreySpohn/yippy/compare/v2.0.1...v2.1.0) (2026-02-26)


### Features

* Add testing, fix lint issue ([a530990](https://github.com/CoreySpohn/yippy/commit/a5309908c2a03d6d0a78ec2cf6a174556d3ceb39))


### Bug Fixes

* Set minimum jax version ([89a5b38](https://github.com/CoreySpohn/yippy/commit/89a5b3848fbf90e3549a362a8e15bd807468ea20))

## [2.0.1](https://github.com/CoreySpohn/yippy/compare/v2.0.0...v2.0.1) (2026-02-25)


### Bug Fixes

* Deprecating non-functional jax configuration options ([47bceec](https://github.com/CoreySpohn/yippy/commit/47bceec272a8d4197f4d26a5fc132bb5355e740b))

## [2.0.0](https://github.com/CoreySpohn/yippy/compare/v1.12.2...v2.0.0) (2026-02-25)


### ⚠ BREAKING CHANGES

* AYO validated performance metrics and 2D maps for pyEDITH refactor

### Features

* AYO validated performance curves and 2D map methods ([52a6766](https://github.com/CoreySpohn/yippy/commit/52a6766c59d9c1c83c9e699c3dbf3a8e1fa3bd0a))
* AYO validated performance metrics and 2D maps for pyEDITH refactor ([f204461](https://github.com/CoreySpohn/yippy/commit/f204461fbb63530ac524ab2212d63c22348804f1))
* Full equinox coronagraph ([3634fff](https://github.com/CoreySpohn/yippy/commit/3634fff61dcc929fd88fdb931323f98a02235abe))
* Performance metrics refactor ([5d8c194](https://github.com/CoreySpohn/yippy/commit/5d8c194436703868ac38b99ae927eadc83a37111))

## [1.12.2](https://github.com/CoreySpohn/yippy/compare/v1.12.1...v1.12.2) (2025-12-10)


### Bug Fixes

* Improve PSF datacube device handling for JAX compatibility ([b94ad19](https://github.com/CoreySpohn/yippy/commit/b94ad19d5b640e9ca1011da738d1ac7326ce4e8a))

## [1.12.1](https://github.com/CoreySpohn/yippy/compare/v1.12.0...v1.12.1) (2025-12-10)


### Bug Fixes

* Remove all references to old sparse matrix reshaped_psfs ([3db774a](https://github.com/CoreySpohn/yippy/commit/3db774a477aa5d22e41dabbbafae7d670689c9ef))

## [1.12.0](https://github.com/CoreySpohn/yippy/compare/v1.11.2...v1.12.0) (2025-12-09)


### Features

* Add support for Inverse Distance Weighting (IDW) PSF synthesis and enable quarter PSF datacube computation for OffJAX ([fe0a554](https://github.com/CoreySpohn/yippy/commit/fe0a5548e1441ed648c3b7ddd22e20408102b0ed))
* GPU support for PSF datacube ([67d5c64](https://github.com/CoreySpohn/yippy/commit/67d5c645d643843766f43577e56f01ddce0bfc46))


### Bug Fixes

* Ensure stellar diameters are a flat array ([6842932](https://github.com/CoreySpohn/yippy/commit/6842932b1c2aa05aab8430fea8c75dcb529c4e60))

## [1.11.2](https://github.com/CoreySpohn/yippy/compare/v1.11.1...v1.11.2) (2025-11-20)


### Bug Fixes

* Revert incorrect merge ([7272fe1](https://github.com/CoreySpohn/yippy/commit/7272fe1fe0607e1f98bc627ae561ed5a6e3d91a0))

## [1.11.1](https://github.com/CoreySpohn/yippy/compare/v1.11.0...v1.11.1) (2025-11-20)


### Miscellaneous Chores

* release 1.11.1 ([36fca7e](https://github.com/CoreySpohn/yippy/commit/36fca7e19fec936946d439642233363ba788a81d))

## [1.11.0](https://github.com/CoreySpohn/yippy/compare/v1.10.2...v1.11.0) (2025-11-20)


### Features

* Add ability to dump coronagraph performance files that match the EXOSIMS format ([edf276c](https://github.com/CoreySpohn/yippy/commit/edf276cabd858b9b82686f7a5585793273763177))
* Performance metrics ([91d86b5](https://github.com/CoreySpohn/yippy/commit/91d86b5b04fd7b71fae7785f4b7ac6e954be2078))

## [1.10.2](https://github.com/CoreySpohn/yippy/compare/v1.10.1...v1.10.2) (2025-07-21)


### Bug Fixes

* testing new shard_map fix ([30e851d](https://github.com/CoreySpohn/yippy/commit/30e851d13edfe926f2edd51c926e2b3d5b7ce28e))

## [1.10.1](https://github.com/CoreySpohn/yippy/compare/v1.10.0...v1.10.1) (2025-07-21)


### Bug Fixes

* testing shard_map fix ([6d71df8](https://github.com/CoreySpohn/yippy/commit/6d71df89d3c25fc06e5d660f17ae078d02e7b525))

## [1.10.0](https://github.com/CoreySpohn/yippy/compare/v1.9.2...v1.10.0) (2025-06-02)


### Features

* Add information on the maximum separation in the image for 1d case ([f16b56e](https://github.com/CoreySpohn/yippy/commit/f16b56e1d82a669ecb7f8afb84fcbc72f59cce56))

## [1.9.2](https://github.com/CoreySpohn/yippy/compare/v1.9.1...v1.9.2) (2025-04-01)


### Bug Fixes

* Add version info to the coronagraph performance file for future proofing ([c39b40b](https://github.com/CoreySpohn/yippy/commit/c39b40bbebdded4c165f1f879b85c014518ca14c))

## [1.9.1](https://github.com/CoreySpohn/yippy/compare/v1.9.0...v1.9.1) (2025-04-01)


### Bug Fixes

* Exclude 0 values from the IWA indexing ([a45177c](https://github.com/CoreySpohn/yippy/commit/a45177c201307fdb5d90491785dd04c7bd14eda8))
* Exclude 0 values from the IWA indexing ([4020d98](https://github.com/CoreySpohn/yippy/commit/4020d98e95b4756b1251e2ea016f16f87d3f8860))

## [1.9.0](https://github.com/CoreySpohn/yippy/compare/v1.8.1...v1.9.0) (2025-01-16)


### Features

* Add calculation of IWA ([461e392](https://github.com/CoreySpohn/yippy/commit/461e3925540f886a1f414bdf16a446a63160b449))


### Bug Fixes

* Now passes a list of integers into the JAX create_device_mesh function to keep up with a change ([59a39f5](https://github.com/CoreySpohn/yippy/commit/59a39f5fd7737f4012711d383877644a2c8b9cf7))

## [1.8.1](https://github.com/CoreySpohn/yippy/compare/v1.8.0...v1.8.1) (2025-01-07)


### Bug Fixes

* Default to 0 values instead of nan values to avoid erroring out ([d126379](https://github.com/CoreySpohn/yippy/commit/d126379fc34bd291d81de5f7f87c7b5465338d11))

## [1.8.0](https://github.com/CoreySpohn/yippy/compare/v1.7.2...v1.8.0) (2025-01-06)


### Features

* Add calculation of raw contrast and throughput ([a561efe](https://github.com/CoreySpohn/yippy/commit/a561efef0591aa975aeafcfa355f474f81e116c1))

## [1.7.2](https://github.com/CoreySpohn/yippy/compare/v1.7.1...v1.7.2) (2025-01-02)


### Bug Fixes

* Now passes lod values instead of lod quantities into the create_psf function ([64c9d2f](https://github.com/CoreySpohn/yippy/commit/64c9d2fbcfb335dbea7b88fc65fd28935748398c))

## [1.7.1](https://github.com/CoreySpohn/yippy/compare/v1.7.0...v1.7.1) (2024-12-16)


### Bug Fixes

* Mask out pixels with no information ([b02b59b](https://github.com/CoreySpohn/yippy/commit/b02b59b86006b8aaac0cd3455ddc290b287d5de1))

## [1.7.0](https://github.com/CoreySpohn/yippy/compare/v1.6.0...v1.7.0) (2024-12-14)


### Features

* Added a function to generate the psf datacube ([9d69197](https://github.com/CoreySpohn/yippy/commit/9d69197a1524df1514848494c95fd1673e7fb890))


### Bug Fixes

* Cut negative values from the fft_shift functions ([2ccaf77](https://github.com/CoreySpohn/yippy/commit/2ccaf775a81f68d77040c386b0af13f3fbdd7ac0))

## [1.6.0](https://github.com/CoreySpohn/yippy/compare/v1.5.0...v1.6.0) (2024-12-12)


### Features

* Using shard_map for parallel processing with JAX ([ca2ba07](https://github.com/CoreySpohn/yippy/commit/ca2ba07b47431ded18b391ab2a1b13f1aa85f515))


### Bug Fixes

* Add safe reciprocal calculation instead of potential division by zeros ([eeadd32](https://github.com/CoreySpohn/yippy/commit/eeadd322367d2e65ed561ce5f396f86f0a27af6a))

## [1.5.0](https://github.com/CoreySpohn/yippy/compare/v1.4.0...v1.5.0) (2024-12-02)


### Features

* Add OffJAx class ([46c82f9](https://github.com/CoreySpohn/yippy/commit/46c82f92e0957924dc3e8be70dba199d3917b8ea))
* Make x and y symmetry optional, remove rotational symmetry ([fe1cb33](https://github.com/CoreySpohn/yippy/commit/fe1cb3346feff5aedd43a649bfde273672b150c7))


### Bug Fixes

* Added x/y symmetry options to the JAX implementation ([08d08e6](https://github.com/CoreySpohn/yippy/commit/08d08e62e70d1533c3b0644560a5ae7569cd570f))

## [1.4.0](https://github.com/CoreySpohn/yippy/compare/v1.3.0...v1.4.0) (2024-08-30)


### Features

* Add expressive logger ([4eec73c](https://github.com/CoreySpohn/yippy/commit/4eec73c74168b1afd8246919ca05d43cf9e6bb7f))
* Add Fourier interpolation utility functions ([d023e1c](https://github.com/CoreySpohn/yippy/commit/d023e1c650d674829c4117738b05f6816ff2762f))
* Implement FFT based interpolation and rotation ([8ecac66](https://github.com/CoreySpohn/yippy/commit/8ecac660316e9f003f27d997874fc0ebcd5202e9))
* Implemented fft interpolation in the One-D case ([5309cb6](https://github.com/CoreySpohn/yippy/commit/5309cb64ef29819831aad53723b344a3132c3ebc))


### Bug Fixes

* Fix the import of the logger ([4021431](https://github.com/CoreySpohn/yippy/commit/4021431b9a131fd703b3fc154c40f394c145dbf7))
* **main:** Improve the one D PSF to only take the log if necessary ([845fdf5](https://github.com/CoreySpohn/yippy/commit/845fdf53d759894ae17b40e1fff1b689acbb49a8))

## [1.3.0](https://github.com/CoreySpohn/yippy/compare/v1.2.0...v1.3.0) (2024-04-23)


### Features

* **main:** Add a temporary sky_trans file ([ad89135](https://github.com/CoreySpohn/yippy/commit/ad89135fc2687b60af018e7a9fde503513ee1854))
* **main:** Added dataclass that handles the header ([3120eda](https://github.com/CoreySpohn/yippy/commit/3120eda53bb75dc96ead74ae3e37c5cd206785ac))

## [1.2.0](https://github.com/CoreySpohn/yippy/compare/v1.1.1...v1.2.0) (2024-04-17)


### Features

* **main:** Added stellar intensity map ([481d333](https://github.com/CoreySpohn/yippy/commit/481d333b89280a906bf8be3642f0eb7bf1fa946e))
* **main:** Adding more support for 2d and quarter symmetric coronagraphs ([3e98780](https://github.com/CoreySpohn/yippy/commit/3e9878034b37535780ee0004f69ad4409b961445))


### Bug Fixes

* **main:** Fixed error in how the quarter symmetric PSFs handled 0*lam/D values ([3e6943f](https://github.com/CoreySpohn/yippy/commit/3e6943f6bfaf89c8b8ba353921bc5a245696e194))

## [1.1.1](https://github.com/CoreySpohn/yippy/compare/v1.1.0...v1.1.1) (2024-04-05)


### Bug Fixes

* **main:** Fixed handling when given single dimensional offax_psf_offsets_list without a second column of zeros ([86f0cc7](https://github.com/CoreySpohn/yippy/commit/86f0cc795d6471b8abaddc3e80278d97aaf93706))

## [1.1.0](https://github.com/CoreySpohn/yippy/compare/v1.0.0...v1.1.0) (2024-04-05)


### Features

* **main:** Add off-axis psfs with automatic unit conversion ([6f5b815](https://github.com/CoreySpohn/yippy/commit/6f5b815093e6fe7898cd625451ad31ab1acee221))

## 1.0.0 (2024-03-22)


### Features

* Automatic versioning and changelog ([ef1acc1](https://github.com/CoreySpohn/yippy/commit/ef1acc1381058fdb32f6b32bb3d695a2035ad048))


### Bug Fixes

* Adding pre-commit hook for conventional commit formatting ([3b52ed6](https://github.com/CoreySpohn/yippy/commit/3b52ed6e3233b7acaa51f5ee8cd2a2b3f317912f))
* putting the workflows in the right folder ought to help ([ff1bf0a](https://github.com/CoreySpohn/yippy/commit/ff1bf0a12850691de801c9a3ba4202f3e8f4f7f1))
