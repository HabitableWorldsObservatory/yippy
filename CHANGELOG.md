# Changelog

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
