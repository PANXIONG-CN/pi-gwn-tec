# Channel Mixer attribution and modifications

Derived from the authors’ Channel Mixer Layer v1.0 / OpenSTL source:
https://github.com/DrHoisu/Channel_Mixer_Layer/tree/v1.0
Tree: ca4ccd29ae334e9baa438aa4734ae463bf98f7f3.
Distributed under the accompanying Apache License 2.0. Original source attribution,
comments and license are retained. This adaptation does not imply author endorsement.

`lib/external_mixer_model.py` extracts the selected gSTA definitions from
`openstl/models/simvp_model.py`, `openstl/modules/simvp_modules.py`, and
`openstl/modules/layers/van.py`. It fixes the released mix4 configuration,
omits unused architectures, replaces timm initialization with PyTorch initialization
(overwritten by a complete strict checkpoint load), and implements local DropPath
with identical evaluation behavior.

`lib/external_mixer.py` implements the history-only first-block permutation and
GIM grid/unit bridge based on the author loader and SimVP wrapper. It accepts only
12 historical frames, reconstructs per-frame UTC-year channels, and restores the
longitude seam. It does not use the original 48-step autoregressive wrapper or
future auxiliaries. No checkpoint, raw data or external framework is distributed.
