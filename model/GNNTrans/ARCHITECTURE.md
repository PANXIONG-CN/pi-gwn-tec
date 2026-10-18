# Single paper-based pyramid adaptation — explicit reconstruction

This version is independently implemented from Liu and Jiang (DOI10.1109/JSTARS.2024.3508794), not official software or uniquely recovered author code. No alternative architecture is trained or exposed as a switch.

## Source-to-code mapping

| Paper | Code | Frozen implementation interpretation |
|---|---|---|
| III-C grid and spherical positions | GIMGridAdapter, coordinates | Remove +180 duplicate; add north-pole row averaged from87.5N; north-to-south. Pool unit-sphere Cartesian coordinates and renormalize for each downsample scale. Restore 71x73 before both training loss and metrics; synthetic row never a target. |
| III-C position kNN vs III-E feature graph ambiguity | fixed_neighbors | Fixed spherical position kNN, k9 including colocated candidate if present; Euclidean chord distance on unit sphere (same ordering as great-circle distance). Round squared distances to1e-12 then stable ascending node-index tie break, including repeated pole nodes. Follows III-C, explicitly does not recover/claim dynamic feature graph of III-E. |
| III-E early spatial reduction | GraphBlock | Candidate feature/coordinate pooling factors4,2,1 for72,36,18 grids; query grid unchanged. Distinct from Conv2d feature downsampling. This factor choice is not specified by the article. |
| Eq7 max-relative graph update | GraphBlock | Linear preprojection; concat(center,max(neighbor-center)); linear update; GELU. Pool preprojected candidate features. No graph normalization layers. Biases enabled as a declared choice (paper equations omit them). |
| Eq8 malformed publisher TeX; three residual connections | GraphBlock | g=GELU(MRConv(Win*x)); u=g+x; v=GELU(W2*GELU(W1*u)); y=x+DropPath(GELU(v+u)). Residual1 x→u; residual2 u→v+u; residual3 x→y. Hidden4C. This is an explicit compatible reconstruction, NOT a verified literal Eq8 transcription. |
| III-D condition Q=graph,K/V=NSF | ConditionTokens, stem_cross | Four feature tokens per frame for standardized AE/Dst/Kp/F107, each scalar encoded to8 dimensions with learned feature identifier. Every historical GIM independently attends its own4 condition tokens (2 heads); no single-key attention degeneration. |
| Common spatial cosSZA adaptation | input | cosSZA enters each node with TEC via2→8 pointwise projection. It is not averaged globally, nor called a nonspatial activity index. All original extra calendar/SN/ap inputs removed under common-information constraint. Learned relative frame positions do not add observed future information. |
| Eq2/3 spatial positional encoding | spatial_encoding | Added after input projection,8channels. Explicit radians, channel0lat/channel1lon; key=(lon*72+sin(lat))*72; sin/cos periods10000^(2k/8). Original units/embedding placement unspecified, declared here. |
| TableI pyramid stem | stem/down | 72x72x8 graph→conv2x2stride2→36x36x16 graph→conv→18x18x32 graph→conv→9x9x64. All divisible: no zero padding. |
| TableI encoder shape | encoder | [B,12,64,9,9]→[B,81,768], spatial tokens with concatenated time×channel features;3 encoder-only blocks. 8 selfattention heads; dmodel768, FFN hidden512 (the table's512 assigned this role as an interpretation). No per-node temporal attention substitution. |
| First-encoder condition crossattention | EncoderBlock(first) | Q=[B,81,768]; K/V=[B,48,768], one token per12times×4activity channels with learned frame+feature IDs. Added before selfattention in first block only. Pre-LayerNorm declared. |
| Learned Transformer position | position | Parameter[1,81,768], separate from Eq3. |
| III-G inverse pyramid graph head and stem skips | up/head | Reshape back to[B*12,64,9,9]; nearest-neighbor upsample2 then1x1projection64→32, add matching18x18stem skip, graph; repeat32→16 and16→8. Output1x1conv8→1. No padding; periodic graph handles across-seam neighbors. Interpolation kernel/skip order are declared reconstruction choices. |

DropPath0.1 in graph residual branch and dropout0.1 in attentions/Transformer FFN are explicit. Parameters use PyTorch initializers except learned position/feature/time embeddings N(0,0.02²). Latitude/longitude coordinates are static method information; global activity and spatial cosSZA remain history-only. A 1e-5 absolute standardized cosSZA seam tolerance addresses trigonometric roundoff; TEC duplicate endpoints require exact equality. No TEC fill, clipping, rescaling by evaluation data, or averaging of target endpoints.

The stem processes frames independently before the multiframe Transformer. Future outputs share the learned direct12-frame output interpretation; stem-head residuals carry each corresponding historical frame as an explicit index-pairing convention of this adaptation, not a uniquely recovered paper detail and not future TEC. Learned mixing across time is inside each token's768features and its attention projections/FFN.

## Training recipe used for the retained comparison

Use paper AdamW, LR2e-4, WD0.1, batch16, warmup20, cosine decay through epoch970, cooldown30 at minimumLR1e-6, maximum1000epochs, EMA0.99996 per optimizer step. Early stopping30epochs on pooled validation RMSE of EMA weights (no early stop before warmup ends); choose first strict minimum; train/val only. EMA convention and patience/minLR are declared adaptations. Train with existing masked MAE in TECU, matching the study objective; original paper MSE is a disclosed difference. Five seeds share exactly this recipe. No extra trial. AMP disabled for numerical simplicity; batch16 was the retained training batch size. Observed timing is supplied in Source Data.

The GWN comparisons use Adam,LR0.003,WD0,batch192,max200,plateau,patience30. Differences in optimization/budget, geometry and conditioning prevent attributing all performance differences to architecture alone. Existing original five-seed predictions are reused after alignment; no comparison model retraining.
