from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from lib.physics.solar import canonical_cos_sza_float64


class DriversStore:
    """Load drivers prepared by tools/prepare_drivers_omni.py and slice by global time index.

    This is optional; for zero-driver PPINN, you do not need this.
    """

    def __init__(self, npz_path: str, device='cpu', *, strict: bool = False, expected_sha256: str | None = None):
        self.device = device
        self.path = str(npz_path)
        self.strict = bool(strict)
        self.source_sha256 = None
        self.driver_clip_or_pad_count = 0
        self.driver_fallback_count = 0
        self.synthetic_driver_count = 0
        self.grid_shape = (71, 73)
        self.synthetic = False
        # Grid lon/lat for cosSZA on-the-fly. Prefer npz-provided lon/lat (supports non-71x73 grids).
        self._lon_deg = None
        self._lat_deg = None
        try:
            if not os.path.isfile(npz_path):
                raise FileNotFoundError(npz_path)
            digest = hashlib.sha256()
            with open(npz_path, 'rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            self.source_sha256 = digest.hexdigest()
            expected = str(expected_sha256 or '').strip().lower()
            if expected and self.source_sha256 != expected:
                raise ValueError(
                    f'driver source SHA-256 mismatch: actual={self.source_sha256}, expected={expected}'
                )
            z = np.load(npz_path)
            # Optional grid meta (for china_hires and other non-default grids)
            try:
                if 'grid_shape' in z.files:
                    gs = np.asarray(z['grid_shape']).reshape(-1)
                    if gs.size >= 2:
                        self.grid_shape = (int(gs[0]), int(gs[1]))
                if ('lon_deg' in z.files) and ('lat_deg' in z.files):
                    lon_deg = np.asarray(z['lon_deg'], dtype=np.float64).reshape(-1)
                    lat_deg = np.asarray(z['lat_deg'], dtype=np.float64).reshape(-1)
                    self._lon_deg = lon_deg
                    self._lat_deg = lat_deg
            except Exception:
                pass
            # Fallback to built-in default global 71x73 grid if lon/lat not provided.
            if self._lon_deg is None or self._lat_deg is None:
                try:
                    # IONEX/GIMtec convention:
                    #   - lat:  87.5 -> -87.5 (north to south), step=2.5 deg, 71 points
                    #   - lon: -180  ->  180  (inclusive), step=5 deg, 73 points
                    lon = np.linspace(-180.0, 180.0, int(self.grid_shape[1]), dtype=np.float64)
                    lat = np.linspace(87.5, -87.5, int(self.grid_shape[0]), dtype=np.float64)
                    Lon, Lat = np.meshgrid(lon, lat)
                    self._lon_deg = Lon.reshape(-1)
                    self._lat_deg = Lat.reshape(-1)
                except Exception:
                    self._lon_deg = None
                    self._lat_deg = None
            self.time = z['time']                  # [T]
            self.AE = z['AE'].astype('float32')
            self.Dst = z['Dst'].astype('float32')
            self.Kp = z['Kp'].astype('float32')
            self.F107 = z['F107'].astype('float32')
            self.cosZ = z['cosSZA'].astype('float32') if 'cosSZA' in z.files else None
        except Exception:
            if self.strict:
                raise
            # Fallback: synthetic zeros drivers (safe for smoke tests)
            self.synthetic = True
            self.driver_fallback_count += 1
            self.synthetic_driver_count += 1
            self.time = np.arange(0, 10_000, dtype=np.int64)
            self.AE = np.zeros((self.time.shape[0],), dtype=np.float32)
            self.Dst = np.zeros_like(self.AE)
            self.Kp = np.zeros_like(self.AE)
            self.F107 = np.zeros_like(self.AE)
            # cosSZA is optional and can be very large to materialize; skip in synthetic mode.
            self.cosZ = None

    def audit_payload(self) -> dict:
        return {
            'driver_clip_or_pad_count': int(self.driver_clip_or_pad_count),
            'driver_fallback_count': int(self.driver_fallback_count),
            'synthetic_driver_count': int(self.synthetic_driver_count),
            'driver_source_sha256': self.source_sha256,
            'driver_source_path': self.path,
        }

    def _compute_cos_sza(self, times_sec: np.ndarray, N: int) -> np.ndarray:
        """Compute cos(SZA) for given UTC unix seconds.

        Args:
            times_sec: int64 seconds array with shape [K] or [B,T].
            N: number of nodes to return (first N in flattened 71x73 grid).

        Returns:
            cosSZA: float64 array with shape [B,T,N] if input is [B,T], else [K,N].

            The canonical calculation remains float64.  Production callers cast
            once, at the final model/PPINN dtype boundary.
        """
        if self._lon_deg is None or self._lat_deg is None:
            raise RuntimeError('grid lon/lat not initialized')
        n = int(min(int(N), int(self._lon_deg.shape[0])))
        t = np.asarray(times_sec, dtype=np.int64)
        orig_shape = t.shape
        flat = t.reshape(-1)
        cosz = canonical_cos_sza_float64(
            flat,
            self._lon_deg[:n],
            self._lat_deg[:n],
        )
        if len(orig_shape) == 2:
            B, T = int(orig_shape[0]), int(orig_shape[1])
            return cosz.reshape(B, T, n)
        return cosz

    def slice_bt(self, start_idx: int, T: int, N: int) -> dict:
        s, e = int(start_idx), int(start_idx + T)
        source_length = int(self.AE.shape[0])
        if s < 0 or e > source_length:
            self.driver_clip_or_pad_count += 1
            if self.strict:
                raise IndexError(
                    f'driver clip/pad would be required: [{s}:{e}] outside [0:{source_length}]'
                )
        # Safe slicing with padding if out-of-range
        def _slice_pad(arr1d):
            L = arr1d.shape[0]
            if s < 0:
                s0 = 0
            else:
                s0 = s
            e0 = min(e, L)
            core = arr1d[s0:e0]
            if core.shape[0] < T:
                pad_len = T - core.shape[0]
                pad_val = core[-1] if core.shape[0] > 0 else (arr1d[0] if L > 0 else 0.0)
                core = np.concatenate([core, np.full((pad_len,), pad_val, dtype=np.float32)], axis=0)
            return torch.from_numpy(core.astype('float32')).to(self.device)

        def _slice_pad_2d(arr2d):
            L = arr2d.shape[0]
            s0 = max(0, s)
            e0 = min(e, L)
            core = arr2d[s0:e0, :N]
            if core.shape[0] < T:
                pad_len = T - core.shape[0]
                pad_row = core[-1:] if core.shape[0] > 0 else np.zeros((1, min(N, arr2d.shape[1])), dtype=np.float32)
                pad_block = np.repeat(pad_row, pad_len, axis=0)
                core = np.concatenate([core, pad_block], axis=0)
            return torch.from_numpy(core.astype('float32')).to(self.device)

        AE = _slice_pad(self.AE).view(1, T, 1).expand(1, T, N)
        Dst = _slice_pad(self.Dst).view(1, T, 1).expand(1, T, N)
        Kp = _slice_pad(self.Kp).view(1, T, 1).expand(1, T, N)
        F107 = _slice_pad(self.F107).view(1, T, 1).expand(1, T, N)
        out = {
            'AE': AE,
            'Dst': Dst,
            'Kp': Kp,
            'F107': F107,
            'grid_shape': self.grid_shape,
        }
        if self.cosZ is not None:
            out['cosSZA'] = _slice_pad_2d(self.cosZ).view(1, T, N)
        elif (not self.synthetic) and (self.time is not None):
            try:
                L = int(self.time.shape[0])
                idx = np.arange(int(s), int(e), dtype=np.int64)
                idx = np.clip(idx, 0, max(0, L - 1))
                times = self.time[idx]
                cos = self._compute_cos_sza(times, N)  # [T,n]
                if cos.ndim == 2:
                    out['cosSZA'] = torch.from_numpy(
                        cos.reshape(1, int(T), -1)
                    ).to(device=self.device, dtype=torch.float32)
            except Exception:
                if self.strict:
                    raise
        if self.strict:
            self._validate_slice(out, batch=1, horizon=int(T), nodes=int(N))
        return out

    def slice_batched(
        self,
        start_idx,
        T: int,
        N: int,
        *,
        fields: tuple[str, ...] | list[str] | None = None,
        require_cos_sza: bool = True,
        include_grid_shape: bool = True,
    ) -> dict:
        """Slice drivers for a batch of windows.

        Args:
            start_idx: int or array-like or torch.Tensor with shape [B].
            T: window length.
            N: number of nodes.

        Returns:
            dict with tensors on self.device:
              - AE/Dst/Kp/F107: [B, T]
              - cosSZA: [B, T, N] (if available)
        """
        # normalize start indices to numpy [B]
        if isinstance(start_idx, torch.Tensor):
            s = start_idx.detach().cpu().numpy().reshape(-1).astype(np.int64)
        else:
            s = np.asarray(start_idx, dtype=np.int64).reshape(-1)
        if s.size == 0:
            if self.strict:
                raise ValueError('driver start_idx is empty')
            self.driver_fallback_count += 1
            s = np.asarray([0], dtype=np.int64)
        B = int(s.shape[0])

        L = int(self.AE.shape[0])
        idx = s[:, None] + np.arange(int(T), dtype=np.int64)[None, :]
        if np.any(idx < 0) or np.any(idx >= L):
            self.driver_clip_or_pad_count += 1
            if self.strict:
                lo = int(idx.min()) if idx.size else 0
                hi = int(idx.max()) if idx.size else 0
                raise IndexError(
                    f'driver clip/pad would be required: requested index range [{lo},{hi}], source [0,{L - 1}]'
                )
        idx = np.clip(idx, 0, max(0, L - 1))

        def _g1(arr1d: np.ndarray) -> torch.Tensor:
            # arr1d: [L] -> [B,T]
            x = arr1d[idx]
            return torch.from_numpy(x.astype('float32')).to(self.device)

        requested = tuple(fields) if fields is not None else ('AE', 'Dst', 'Kp', 'F107')
        sources = {'AE': self.AE, 'Dst': self.Dst, 'Kp': self.Kp, 'F107': self.F107}
        unknown = sorted(set(requested) - set(sources))
        if unknown:
            raise ValueError(f'unknown driver fields requested: {unknown}')
        out = {field: _g1(sources[field]) for field in requested}
        if include_grid_shape:
            out['grid_shape'] = self.grid_shape
        if require_cos_sza and self.cosZ is not None:
            ncos = int(min(int(N), int(self.cosZ.shape[1])))
            cos = self.cosZ[idx, :ncos].astype('float32')  # [B,T,ncos]
            if ncos < int(N):
                self.driver_clip_or_pad_count += 1
                if self.strict:
                    raise ValueError(
                        f'cosSZA shape requires padding: source nodes={ncos}, requested nodes={int(N)}'
                    )
                pad = np.zeros((B, int(T), int(N) - ncos), dtype=np.float32)
                cos = np.concatenate([cos, pad], axis=2)
            out['cosSZA'] = torch.from_numpy(cos).to(self.device)
        elif require_cos_sza and (not self.synthetic) and (self.time is not None):
            try:
                times = self.time[idx]  # [B,T]
                cos = self._compute_cos_sza(times, N)  # [B,T,n]
                if cos.ndim == 3:
                    out['cosSZA'] = torch.from_numpy(cos).to(
                        device=self.device, dtype=torch.float32
                    )
            except Exception:
                if self.strict:
                    raise
        if self.strict:
            self._validate_slice(
                out,
                batch=B,
                horizon=int(T),
                nodes=int(N),
                required_fields=requested,
                require_cos_sza=bool(require_cos_sza),
                require_grid_shape=bool(include_grid_shape),
            )
        return out

    def _validate_slice(
        self,
        values: dict,
        *,
        batch: int,
        horizon: int,
        nodes: int,
        required_fields=('AE', 'Dst', 'Kp', 'F107'),
        require_cos_sza: bool = True,
        require_grid_shape: bool = True,
    ) -> None:
        for field in required_fields:
            value = values.get(field)
            expected = (batch, horizon)
            # slice_bt retains its historical [1,T,N] representation.
            if value is not None and tuple(value.shape) == (batch, horizon, nodes):
                pass
            elif value is None or tuple(value.shape) != expected:
                raise ValueError(
                    f'{field} driver shape mismatch: got {None if value is None else tuple(value.shape)}, '
                    f'expected {expected}'
                )
            if not torch.isfinite(value).all().item():
                raise ValueError(f'{field} driver contains nonfinite values')
        if require_cos_sza:
            cos_sza = values.get('cosSZA')
            if cos_sza is None:
                raise ValueError('cosSZA driver is missing')
            if tuple(cos_sza.shape) != (batch, horizon, nodes):
                raise ValueError(
                    f'cosSZA driver shape mismatch: got {tuple(cos_sza.shape)}, '
                    f'expected {(batch, horizon, nodes)}'
                )
            if not torch.isfinite(cos_sza).all().item():
                raise ValueError('cosSZA driver contains nonfinite values')
        if require_grid_shape:
            rows, columns = (int(v) for v in self.grid_shape)
            if rows * columns != nodes:
                raise ValueError(
                    f'driver grid_shape mismatch: got {self.grid_shape}, expected product {nodes}'
                )
