from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch


class DriversStore:
    """Load drivers prepared by tools/prepare_drivers_omni.py and slice by global time index.

    This is optional; for zero-driver PPINN, you do not need this.
    """

    def __init__(self, npz_path: str, device='cpu'):
        self.device = device
        self.grid_shape = (71, 73)
        self.synthetic = False
        # Grid lon/lat for cosSZA on-the-fly. Prefer npz-provided lon/lat (supports non-71x73 grids).
        self._lon_deg = None
        self._sin_lat = None
        self._cos_lat = None
        try:
            z = np.load(npz_path)
            # Optional grid meta (for china_hires and other non-default grids)
            try:
                if 'grid_shape' in z.files:
                    gs = np.asarray(z['grid_shape']).reshape(-1)
                    if gs.size >= 2:
                        self.grid_shape = (int(gs[0]), int(gs[1]))
                if ('lon_deg' in z.files) and ('lat_deg' in z.files):
                    lon_deg = np.asarray(z['lon_deg'], dtype=np.float32).reshape(-1)
                    lat_deg = np.asarray(z['lat_deg'], dtype=np.float32).reshape(-1)
                    lat_rad = np.deg2rad(lat_deg).astype(np.float32)
                    self._lon_deg = lon_deg
                    self._sin_lat = np.sin(lat_rad).astype(np.float32)
                    self._cos_lat = np.cos(lat_rad).astype(np.float32)
            except Exception:
                pass
            # Fallback to built-in default global 71x73 grid if lon/lat not provided.
            if self._lon_deg is None or self._sin_lat is None or self._cos_lat is None:
                try:
                    # IONEX/GIMtec convention:
                    #   - lat:  87.5 -> -87.5 (north to south), step=2.5 deg, 71 points
                    #   - lon: -180  ->  180  (inclusive), step=5 deg, 73 points
                    lon = np.linspace(-180.0, 180.0, int(self.grid_shape[1]), dtype=np.float32)
                    lat = np.linspace(87.5, -87.5, int(self.grid_shape[0]), dtype=np.float32)
                    Lon, Lat = np.meshgrid(lon, lat)
                    lon_deg = Lon.reshape(-1).astype(np.float32)
                    lat_rad = np.deg2rad(Lat.reshape(-1).astype(np.float32))
                    self._lon_deg = lon_deg
                    self._sin_lat = np.sin(lat_rad).astype(np.float32)
                    self._cos_lat = np.cos(lat_rad).astype(np.float32)
                except Exception:
                    self._lon_deg = None
                    self._sin_lat = None
                    self._cos_lat = None
            self.time = z['time']                  # [T]
            self.AE = z['AE'].astype('float32')
            self.Dst = z['Dst'].astype('float32')
            self.Kp = z['Kp'].astype('float32')
            self.F107 = z['F107'].astype('float32')
            self.cosZ = z['cosSZA'].astype('float32') if 'cosSZA' in z.files else None
        except Exception:
            # Fallback: synthetic zeros drivers (safe for smoke tests)
            self.synthetic = True
            self.time = np.arange(0, 10_000, dtype=np.int64)
            self.AE = np.zeros((self.time.shape[0],), dtype=np.float32)
            self.Dst = np.zeros_like(self.AE)
            self.Kp = np.zeros_like(self.AE)
            self.F107 = np.zeros_like(self.AE)
            # cosSZA is optional and can be very large to materialize; skip in synthetic mode.
            self.cosZ = None

    def _compute_cos_sza(self, times_sec: np.ndarray, N: int) -> np.ndarray:
        """Compute cos(SZA) for given UTC unix seconds.

        Args:
            times_sec: int64 seconds array with shape [K] or [B,T].
            N: number of nodes to return (first N in flattened 71x73 grid).

        Returns:
            cosSZA: float32 array with shape [B,T,N] if input is [B,T], else [K,N].
        """
        if self._lon_deg is None or self._sin_lat is None or self._cos_lat is None:
            raise RuntimeError('grid lon/lat not initialized')
        n = int(min(int(N), int(self._lon_deg.shape[0])))
        t = np.asarray(times_sec, dtype=np.int64)
        orig_shape = t.shape
        flat = t.reshape(-1)

        dts = [datetime.fromtimestamp(int(s), tz=timezone.utc) for s in flat.tolist()]
        doy = np.array([dt.timetuple().tm_yday for dt in dts], dtype=np.float32)
        hour = np.array([dt.hour for dt in dts], dtype=np.float32)
        minute = np.array([dt.minute for dt in dts], dtype=np.float32)

        gamma = 2.0 * np.pi * (doy - 1.0) / 365.0
        decl = (
            0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
        ).astype(np.float32)
        eqtime = 229.18 * (
            0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
        )
        eqtime = eqtime.astype(np.float32)

        time_min = hour * 60.0 + minute + eqtime  # [K]
        ht_deg = (time_min / 4.0 - 180.0).reshape(-1, 1) + self._lon_deg[:n].reshape(1, -1)  # [K,n]
        ht = np.deg2rad(ht_deg).astype(np.float32)
        sin_decl = np.sin(decl).astype(np.float32).reshape(-1, 1)
        cos_decl = np.cos(decl).astype(np.float32).reshape(-1, 1)
        cosz = self._sin_lat[:n].reshape(1, -1) * sin_decl + self._cos_lat[:n].reshape(1, -1) * cos_decl * np.cos(ht)
        cosz = np.clip(cosz, 0.0, 1.0).astype(np.float32)
        if len(orig_shape) == 2:
            B, T = int(orig_shape[0]), int(orig_shape[1])
            return cosz.reshape(B, T, n)
        return cosz

    def slice_bt(self, start_idx: int, T: int, N: int) -> dict:
        s, e = int(start_idx), int(start_idx + T)
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
            'alpha_nd': 1e-2, 'beta_nd': 1e-2, 'use_adv': False,
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
                    out['cosSZA'] = torch.from_numpy(cos.reshape(1, int(T), -1)).to(self.device)
            except Exception:
                pass
        return out

    def slice_batched(self, start_idx, T: int, N: int) -> dict:
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
            s = np.asarray([0], dtype=np.int64)
        B = int(s.shape[0])

        L = int(self.AE.shape[0])
        idx = s[:, None] + np.arange(int(T), dtype=np.int64)[None, :]
        idx = np.clip(idx, 0, max(0, L - 1))

        def _g1(arr1d: np.ndarray) -> torch.Tensor:
            # arr1d: [L] -> [B,T]
            x = arr1d[idx]
            return torch.from_numpy(x.astype('float32')).to(self.device)

        out = {
            'AE': _g1(self.AE),
            'Dst': _g1(self.Dst),
            'Kp': _g1(self.Kp),
            'F107': _g1(self.F107),
            'grid_shape': self.grid_shape,
            'alpha_nd': 1e-2, 'beta_nd': 1e-2, 'use_adv': False,
        }
        if self.cosZ is not None:
            ncos = int(min(int(N), int(self.cosZ.shape[1])))
            cos = self.cosZ[idx, :ncos].astype('float32')  # [B,T,ncos]
            if ncos < int(N):
                pad = np.zeros((B, int(T), int(N) - ncos), dtype=np.float32)
                cos = np.concatenate([cos, pad], axis=2)
            out['cosSZA'] = torch.from_numpy(cos).to(self.device)
        elif (not self.synthetic) and (self.time is not None):
            try:
                times = self.time[idx]  # [B,T]
                cos = self._compute_cos_sza(times, N)  # [B,T,n]
                if cos.ndim == 3:
                    out['cosSZA'] = torch.from_numpy(cos).to(self.device)
            except Exception:
                pass
        return out
