"""Training-year applicability table for the gated nighttime term (CPU only).

Eligible elements are exactly those the implemented night term acts on: the eleven internal
two-hour transitions of every training target day, start frame with cos(SZA) <= 1e-6 from the
frozen penalty driver store, both TEC values finite.  An element "increases" when the observed
TEC difference is strictly positive.  Bins: Kp class at the transition start (<3, 3-5, >=5),
calendar month, two-hour local-solar-time bin (UTC hour + longitude/15 at the start), and
ten-degree geographic latitude band (south to north).  The night term stays active only in bins
with at least MIN_COUNT eligible elements and an increase fraction <= THRESHOLD.
Only training years are read; validation and evaluation labels are never opened.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
from tecfc.training import global_train as base  # noqa: E402
from tecfc.objectives import penalty as pi  # noqa: E402

TRAIN_YEARS = (2009, 2010, 2011, 2012, 2014, 2016, 2018)
LAT = 87.5 - 2.5 * np.arange(71, dtype=np.float64)
LON = -180. + 5. * np.arange(73, dtype=np.float64)
NIGHT = 1e-6
KP_EDGES = (3., 5.)
THRESHOLD = .25
MIN_COUNT = 1000
SHAPE = (3, 12, 12, 18)  # Kp class, month, local-time bin, latitude band
DEFINITION = dict(
    eligible='internal transitions h=0..10 of each training target day; cosSZA(h)<=1e-6 at the start '
             '(same mask and driver store as the loss); both TEC values finite',
    increase='TEC(h+1)-TEC(h) > 0',
    kp_class='Kp at the transition start from the frozen penalty driver store: 0 (<3), 1 (3 to <5), 2 (>=5)',
    month='calendar month of the target day (UTC)',
    local_time_bin='floor(((UTC hour of the start + longitude/15) mod 24)/2), 12 bins',
    latitude_band='floor((latitude+90)/10), 18 bands from 90S to 90N; the 73rd longitude column is the +180 copy',
    gate='1 if count >= %d and increase fraction <= %.2f, else 0' % (MIN_COUNT, THRESHOLD))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def node_bins():
    lat_band = np.floor((LAT + 90.) / 10.).astype(np.int64)            # [71]
    if lat_band.min() < 0 or lat_band.max() > 17:
        raise ValueError('Latitude band outside 0..17')
    return np.repeat(lat_band, 73)                                        # [5183], latitude-major


def transition_bins(origins, kp):
    """Return flat bin indices [B, 11, 5183] for transition starts h = 0..10."""
    origins = np.asarray(origins, np.int64)
    starts = origins[:, None] + np.arange(11, dtype=np.int64)[None, :] * 7200       # [B, 11]
    hour = (starts % 86400).astype(np.float64) / 3600.
    local = np.mod(hour[:, :, None] + LON[None, None, :] / 15., 24.)                # [B, 11, 73]
    lt_bin = np.floor(local / 2.).astype(np.int64)
    lt_bin = np.repeat(lt_bin[:, :, None, :], 71, axis=2).reshape(len(origins), 11, 5183)
    month = (starts.astype('datetime64[s]').astype('datetime64[M]').astype(np.int64) % 12)  # [B, 11]
    kp = np.asarray(kp, np.float64)[:, :11]
    kp_class = (kp >= KP_EDGES[0]).astype(np.int64) + (kp >= KP_EDGES[1]).astype(np.int64)
    lat_band = node_bins()[None, None, :]
    flat = ((kp_class[:, :, None] * 12 + month[:, :, None]) * 12 + lt_bin) * 18 + lat_band
    return flat


def derive(config_path, out_dir):
    config = json.loads(Path(config_path).read_text())
    files = config['data_files']
    if not set(map(int, files)) >= set(TRAIN_YEARS):
        raise ValueError('Training years missing from the configuration')
    if sha(config['physics']['driver']) != config['physics']['driver_sha256']:
        raise ValueError('Changed penalty driver store')
    drivers = pi.Drivers(config['physics']['driver'], 'cpu')
    with base.reference_modules(config['reference_root'], config['asset_root']) as (gim, _):
        days = list(base.DevelopmentWindows('train', dict(config, data_files={y: files[y] for y in files}), gim).days)
    if len(days) != 2553:
        raise ValueError('Unexpected training window count %d' % len(days))
    count = np.zeros(int(np.prod(SHAPE)), np.int64)
    increase = np.zeros(int(np.prod(SHAPE)), np.int64)
    arrays = {}
    lat_rows = node_bins()
    for start in range(0, len(days), 64):
        chunk = days[start:start + 64]
        origins = np.array([int(np.datetime64(d, 's').astype(np.int64)) for d in chunk], np.int64)
        years = origins.astype('datetime64[s]').astype('datetime64[Y]').astype(np.int64) + 1970
        if not set(years.tolist()) <= set(TRAIN_YEARS):
            raise ValueError('Non-training target day')
        query = origins[:, None] + np.arange(12, dtype=np.int64)[None, :] * 7200
        ix = np.searchsorted(drivers.store.time, query)
        if not np.array_equal(drivers.store.time[ix], query):
            raise ValueError('Driver times disagree')
        raw = drivers.store.slice_batched(ix[:, 0], 12, 5183, fields=('Kp',), require_cos_sza=True,
                                          include_grid_shape=False)
        cos = raw['cosSZA'].detach().cpu().numpy().astype(np.float64)              # [B, 12, 5183]
        kp = raw['Kp'].detach().cpu().numpy()                                       # [B, 12]
        tec = np.empty((len(chunk), 12, 5183), np.float64)
        for i, day in enumerate(chunk):
            year = day[:4]
            if year not in arrays:
                arrays[year] = np.load(files[year]['path'], mmap_mode='r', allow_pickle=False)
            offset = int((np.datetime64(day, 'D') - np.datetime64(year + '-01-01')).astype(int))
            tec[i] = np.asarray(arrays[year][offset], np.float64).reshape(12, 5183)
        night = cos[:, :11, :] <= NIGHT
        valid = np.isfinite(tec[:, :11]) & np.isfinite(tec[:, 1:])
        eligible = night & valid
        grow = eligible & ((tec[:, 1:] - tec[:, :11]) > 0)
        flat = transition_bins(origins, kp)
        count += np.bincount(flat[eligible], minlength=count.size)
        increase += np.bincount(flat[grow], minlength=count.size)
    count, increase = count.reshape(SHAPE), increase.reshape(SHAPE)
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = np.where(count > 0, increase / np.maximum(count, 1), np.nan)
    gate = ((count >= MIN_COUNT) & (fraction <= THRESHOLD)).astype(np.uint8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / 'NIGHT_GATE.npz'
    np.savez(table, count=count, increase=increase, gate=gate,
             axes=np.array(['kp_class', 'month', 'local_time_bin', 'latitude_band']))

    def share(axis_keep):
        axes = tuple(i for i in range(4) if i != axis_keep)
        c, g = count.sum(axis=axes), increase.sum(axis=axes)
        return [dict(bin=int(i), eligible=int(c[i]), increase_percent=float(100 * g[i] / c[i]) if c[i] else None)
                for i in range(len(c))]

    covered = int(count[gate == 1].sum())
    summary = dict(stage='night_gate', definition=DEFINITION,
                   threshold=THRESHOLD, minimum_count=MIN_COUNT, shape=list(SHAPE),
                   training_days=len(days), eligible_elements=int(count.sum()),
                   increase_elements=int(increase.sum()),
                   increase_percent=float(100 * increase.sum() / count.sum()),
                   gate_on_bins=int(gate.sum()), bins_with_data=int((count > 0).sum()),
                   eligible_share_in_gate_percent=float(100 * covered / count.sum()),
                   increase_percent_inside_gate=float(100 * increase[gate == 1].sum() / max(covered, 1)),
                   by_kp_class=share(0), by_month=share(1), by_local_time_bin=share(2), by_latitude_band=share(3),
                   table=table.name, table_sha256=sha(table),
                   driver_sha256=config['physics']['driver_sha256'],
                   data_sha256={y: files[y]['sha256'] for y in map(str, TRAIN_YEARS)},
                   script_sha256=sha(__file__), labels_opened=sorted(arrays))
    (out_dir / 'NIGHT_GATE.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary
