#!/usr/bin/env python3
"""
ecmwf_fetch.py - download the latest ECMWF IFS open-data forecast and write ecmwf_grid.js
for weather.html ("Local ECMWF file" data source). No Open-Meteo calls are used.

Setup (once):
    pip install ecmwf-opendata eccodes numpy

Run (about 300-400 MB is downloaded per forecast run; files are cached in ./ecmwf_cache):
    python ecmwf_fetch.py

Try the page without downloading anything (synthetic data, clearly labelled DEMO in the page):
    python ecmwf_fetch.py --demo

Then put ecmwf_grid.js in the same folder as weather.html and open or reload the page.
Run it again every 6 hours or so for a fresh forecast (ECMWF publishes 00/06/12/18 UTC runs).

Useful options:
    --bbox S W N E     area to keep (default 10 -25 70 65: Europe, N Africa, Middle East)
    --res 0.5          output grid spacing in degrees; must be a multiple of the source spacing
                       (0.25 for --resol 0p25). Smaller = sharper map but a bigger file.
    --maxstep 90       forecast hours to download (3-hourly). 90 covers a run that is up to
                       about 18 h old plus the page's 72 h window.
    --resol 0p25       ECMWF resolution key. ECMWF added a 0.1 degree (about 9 km) product on
                       20 Oct 2026; try --resol 0p1 --res 0.1 once 0p25 works (untested).
    --source ecmwf     ecmwf | aws | azure | google (mirrors; use one if ecmwf is slow)
    --out PATH         output file (default ecmwf_grid.js in the current folder)
"""
import argparse
import base64
import datetime as dt
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

KT = 1.943844  # m/s -> knots
OUT_VARS = ['temperature_2m', 'dew_point_2m', 'wind_speed_10m', 'wind_direction_10m',
            'wind_gusts_10m', 'precipitation', 'cloud_cover', 'pressure_msl', 'cape']
# ECMWF short names to try for each quantity, in order (open-data names differ between docs versions)
CAND = {'2t': ['2t'], '2d': ['2d'], '10u': ['10u'], '10v': ['10v'],
        'gust': ['10fg', 'fg10', '10fg3', 'i10fg'], 'tp': ['tp'], 'tcc': ['tcc'],
        'msl': ['msl'], 'cape': ['cape', 'mucape']}


def quantize(a):
    """float32 array -> (offset, scale, int16 array). -32768 marks missing values."""
    ok = np.isfinite(a)
    if not ok.any():
        return 0.0, 1.0, np.full(a.shape, -32768, np.int16)
    lo, hi = float(a[ok].min()), float(a[ok].max())
    scale = (hi - lo) / 64000.0 or 1.0
    q = np.round((np.where(ok, a, lo) - lo) / scale).astype(np.int32) - 32000
    q = np.where(ok, q, -32768).astype(np.int16)
    return lo, scale, q


def write_js(path, run, steps, lats, lons, res, data, missing, demo, source):
    vars_ = {}
    for name in OUT_VARS:
        if name not in data:
            continue
        off, scale, q = quantize(np.asarray(data[name], np.float32))
        vars_[name] = {'off': off, 'scale': scale,
                       'data': base64.b64encode(gzip.compress(q.astype('<i2').tobytes(), 6)).decode()}
    obj = {'run': run.strftime('%Y-%m-%dT%H:%M:%SZ'), 't0': int(run.replace(tzinfo=dt.timezone.utc).timestamp()),
           'steps': [int(s) for s in steps], 'lat0': float(lats[0]), 'lon0': float(lons[0]), 'res': res,
           'nx': len(lons), 'ny': len(lats), 'vars': vars_, 'missing': missing, 'demo': demo, 'source': source,
           'created': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('window.ECMWF_LOCAL=')
        json.dump(obj, f, separators=(',', ':'))
        f.write(';\n')
    print(f'Wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB, {len(lons)}x{len(lats)} points, '
          f'{len(steps)} steps, run {obj["run"]})')


# ------------------------------------------------------------------ demo (synthetic) data
def demo(args):
    S, W, N, E = args.bbox
    res = args.res
    lats = np.arange(math.ceil(S / res) * res, N + 1e-9, res)
    lons = np.arange(math.ceil(W / res) * res, E + 1e-9, res)
    steps = list(range(0, args.maxstep + 1, 3))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    run = (now - dt.timedelta(hours=7)).replace(hour=((now - dt.timedelta(hours=7)).hour // 6) * 6)
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    out = {k: [] for k in OUT_VARS}
    for h in steps:
        hh = h + run.hour
        t = 15 - 0.4 * (LA - 45) + 4 * np.sin(hh / 24 * 2 * np.pi - 1) + 3 * np.sin(LO / 15)
        out['temperature_2m'].append(t)
        out['dew_point_2m'].append(t - 5 - 2 * np.cos(LA / 9 + h / 20))
        sp = np.maximum(0, 12 + 9 * np.sin(LO / 8 + h / 10) + 6 * np.cos(LA / 6))
        out['wind_speed_10m'].append(sp)
        out['wind_direction_10m'].append((LO * 3 + LA * 5 + h * 4) % 360)
        out['wind_gusts_10m'].append(sp * 1.5)
        out['precipitation'].append(np.maximum(0, 3 * np.sin(LA / 4 + h / 9) * np.cos(LO / 6)))
        out['cloud_cover'].append(np.clip(50 + 45 * np.sin(LA / 7 + LO / 9 + h / 12), 0, 100))
        out['pressure_msl'].append(1013 + 18 * np.sin(LO / 12 + h / 30) * np.cos(LA / 10))
        out['cape'].append(np.maximum(0, 900 * np.sin(LA / 5) * np.sin(h / 8)))
    data = {k: np.stack(v) for k, v in out.items()}
    return run, steps, lats, lons, res, data, [], True


# ------------------------------------------------------------------ real download
class Grid:
    """Lat/lon selection computed once from the first GRIB message and applied to every field."""
    def __init__(self):
        self.ready = False

    def setup(self, la1, la2, lo1, lo2, ni, nj, bbox, res):
        lats = np.linspace(la1, la2, nj)
        if lo2 < lo1:
            lo2 += 360
        lons = np.linspace(lo1, lo2, ni)
        lons = ((lons + 180) % 360) - 180
        self.lat_order = np.argsort(lats)
        self.lon_order = np.argsort(lons)
        lats, lons = lats[self.lat_order], lons[self.lon_order]
        S, W, N, E = bbox
        mult = lambda x: np.isclose(x / res, np.round(x / res), atol=1e-6)
        self.ilat = np.where((lats >= S - 1e-9) & (lats <= N + 1e-9) & mult(lats))[0]
        self.ilon = np.where((lons >= W - 1e-9) & (lons <= E + 1e-9) & mult(lons))[0]
        if len(self.ilat) < 2 or len(self.ilon) < 2:
            sys.exit('The bounding box / --res selects fewer than 2 grid rows or columns. Check --bbox and --res.')
        self.lats, self.lons = lats[self.ilat], lons[self.ilon]
        self.ready = True

    def apply(self, v):
        return v[np.ix_(self.lat_order, self.lon_order)][np.ix_(self.ilat, self.ilon)]


def read_grib(path, grid, bbox, res):
    from eccodes import codes_grib_new_from_file, codes_get, codes_get_values, codes_release
    fields = {}
    with open(path, 'rb') as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                step = int(codes_get(gid, 'endStep'))
                ni, nj = int(codes_get(gid, 'Ni')), int(codes_get(gid, 'Nj'))
                if not grid.ready:
                    grid.setup(codes_get(gid, 'latitudeOfFirstGridPointInDegrees'),
                               codes_get(gid, 'latitudeOfLastGridPointInDegrees'),
                               codes_get(gid, 'longitudeOfFirstGridPointInDegrees'),
                               codes_get(gid, 'longitudeOfLastGridPointInDegrees'), ni, nj, bbox, res)
                fields[step] = grid.apply(codes_get_values(gid).reshape(nj, ni).astype(np.float32))
            finally:
                codes_release(gid)
    return fields


def download(args):
    try:
        from ecmwf.opendata import Client
        import eccodes  # noqa: F401
    except ImportError as e:
        sys.exit(f'Missing package ({e}). Run:  pip install ecmwf-opendata eccodes numpy')
    client = Client(source=args.source, model='ifs', resol=args.resol)
    steps = list(range(0, args.maxstep + 1, 3))
    req = {'type': 'fc', 'step': steps}
    date = hour = None
    try:
        latest = client.latest(type='fc', param='2t', step=steps[-1])
        date, hour = latest.strftime('%Y%m%d'), latest.hour
        run = latest.replace(tzinfo=None)
        print(f'Latest available run: {latest:%Y-%m-%d %H}Z')
    except Exception as e:
        print(f'Could not ask for the latest run ({e}); using the client default.')
        run = None
    if date:
        req.update(date=date, time=hour)
    os.makedirs(args.cache, exist_ok=True)
    tag = f'{date}{hour:02d}' if date else 'latest'
    for old in glob.glob(os.path.join(args.cache, '*.grib2')):  # drop other runs
        if tag not in os.path.basename(old):
            os.remove(old)
    grid, missing, got = Grid(), [], {}
    for key, names in CAND.items():
        for sn in names:
            target = os.path.join(args.cache, f'{tag}_{sn}.grib2')
            try:
                if not os.path.exists(target):
                    print(f'Downloading {sn} ({len(steps)} steps)...')
                    r = client.retrieve(param=sn, target=target, **req)
                    if run is None and getattr(r, 'datetime', None):
                        run = r.datetime.replace(tzinfo=None)
                fields = read_grib(target, grid, args.bbox, args.res)
                if not all(s in fields for s in steps):
                    raise RuntimeError('some forecast steps are missing in the file')
                got[key] = np.stack([fields[s] for s in steps])
                print(f'  {key}: using ECMWF parameter "{sn}"')
                break
            except Exception as e:
                if os.path.exists(target):
                    os.remove(target)
                print(f'  {key}: "{sn}" not usable ({str(e)[:120]})')
    if '2t' not in got:
        sys.exit('Temperature (2t) could not be downloaded, so nothing was written. Check your connection / --source.')
    if run is None:
        sys.exit('Could not work out the forecast run time.')
    data = {'temperature_2m': got['2t'] - 273.15}
    names = {'2d': ('dew_point_2m', 'dew point'), '10u': (None, 'wind u'), '10v': (None, 'wind v'),
             'gust': ('wind_gusts_10m', 'wind gusts'), 'tcc': ('cloud_cover', 'cloud cover'),
             'msl': ('pressure_msl', 'pressure'), 'cape': ('cape', 'CAPE'), 'tp': ('precipitation', 'precipitation')}
    if '2d' in got:
        data['dew_point_2m'] = got['2d'] - 273.15
    if '10u' in got and '10v' in got:
        u, v = got['10u'], got['10v']
        data['wind_speed_10m'] = np.hypot(u, v) * KT
        data['wind_direction_10m'] = (270 - np.degrees(np.arctan2(v, u))) % 360
    if 'gust' in got:
        data['wind_gusts_10m'] = got['gust'] * KT
    if 'tcc' in got:
        data['cloud_cover'] = got['tcc'] * (100 if np.nanmax(got['tcc']) <= 1.5 else 1)
    if 'msl' in got:
        data['pressure_msl'] = got['msl'] / 100.0
    if 'cape' in got:
        data['cape'] = got['cape']
    if 'tp' in got:  # accumulated metres since the start -> mm/h per step
        tp = got['tp']
        rate = np.diff(tp, axis=0) / np.diff(steps).reshape(-1, 1, 1) * 1000.0
        data['precipitation'] = np.clip(np.concatenate([rate[:1], rate]), 0, None)
    missing = [OUT_VARS[i] for i in range(len(OUT_VARS)) if OUT_VARS[i] not in data]
    return run, steps, grid.lats, grid.lons, args.res, data, missing, False


def main():
    ap = argparse.ArgumentParser(description='ECMWF open data -> ecmwf_grid.js for weather.html')
    ap.add_argument('--bbox', nargs=4, type=float, default=[10, -25, 70, 65], metavar=('S', 'W', 'N', 'E'))
    ap.add_argument('--res', type=float, default=0.5)
    ap.add_argument('--maxstep', type=int, default=90)
    ap.add_argument('--resol', default='0p25')
    ap.add_argument('--source', default='ecmwf', choices=['ecmwf', 'aws', 'azure', 'google'])
    ap.add_argument('--cache', default='ecmwf_cache')
    ap.add_argument('--out', default='ecmwf_grid.js')
    ap.add_argument('--demo', action='store_true', help='write synthetic data (no download) to test the page')
    args = ap.parse_args()
    if args.maxstep % 3:
        sys.exit('--maxstep must be a multiple of 3.')
    if args.demo:
        run, steps, lats, lons, res, data, missing, is_demo = demo(args)
    else:
        run, steps, lats, lons, res, data, missing, is_demo = download(args)
    if missing:
        print('Not available (those layers will be empty in the page):', ', '.join(missing))
    write_js(args.out, run, steps, lats, lons, res, data, missing, is_demo,
             'DEMO synthetic data' if is_demo else f'ECMWF open data ({args.resol}, source {args.source})')


if __name__ == '__main__':
    main()
