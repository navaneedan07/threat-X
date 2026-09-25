"""Dataset inspection for ZIP-wrapped ERA5 NetCDF files."""
import zipfile
import io
import xarray as xr

for outer_fname in ['data/raw/era5_amphan.nc', 'data/raw/era5_heatwave.nc']:
    print()
    print('='*60)
    print('ARCHIVE:', outer_fname)
    print('='*60)
    with zipfile.ZipFile(outer_fname) as z:
        for inner_name in z.namelist():
            data = z.read(inner_name)
            buf = io.BytesIO(data)
            print(f'\n  INNER: {inner_name}')
            opened = False
            for engine in ['netcdf4', 'scipy', 'h5netcdf']:
                try:
                    buf.seek(0)
                    ds = xr.open_dataset(buf, engine=engine)
                    print(f'  opened with {engine}')
                    print('  Variables:', list(ds.data_vars))
                    print('  Coords:', list(ds.coords))
                    print('  Dims:', dict(ds.dims))
                    for v in ds.data_vars:
                        da = ds[v]
                        units = da.attrs.get('units', '?')
                        lname = da.attrs.get('long_name', '?')
                        print(f'    VAR {v}: shape={da.shape} dims={da.dims}')
                        print(f'        units={units} | {lname}')
                    for tname in ['valid_time', 'time']:
                        if tname in ds.coords:
                            t = ds[tname]
                            print(f'  {tname}: {t.values[0]} ... {t.values[-1]} ({len(t)} steps)')
                            break
                    if 'latitude' in ds.coords:
                        print(f'  lat: {float(ds.latitude.min()):.3f} to {float(ds.latitude.max()):.3f}')
                        print(f'  lon: {float(ds.longitude.min()):.3f} to {float(ds.longitude.max()):.3f}')
                    for cname in ['pressure_level', 'level']:
                        if cname in ds.coords:
                            print(f'  {cname}: {list(ds[cname].values)}')
                    ds.close()
                    opened = True
                    break
                except Exception as e:
                    print(f'  {engine} failed: {str(e)[:120]}')
            if not opened:
                print('  COULD NOT OPEN')
