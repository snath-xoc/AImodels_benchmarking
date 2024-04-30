##regridding for data, needs input lon/lat and target lon/lat

import xesmf
import xarray as xr


def regrid(lons_out, lats_out, df_in):
    """

    Thank god for xesmf, this implements conservative remapping assuming xarray

    Input
    -----

    lons_out: ndarray (lons_out)
              target resolution's longitude values

    lats_out: ndarray (lats_out)
              target resolution's latitude values

    df_in: xr.Dataset
           original dataset that needs to be regridded

    Returns
    -------

    Regridder object to then be called to regrid xr.DataArray

    """

    df_regridded = xr.Dataset(
        {
            "lat": (["lat"], lats_out, {"units": "degrees_north"}),
            "lon": (["lon"], lons_out, {"units": "degrees_east"}),
        }
    )

    regridder = xesmf.Regridder(df_in, df_regridded, "conservative")

    return regridder
