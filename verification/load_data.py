## functions for loading data
import numpy as np
import xarray as xr
import glob
from joblib import Parallel, delayed
import h5py

import regridding
import importlib

importlib.reload(regridding)
from regridding import regrid
from tqdm import tqdm

# import gcsfs

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

AI_MODEL_PATH = "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/"

TRUTH_PATH_netcdf = (
    "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/ICPAC_region/24h/"
)
TRUTH_PATH = "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/"
TRUTH_PATH_LATE = (
    "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/late_run/"
)
ERA5_PATH = "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/graphcast/sample_data/surface/total_precipitation/"

ERA5_VAR_LOOKUP = {"tp": "total_precipitation_24hr"}


def get_cGAN(dir, year):
    """
    get cGAN output, kept this code flexible so that I can switch between different cGAN versions

    Input
    -----

    dir: str
         directory where cGAN output is stored
    year: int
         year for which to look for data

    Output
    ------

    xr.Dataset (time, lon, lat,) of cGAN output

    """

    files = glob.glob(dir + f"GAN_forecasts_{year}/*.nc")

    return xr.open_mfdataset(files, combine="by_coords")


def get_IMERG_year(year):
    files = glob.glob(TRUTH_PATH_netcdf + f"{year}*.nc")

    return xr.open_mfdataset(files, combine="by_coords").rename(
        {"latitude": "lat", "longitude": "lon"}
    )


def get_all_day_hdf5(file, idx_x, idx_y, late=False):
    """

    Return precipitation values for IMERG

    Inputs
    ------

    file: str
          Single file path

    idx_x: ndarray (lons,)
           indices within IMERG dataset that correspond to ICAPC regions lon

    idx_y: ndarray (lats,)
           indices within IMERG dataset that correspond to ICAPC regions lat

    late: Boolean
          Sometimes the data isn't available and we use the "late" runs in which case variables are named different

    Returns
    -------

    values: ndarray (time, lat, lon)
            precipitation values

    """

    with h5py.File(file, "r") as f:
        if late:
            values = f["Grid"]["precipitationCal"][:]
        else:
            values = f["Grid"]["precipitation"][:]

    if values.ndim == 2:
        values = values[None, :, :]

    values = np.swapaxes(values, 1, 2)[:, idx_y, :][:, :, idx_x]

    return values


def get_mean_over_day(time, idx_x, idx_y):
    """
    Loop through all IMERG HDF5 files available at a given day and return their mean

    Inputs
    ------

    time: np.datetime64[ns]
          date on which to loop through

    idx_x: ndarray (lons,)
           indices within IMERG dataset that correspond to ICAPC regions lon, to be passed onto get_all_day_hdf5

    idx_y: ndarray (lats,)
           indices within IMERG dataset that correspond to ICAPC regions lat, to be passed onto get_all_day_hdf5

    Returns
    -------

    values: ndarray (time, lat, lon)
            precipitation values

    Notes
    -----

    Checks if files are available in the default folder otherwise switched to "late_run". If nothing is available it returns
    a full ndarray of shape (1, idx_y, idx_x) with NaN values

    """

    late = False

    date = str(time).split("T")[0].replace("-", "")
    year = date[:4]
    month = months[int(date[4:6]) - 1]

    files = glob.glob(TRUTH_PATH + "%s/%s/*%s*.HDF5" % (year, month, date))
    # print(files)

    if len(files) == 0:
        files = glob.glob(TRUTH_PATH_LATE + "%s/%s/*%s*.HDF5" % (year, month, date))
        late = True

        if len(files) == 0:
            print("Could not find any IMERG data for, ", time)
            return np.full([len(idx_y), len(idx_x)], np.nan)

    try:
        return np.nanmean(
            np.vstack(
                ([get_all_day_hdf5(file, idx_x, idx_y, late=late) for file in files])
            ),
            axis=0,
        )

    except:
        return np.full([len(idx_y), len(idx_x)], np.nan)


def get_ai_models(models, year):
    """

    Get AI model forecast output for a given year. Assumes that NETCDF4 file of output is generated for the whole year and ICPAC region

    Input
    -----

    models: str or list
            model to get forecast for, currently only dmgc (graphcast) and fuxi

    year: int
          year to get forecast for

    Returns
    ------

    List of xr.Datasets for the forecasts, length is same as number of models

    """

    if not isinstance(models, list):
        models = [models]

    df = []

    for model in models:
        files = glob.glob(AI_MODEL_PATH + "%s/*%s*.nc" % (model, str(year)))
        df.append(xr.open_mfdataset(files))

    return df


def get_era5():
    """

    Brute force function to get all era5 data

    """

    files = glob.glob(ERA5_PATH + "*.nc")

    return xr.open_mfdataset(files)


def get_lon_lat_reg(df_reg):
    """

    Get corresponding lon and lat on the IMERG grid (0.1) for that in the AI/ERA5 data (0.25), to then be
    used for conservative regridding

    Input
    -----

    df_reg: xr.Dataset (time,lat,lon,)
            Regional dataset as extracted for AI/ERA5

    Returns
    -------

    lat_reg and lon_reg representing ICPAC region lat and lon as existing in IMERG grid
    """

    sample_IMERG = glob.glob(
        "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/2021/Jan/*.HDF5"
    )[0]

    with h5py.File(sample_IMERG, "r") as f:
        lats_IMERG = f["Grid"]["lat"][:]
        lons_IMERG = f["Grid"]["lon"][:]

    df_IMERG_test = xr.open_dataset(glob.glob(TRUTH_PATH_netcdf + f"{2020}*.nc")[0])

    return (
        df_IMERG_test.latitude.values,
        df_IMERG_test.longitude.values,
        lats_IMERG,
        lons_IMERG,
    )


def load_and_regrid_data(
    models, year, era5=True, var="tp", ARCO_ERA5=False, true_netcdf=False
):
    """
    Load AI model forecast output and their corresponding truth, observation values. If era5 is True then era5 us also provided.

    Input
    -----

    models: str or list
            models to get forecast output for

    year: int
          year for which to get available forecasts for

    var: Variable name in the dataset
         Optional, default is tp for total_precipitation

    ARCO_ERA5: Boolean
               Experimental if we want to load from WeatherBench2 cloud storage bucket

    Output
    ------

    df_ai_regridded: xr.Dataset (model, time, lat, lon,)

    df_era5_regridded (if era5 is true): xr.Dataset (time, lat, lon,)

    df_IMERG: xr.Dataset (time, lat, lon,)

    """

    df_ai = get_ai_models(
        models, year
    )  ## we assume all ai output are downloaded at same grid type/resolution as ERA5

    if true_netcdf:
        df_IMERG = get_IMERG_year(year)

        lats_reg = df_IMERG.lat.values
        lons_reg = df_IMERG.lon.values

        df_ai_regridded = []

        for model, df in zip(models, df_ai):
            regridder = regrid(lons_reg, lats_reg, df)
            df_ai_regridded.append(
                regridder(
                    df[var].mean("step").drop_vars(["surface"]), keep_attrs=True
                ).expand_dims(dim={"model": [model]}, axis=0)
            )

        df_ai_regridded = xr.concat(df_ai_regridded, "model")
        df_IMERG["time"] = df_IMERG.time - np.timedelta64(6, "h")

    else:
        lats_reg, lons_reg, lats_IMERG, lons_IMERG = get_lon_lat_reg(df_ai[0])

        df_ai_regridded = []

        for model, df in zip(models, df_ai):
            regridder = regrid(lons_reg, lats_reg, df)
            df_ai_regridded.append(
                regridder(df[var], keep_attrs=True).expand_dims(
                    dim={"model": [model]}, axis=0
                )
            )

        df_ai_regridded = xr.concat(df_ai_regridded, "model")

        idx_x = np.squeeze(np.argwhere(np.isin(lons_IMERG, df_ai_regridded.lon.values)))
        idx_y = np.squeeze(np.argwhere(np.isin(lats_IMERG, df_ai_regridded.lat.values)))

        ## simple for graphcast output
        # values_IMERG = Parallel(n_jobs=3)(delayed(get_mean_over_day)(time, idx_x, idx_y) for time in tqdm(df_ai_regridded.time.values))
        values_IMERG = [
            get_mean_over_day(time, idx_x, idx_y)
            for time in tqdm(df_ai_regridded.time.values)
        ]

        values_IMERG = np.stack((values_IMERG))
        print(values_IMERG.shape)

        df_IMERG = (
            xr.DataArray(
                data=values_IMERG,
                dims=["time", "lat", "lon"],
                coords={
                    "lat": (["lat"], lats_reg, {"units": "degrees_north"}),
                    "lon": (["lon"], lons_reg, {"units": "degrees_east"}),
                    "time": df_ai_regridded.time.values,
                },
            )
            .rename("precipitation")
            .to_dataset()
        )

        print(
            "Regridding data for models:",
            models,
            " at year:",
            year,
            lats_reg.shape,
            lons_reg.shape,
        )

    if era5:
        var_era5 = var

        if ARCO_ERA5:
            times = np.arange(
                "2018-01-01",
                "2023-01-01",
                np.timedelta64(6, "h"),
                dtype="datetime64[ns]",
            )

            var_era5 = ERA5_VAR_LOOKUP[var]
            fs = gcsfs.GCSFileSystem(token="anon")
            store = fs.get_mapper(
                "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
            )
            df_era5 = xr.open_zarr(store=store, consolidated=True).sel(
                {
                    "time": times,
                }
            )
            lats_int = np.intersect1d(df_ai[0].lat.values, df_era5.latitude.values)
            lons_int = np.intersect1d(
                df_ai[0].lon.values, df_era5.longitude.values - 180
            )

            df_era5 = df_era5.sel(
                {
                    "latitude": lats_int,
                    "longitude": lons_int,
                }
            )

        else:
            df_era5 = get_era5()

            lats_int = np.intersect1d(df_ai[0].lat.values, df_era5.lat.values)
            lons_int = np.intersect1d(df_ai[0].lon.values, df_era5.lon.values)

            print(lats_int.shape, lons_int.shape)

            df_era5 = df_era5.sel(
                {
                    "lat": lats_int,
                    "lon": lons_int,
                }
            )
        regridder = regrid(lons_reg, lats_reg, df_era5)
        df_era5_regridded = regridder(df_era5[var_era5], keep_attrs=True)

        # values_IMERG = Parallel(n_jobs=3)(delayed(get_mean_over_day)(time, idx_x, idx_y) for time in df_era5_regridded.time.values)
        values_IMERG = [
            get_mean_over_day(time, idx_x, idx_y)
            for time in tqdm(df_era5_regridded.time.values)
        ]
        values_IMERG = np.stack((values_IMERG))

        df_IMERG_era5 = xr.DataArray(
            data=values_IMERG,
            dims=["time", "lat", "lon"],
            coords={
                "lat": (["lat"], lats_reg, {"units": "degrees_north"}),
                "lon": (["lon"], lons_reg, {"units": "degrees_east"}),
                "time": df_era5_regridded.time.values,
            },
        )
        return df_ai_regridded, df_era5_regridded, df_IMERG, df_IMERG_era5

    else:
        return df_ai_regridded, df_IMERG


def postprocess(df, df_ref):
    """

    Simple mean variance post-processing for AI precip data

    """

    func = lambda x, y: np.repeat(
        np.nanstd(np.log(1 + y), axis=0)[np.newaxis, :, :], x.shape[0], axis=0
    ) * (
        x - np.repeat(np.nanmedian(x, axis=0)[np.newaxis, :, :], x.shape[0], axis=0)
    ) / np.repeat(
        np.nanstd(np.log(1 + x), axis=0)[np.newaxis, :, :], x.shape[0], axis=0
    ) + np.repeat(
        np.nanmedian(y, axis=0)[np.newaxis, :, :], x.shape[0], axis=0
    )

    postprocessed_ai = []

    for model in df.model.values:
        postprocessed_ai.append(
            xr.apply_ufunc(
                func,
                df.sel({"model": model}).load(),
                df_ref.load(),
            ).expand_dims(dim={"model": [model]}, axis=0)
        )

    postprocessed_ai = xr.concat(postprocessed_ai, "model")

    return postprocessed_ai
