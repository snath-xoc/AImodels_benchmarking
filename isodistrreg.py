import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import xarray as xr
import numpy as np
from tqdm import tqdm
import time


def fit_idr_elev(df_ai, df_obs, elev, times_train, times_test, lowers, uppers):
    X_all_elev_train, y_all_elev_train = prepare_data_over_elev(
        df_ai, df_obs, elev, times_train, lowers, uppers
    )

    print("Preparing data")
    X_all_elev_test, y_all_elev_test = prepare_data_over_elev(
        df_ai, df_obs, elev, times_test, lowers, uppers
    )

    preds_test = []
    idr_fit = []
    start_time = time.time()
    i_elev = 0
    for X_train, y_train in zip(X_all_elev_train, y_all_elev_train):
        print("Fitting on %i samples" % len(y_train))

        fit_results = fit_idr(X_train, y_train, X_all_elev_test[i_elev])

        idr_fit.append(fit_results[0])
        preds_test.append(fit_results[1])

        print("Completed fitting in ----", time.time() - start_time, "s---- ")

        i_elev += 1

    return idr_fit, preds_test, y_all_elev_test


def fit_idr_gp(df_ai, df_obs, times_train, times_test):
    print("Preparing data")

    preds_test = []
    idr_fit = []
    y_all_gp_test = []

    mask = xr.open_dataset("~/cGAN/constants-regICPAC/lsm.nc").lsm.values.astype(bool)

    lons, lats = np.meshgrid(df_ai.lon.values, df_ai.lat.values)
    lats = lats[mask]
    lons = lons[mask]

    start_time = time.time()
    i_gp = 0
    print("Fitting on %i grid points" % mask.sum())
    for lat, lon in tqdm(zip(lats.flatten(), lons.flatten())):
        X_train, y_train = prepare_data_over_gp(df_ai, df_obs, lat, lon, times_train)
        X_test, y_test = prepare_data_over_gp(df_ai, df_obs, lat, lon, times_test)
        fit_results = fit_idr(X_train, y_train, X_test)

        idr_fit.append(fit_results[0])
        preds_test.append(fit_results[1])
        y_all_gp_test.append(y_test)

        i_gp += 1
    print("Completed fitting in ----", time.time() - start_time, "s---- ")

    return idr_fit, preds_test, y_all_gp_test


def fit_idr(X_train, y_train, X_test):
    idr = importr("isodistrreg")

    idr_fit = idr.idr(y=y_train, X=X_train)
    # print('fitting complete')

    preds_test = ro.r.predict(idr_fit, X_test)

    return idr_fit, preds_test


def pit(preds, y_test):
    return ro.r.pit(preds, y_test)


def crps(preds, y_test):
    return ro.r.crps(preds, y_test)


def crps_over_gp(preds, y_test, elev):
    if not isinstance(elev, np.ndarray):
        elev = np.squeeze(elev.values)

    mask = xr.open_dataset("~/cGAN/constants-regICPAC/lsm.nc").lsm.values.astype(bool)

    shape_x = elev.shape[0]
    shape_y = elev.shape[1]

    crps_all = np.zeros_like(elev).flatten()
    crps_land = np.zeros(mask.sum())

    i_gp = 0
    for pred, y_t in zip(preds, y_test):
        crps_land[i_gp] = np.nanmean(np.asarray(crps(pred, y_t)))
        i_gp += 1

    crps_all[mask] = crps_land

    return crps_all.reshape(shape_x, shape_y)


def crps_over_elev(preds, elev, y_test, lowers, uppers):
    if not isinstance(elev, np.ndarray):
        elev = np.squeeze(elev.values)

    crps_all = np.zeros_like(elev)

    i_elev = 0
    for lower, upper in zip(lowers, uppers):
        idx_elev = np.argwhere(np.logical_and(elev >= int(lower), elev < int(upper)))
        crps_all[idx_elev] = np.nanmean(
            np.asarray(crps(preds[i_elev], y_test[i_elev])).reshape(-1, len(idx_elev)),
            axis=0,
        )

    return crps_all


def prepare_data_over_elev(df_ai, df_obs, elev, times, lowers, uppers):
    X = []
    y = []

    for lower, upper in zip(lowers, uppers):
        output = prepare_data(df_ai, df_obs, elev, times, lower, upper)

        y.append(output[1])
        X.append(output[0])

    return X, y


def prepare_data_over_gp(df_ai, df_obs, lat, lon, times):
    y = ro.FloatVector(
        df_obs.sel({"time": times, "lat": lat, "lon": lon}).values.flatten()
    )

    with (ro.default_converter + pandas2ri.converter).context():
        x_vals = df_ai.sel({"time": times, "lat": lat, "lon": lon}).values.flatten()
        x_vals = xr.DataArray(np.squeeze(x_vals)).rename("preds")
        X = ro.conversion.get_conversion().py2rpy(x_vals.to_dataframe())

    return X, y


def prepare_data(df_ai, df_obs, elev, times, lower=0, upper=1500):
    lower = str(lower)
    upper = str(upper)

    elev = xr.DataArray(
        data=np.squeeze(elev.values),
        dims=["lat", "lon"],
        coords={
            "lat": (["lat"], df_ai.lat.values, {"units": "degrees_north"}),
            "lon": (["lon"], df_ai.lon.values, {"units": "degrees_east"}),
        },
    )

    df_ai = df_ai.rename("preds").sel({"time": times})
    df_obs = df_obs.rename("obs").sel({"time": times})

    df_ai["elev"] = elev.rename("elev")
    df_obs["elev"] = elev.rename("elev")

    y = ro.FloatVector(
        df_obs.stack(x=("lat", "lon"))
        .query(x="elev>=%s" % lower)
        .query(x="elev<%s" % upper)
        .values.flatten()
    )

    with (ro.default_converter + pandas2ri.converter).context():
        x_vals = (
            df_ai.stack(x=("lat", "lon"))
            .query(x="elev>=%s" % lower)
            .query(x="elev<%s" % upper)
            .values.flatten()
        )
        X = xr.DataArray(np.squeeze(x_vals)).rename("preds")
        X = ro.conversion.get_conversion().py2rpy(X.to_dataframe())

    # print(X)

    return X, y
