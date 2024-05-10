## Verification functions for AI model intercomparison

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pandas as pd

from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

from regridding import regrid


def explained_variance_xarray(df, df_ref, times=None, log=True):
    """
    Calculate Explained Variance xarray of predicted values

    Input
    -----

    df: xr.DataArray (time, lat, lon,)
        predicted values

    df_reg: xr.DataArray (time, lat, lon,)
        reference truth values

    times: ndarray
           times for which to calculate


    Returns
    -------

    xr.DataArray of Explained Vairance

    Note
    -----

    Assumes df is in m and multiplies by 1000 to get mm

    """
    if isinstance(times, np.ndarray):
        df = df.sel({"time": times})
        df_ref = df_ref.sel({"time": times})

    func = lambda x, y: 1 - np.nanvar(x * 1000 / 6, axis=-1) / np.nanvar(y, axis=-1)

    if log:
        func = lambda x, y: 1 - np.nanvar(
            np.log(1 + x * 1000 / 6), axis=-1
        ) / np.nanvar(np.log(1 + y), axis=-1)

    mae = xr.apply_ufunc(
        func, df.load(), df_ref.load(), input_core_dims=[["time"], ["time"]]
    )

    return mae


def MAE_xarray(df, df_ref, times=None, normalised=False, log=True):
    """
    Calculate Mean Absolute Error (MAE) xarray of predicted values

    Input
    -----

    df: xr.DataArray (time, lat, lon,)
        predicted values

    df_reg: xr.DataArray (time, lat, lon,)
        reference truth values

    times: ndarray
           times for which to calculate


    Returns
    -------

    xr.DataArray of MAE

    Note
    -----

    Assumes df is in m and multiplies by 1000 to get mm

    """
    if isinstance(times, np.ndarray):
        df = df.sel({"time": times})
        df_ref = df_ref.sel({"time": times})

    func = lambda x, y: np.nanmean(np.abs(x[:, :, :-1] - y[:, :, 1:]), axis=-1)

    if normalised and log:
        func = lambda x, y: np.nanmean(
            np.abs(np.log(1e-5 + x[:, :, :-1]) - np.log(1e-5 + y[:, :, 1:])), axis=-1
        ) / np.nanstd(np.log(1e-5 + y), axis=-1)
    elif log:
        func = lambda x, y: np.nanmean(
            np.abs(np.log(1e-5 + x[:, :, :-1]) - np.log(1e-5 + y[:, :, 1:])), axis=-1
        )

    elif normalised:
        func = lambda x, y: np.nanmean(
            np.abs(x[:, :, :-1] - y[:, :, 1:]), axis=-1
        ) / np.nanstd(y, axis=-1)

    mae = xr.apply_ufunc(
        func, df.load(), df_ref.load(), input_core_dims=[["time"], ["time"]]
    )

    return mae


def QQ_plot_elev(df, df_ref, elev, level=1500, times=None, log=False):
    """

    Categorises region into above a given level (default set to approx that of Nairobi (1500 m)

    Input
    -----

    df: xr.DataArray (time, lat, lon,)
        predicted values

    df_reg: xr.DataArray (time, lat, lon,)
        reference truth values

    elev: xr.DataArray (lat, lon)
        Elevation in m

    level: int
        Elevation level above which to get values

    time: Optional ndarray
        Times to calculate QQ plot over

    log: Boolean
        Whether to take log(1+prec), default is False

    Returns
    -------

    Quantile-Quantile plot

    Note
    -----

    Assumes df is in m and multiplies by 1000 to get mm

    """
    if isinstance(times, np.ndarray):
        df = df.sel({"time": times})
        df_ref = df_ref.sel({"time": times})

    regridder = regrid(df.lon.values, df.lat.values, elev)
    elev = regridder(elev, keep_attrs=True)

    elev = xr.DataArray(
        data=np.squeeze(elev.values),
        dims=["lat", "lon"],
        coords={
            "lat": (["lat"], df.lat.values, {"units": "degrees_north"}),
            "lon": (["lon"], df.lon.values, {"units": "degrees_east"}),
        },
    )

    df["elev"] = elev.rename("elev")

    elev = xr.DataArray(
        data=np.squeeze(elev.values),
        dims=["latitude", "longitude"],
        coords={
            "latitude": (["latitude"], df.lat.values, {"units": "degrees_north"}),
            "longitude": (["longitude"], df.lon.values, {"units": "degrees_east"}),
        },
    )
    df_ref["elev"] = elev.rename("elev")

    if log:
        residuals = np.log(
            1e-5
            + df_ref.stack(x=("latitude", "longitude"))
            .query(x="elev>%s" % level)
            .values[1:]
        ) - np.log(
            1e-5 + df.stack(x=("lat", "lon")).query(x="elev>%s" % level).values[:-1]
        )
    else:
        residuals = df_ref.stack(x=("latitude", "longitude")).query(
            x="elev>%s" % level
        ).values[1:] - (
            df.stack(x=("lat", "lon")).query(x="elev>%s" % level).values[:-1]
        )
    residuals[np.abs(residuals) >= 80] = 0
    # residuals = StandardScaler().fit_transform(residuals.reshape(-1, 1))
    residuals = residuals[~pd.isnull(residuals)]

    v = np.array([-0.5, 0.5, 1.5])
    cmap, norm = from_levels_and_colors(
        v, colors=["white", "#e41a1c"], extend="neither"
    )

    figure = plt.figure(figsize=(12, 8))

    emp_res = np.sort(residuals.flatten())
    normal_res = np.sort(np.random.normal(size=len(emp_res)))

    lb = np.min([emp_res.min(), normal_res.min()])
    ub = np.max([emp_res.max(), normal_res.max()])

    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"mathtext.default": "regular"})

    fs_title = 16

    grid = plt.GridSpec(5, 3, wspace=0.2, hspace=1.5)  # create a grid for the subplots

    ax_qq = plt.subplot(grid[:3, :])

    ax_qq.plot(np.linspace(lb, ub + 1, 20), np.linspace(lb, ub + 1, 20), "k")
    ax_qq.scatter(
        normal_res, emp_res, c="palegoldenrod", edgecolor="goldenrod", alpha=0.5
    )
    ax_qq.set_title("QQ plot at elevation greater than " + str(level) + "m")
    ax_qq.set_xlim(normal_res.min() - 1, normal_res.max() + 1)
    ax_qq.set_ylim(emp_res.min() - 1, emp_res.max() + 1)

    reg_grid = np.zeros([len(df.lat.values), len(df.lon.values)])
    elev_mask = elev.values > level

    reg_grid[elev_mask] = 1

    ax_sub = plt.subplot(grid[3:, 2:], projection=ccrs.Robinson(central_longitude=0))

    ax_sub.add_feature(cfeature.BORDERS, linewidth=1)
    ax_sub.add_feature(cfeature.COASTLINE, linewidth=1)

    mesh_reg = ax_sub.pcolormesh(
        df.lon.values,
        df.lat.values,
        reg_grid,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        transform=ccrs.PlateCarree(),
    )


def Truth_vs_pred_plot_elev(df, df_ref, elev, level=1500, times=None, log=False):
    """

    Categorises region into above a given level (default set to approx that of Nairobi (1500 m)

    Input
    -----

    df: xr.DataArray (time, lat, lon,)
        predicted values

    df_reg: xr.DataArray (time, lat, lon,)
        reference truth values

    elev: xr.DataArray (lat, lon)
        Elevation in m

    level: int
        Elevation level above which to get values

    time: Optional ndarray
        Times to calculate QQ plot over

    log: Boolean
        Whether to take log(1+prec), default is False

    Returns
    -------

    Truth vs Pred plot

    Note
    -----

    Assumes df is in m and multiplies by 1000 to get mm

    """
    if isinstance(times, np.ndarray):
        df = df.sel({"time": times})
        df_ref = df_ref.sel({"time": times})

    regridder = regrid(df.lon.values, df.lat.values, elev)
    elev = regridder(elev, keep_attrs=True)

    elev = xr.DataArray(
        data=np.squeeze(elev.values),
        dims=["lat", "lon"],
        coords={
            "lat": (["lat"], df.lat.values, {"units": "degrees_north"}),
            "lon": (["lon"], df.lon.values, {"units": "degrees_east"}),
        },
    )

    df["elev"] = elev.rename("elev")
    df_ref["elev"] = elev.rename("elev")

    if log:
        truth = np.log(
            1 + df_ref.stack(x=("lat", "lon")).query(x="elev>%s" % level).values
        )
        pred = np.log(
            1 + df.stack(x=("lat", "lon")).query(x="elev>%s" % level).values 
        )

    else:
        truth = df_ref.stack(x=("lat", "lon")).query(x="elev>%s" % level).values
        pred = df.stack(x=("lat", "lon")).query(x="elev>%s" % level).values 

    nan_values = np.isnan(truth) + np.isnan(pred) + np.isnan(truth) + np.isnan(pred)

    truth = truth[~nan_values]
    pred = pred[~nan_values]
    pred[pred > truth.max()] = truth.max()
    print(truth.max())
    v = np.array([-0.5, 0.5, 1.5])
    cmap, norm = from_levels_and_colors(
        v, colors=["white", "#e41a1c"], extend="neither"
    )

    figure = plt.figure(figsize=(12, 8))

    lb = np.min([truth.min(), pred.min()])
    ub = np.max([truth.max(), pred.max()])

    fig = plt.figure(figsize=(8, 5))
    plt.rcParams.update({"font.size": 10})
    plt.rcParams.update({"mathtext.default": "regular"})

    fs_title = 16

    grid = plt.GridSpec(5, 3, wspace=0.2, hspace=1.5)  # create a grid for the subplots

    ax_qq = plt.subplot(grid[:3, :])

    ax_qq.plot(np.linspace(lb, ub + 1, 20), np.linspace(lb, ub + 1, 20), "k")
    ax_qq.scatter(
            truth[1:], pred[:-1], c="mediumturquoise", edgecolor="lightseagreen", alpha=0.5
    )
    ax_qq.set_title(
        "Prediction vs truth values at elevation greater than " + str(level) + "m"
    )
    # ax_qq.set_xlim(truth.min()-1,truth.max()+1)
    # ax_qq.set_ylim(pred.min()-1,pred.max()+1)

    reg_grid = np.zeros([len(df.lat.values), len(df.lon.values)])
    elev_mask = elev.values > level

    reg_grid[elev_mask] = 1

    ax_sub = plt.subplot(grid[3:, 2:], projection=ccrs.Robinson(central_longitude=0))

    ax_sub.add_feature(cfeature.BORDERS, linewidth=1)
    ax_sub.add_feature(cfeature.COASTLINE, linewidth=1)

    mesh_reg = ax_sub.pcolormesh(
        df.lon.values,
        df.lat.values,
        reg_grid,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        transform=ccrs.PlateCarree(),
    )


def compute_centred_coord_array(M, N):
    """
    Compute a 2D coordinate array, where the origin is at the center.

    Parameters
    ----------
    M : int
      The height of the array.
    N : int
      The width of the array.

    Returns
    -------
    out : ndarray
      The coordinate array.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def rapsd(
    fields, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs
):
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.

    Parameters
    ----------
    field: array_like
        A 2d array of shape (m, n) containing the input field.
    fft_method: object
        A module or object implementing the same methods as numpy.fft and
        scipy.fftpack. If set to None, field is assumed to represent the
        shifted discrete Fourier transform of the input field, where the
        origin is at the center of the array
        (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.

    Returns
    -------
    out: ndarray
      One-dimensional array containing the RAPSD. The length of the array is
      int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
      One-dimensional array containing the Fourier frequencies.

    References
    ----------
    :cite:`RC2011`
    """

    if len(fields.shape) != 3:
        raise ValueError(
            f"{len(fields.shape)} dimensions are found, but the number "
            "of dimensions should be 3"
        )

    if np.sum(np.isnan(fields)) > 0:
        # raise ValueError("input field should not contain nans")
        fields[np.isnan(fields)] = 0
        fields[np.isinf(fields)] = 0

    fields = fields

    results = []
    for field in fields:
        m, n = field.shape

        yc, xc = compute_centred_coord_array(m, n)
        r_grid = np.sqrt(xc * xc + yc * yc).round()
        l = max(field.shape[0], field.shape[1])

        if l % 2 == 1:
            r_range = np.arange(0, int(l / 2) + 1)
        else:
            r_range = np.arange(0, int(l / 2))

        if fft_method is not None:
            psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
            psd = np.abs(psd) ** 2 / psd.size
        else:
            psd = field

        result = []
        for r in r_range:
            mask = r_grid == r
            psd_vals = psd[mask]
            result.append(np.mean(psd_vals))

        results.append(np.array(result))

    results = np.mean(np.stack((results)), axis=0)

    if normalize:
        results /= np.sum(results)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return results, freq
    else:
        return results


from taylor_diagram import TaylorDiagram


def plot_taylor(pred, obs, elev, marker="o", empty=False, fig=None):
    refstd = np.nanstd(obs)
    print(refstd)

    samples = np.array(
        [
            [
                pred[:-1, elev > lev].std(ddof=1),
                np.corrcoef(obs[1:, elev > lev], pred[:-1, elev > lev])[0, 1],
            ]
            for lev in [0, 1000, 2000, 3000]
        ]
    )

    print(samples)

    if fig != None:
        fig = plt.figure(figsize=(10, 4))

    # Taylor diagram
    dia = TaylorDiagram(
        refstd,
        fig=fig,
        rect=122,
        label="Reference",
        srange=(0, (samples.max() / refstd)),
    )

    colors = plt.matplotlib.cm.viridis(np.linspace(0, 1, len(samples)))

    # Add the models to Taylor diagram
    for i, (stddev, corrcoef) in enumerate(samples):
        mfc = colors[i]

        if empty:
            mfc = "white"

        dia.add_sample(
            stddev,
            corrcoef,
            marker=marker,
            ms=10,
            ls="",
            mfc=mfc,
            mec=colors[i],
        )

    # Add grid
    dia.add_grid()

    # Add RMS contours, and label them
    contours = dia.add_contours(colors="0.5")
    plt.clabel(contours, inline=1, fontsize=10, fmt="%.2f")

    return
