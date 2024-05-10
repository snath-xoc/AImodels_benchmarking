## class object of ar generator

import numpy as np
import time
import xarray as xr
from sklearn.covariance import LedoitWolf
from scipy.stats import (
    genpareto,
    linregress,
    norm,
    percentileofscore,
    scoreatpercentile,
)
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")


class model(object):

    """
    First order AR process with spatially correlated noise terms

    Attributes:
    ----------

    seasonal_cycle: xarray (months, gp)
                    seasonal climatological cycle

    ar_coeffs: xr.DataArray (coeff, lat, lon)
               Coefficients for AR(1) process, intercept is fitted by default

    residuals: xr.DataArray (time, gp)
               residuals with ar process analytically corrected for

    cov: (lat, lon)
         covariance matrix with regional average for jittered sampling

    gpd_params: (moment, lat, lon)
         xr.DataArray with first four moment of general pareto distribution fit

    Functions:
    ---------

    fit: Input (X: xarray (time, gp))
        fits over a field for the above attributes

    predict: Input (n_realisation, date, length)
        predicts n realisations over a given time period starting at a given date

    """

    def __init__(self):
        """
        regular initialisation

        """

    def get_seasonal_cycle(self, X):
        self.seasonal_cycle = X.groupby("time.month").mean("time")
        self.residuals = X.groupby("time.month") - self.seasonal_cycle
        self.residuals = self.residuals.compute()

        return

    def calculate_ar_coeff(self):
        def calculate_ar_coeff_gp(x):
            coeffs = np.zeros([2])

            fit_results = linregress(x[:-1], x[1:])
            coeffs[0] = fit_results.intercept
            coeffs[1] = fit_results.slope

            return coeffs

        self.ar_coeff = xr.apply_ufunc(
            calculate_ar_coeff_gp,
            self.residuals.precipitation,
            input_core_dims=[["time"]],
            output_core_dims=[["coeff"]],
            vectorize=True,
        )

        return

    def calculate_cov(self):
        mean = np.squeeze(
            self.residuals.precipitation.mean(["lat", "lon"]).fillna(0).values
        )
        lons, lats = np.meshgrid(self.residuals.lon.values, self.residuals.lat.values)

        self.cov = xr.DataArray(
            data=np.zeros_like(lons),
            dims=["lat", "lon"],
            coords={
                "lat": (["lat"], self.residuals.lat.values, {"units": "degrees_north"}),
                "lon": (["lon"], self.residuals.lon.values, {"units": "degrees_east"}),
            },
        )

        for lat, lon in zip(lats.flatten(), lons.flatten()):
            gp_vals = np.squeeze(
                self.residuals.sel({"lat": lat, "lon": lon})
                .precipitation.fillna(0)
                .values
            )
            self.cov.loc[{"lat": lat, "lon": lon}] = (
                (gp_vals * mean).sum() - gp_vals.sum() * mean.sum() / gp_vals.size
            ) / (gp_vals.size - 1)

        return

    def calculate_gpd(self):
        """
        deprecated
        """

        def calculate_gpd_gp(x):
            coeffs = np.zeros([3])

            c, mean, var = genpareto.fit(x)
            coeffs[0] = c
            coeffs[1] = mean
            coeffs[2] = var

            return coeffs

        self.gpd_coeff = xr.apply_ufunc(
            calculate_gpd_gp,
            self.residuals.precipitation,
            input_core_dims=[["time"]],
            output_core_dims=[["moment"]],
            vectorize=True,
        )
        return

    def fit(self, X):
        print("Performing fit for AR(1) coefficients and GPD parameters")
        start_time = time.time()

        self.get_seasonal_cycle(X)
        print("Extracted seasonal cycle")
        self.calculate_ar_coeff()
        self.calculate_cov()
        print(
            "Extracted AR(1) coefficients and covariance matrix and all done in ----",
            time.time() - start_time,
            "s----",
        )
        # self.calculate_gpd()

    def jitter(self, noise, cov, residuals):
        j = []

        for n in noise:
            q = percentileofscore(residuals, n)
            norm_val = norm.ppf(q / 100) + norm.rvs(0, cov)
            q = norm.cdf(norm_val) * 100

            j.append(np.percentile(residuals, q, method="weibull"))

        return np.array(j)

    def generate_noise(self, noise, residuals, date, length, parallel=False):
        lons, lats = np.meshgrid(residuals.lon.values, residuals.lat.values)

        if parallel:
            noise_temp = Parallel(n_jobs=10)(
                delayed(self.jitter)(
                    noise.sel({"lat": lat, "lon": lon}).precipitation.values,
                    1 - self.cov.sel({"lat": lat, "lon": lon}).values ** 2,
                    residuals.sel({"lat": lat, "lon": lon}).precipitation.values,
                )
                for lat, lon in zip(lats.flatten(), lons.flatten())
            )

        else:
            noise_temp = []
            idx = 0
            for lat, lon in zip(lats.flatten(), lons.flatten()):
                noise_temp.append(
                    self.jitter(
                        noise.sel({"lat": lat, "lon": lon}).precipitation.values,
                        1 - self.cov.sel({"lat": lat, "lon": lon}).values ** 2,
                        residuals.sel({"lat": lat, "lon": lon}).precipitation.values,
                    )
                )
                idx += 1

        jittered_noise = xr.DataArray(
            data=np.zeros([length + 3, lons.shape[0], lons.shape[1]]),
            dims=["sel_time", "lat", "lon"],
            coords={
                "sel_time": np.arange(
                    date - np.timedelta64(3, "D"),
                    date + np.timedelta64(length, "D"),
                    np.timedelta64(1, "D"),
                    dtype="datetime64[ns]",
                ),
                "lat": (["lat"], self.residuals.lat.values, {"units": "degrees_north"}),
                "lon": (["lon"], self.residuals.lon.values, {"units": "degrees_east"}),
            },
        )

        idx = 0
        for lat, lon in zip(lats.flatten(), lons.flatten()):
            jittered_noise.loc[{"lat": lat, "lon": lon}] = np.squeeze(noise_temp[idx])

            idx += 1

        return jittered_noise

    def predict_xr(self, n_realisations, date, length):
        def predict(x, ar_coeff, noise):
            x_buffer = np.zeros([len(x) + 3])

            i = 0
            for n, x_val in zip(noise, x_buffer[:-1]):
                x_buffer[i + 1] = ar_coeff[0] + ar_coeff[1] * x_val + n
                i += 1

            return x_buffer[3:]

        predictions = xr.DataArray(
            data=np.zeros(
                [
                    n_realisations,
                    length,
                    len(self.residuals.lat.values),
                    len(self.residuals.lon.values),
                ]
            ),
            dims=["n_realisations", "time", "lat", "lon"],
            coords={
                "lat": (["lat"], self.residuals.lat.values, {"units": "degrees_north"}),
                "lon": (["lon"], self.residuals.lon.values, {"units": "degrees_east"}),
                "time": np.arange(
                    date,
                    date + np.timedelta64(length, "D"),
                    np.timedelta64(1, "D"),
                    dtype="datetime64[ns]",
                ),
                "n_realisations": np.arange(n_realisations),
            },
        )
        month = str(date).split("T")[0].split("-")[1]
        day = str(date).split("T")[0].split("-")[2]
        times = np.hstack(
            (
                [
                    np.arange(
                        np.datetime64("%s-%s-%s" % (year, month, day))
                        - np.timedelta64(15, "D"),
                        np.datetime64("%s-%s-%s" % (year, month, day))
                        + np.timedelta64(15, "D"),
                        np.timedelta64(1, "D"),
                        dtype="datetime64[ns]",
                    )
                    for year in ["2018", "2019", "2020", "2021"]
                ]
            )
        )

        times = [time for time in times if np.isin(time, self.residuals.time.values)]
        sel_time = np.random.choice(times, size=length + 3)
        noise = self.residuals.sel({"time": sel_time})
        residuals = self.residuals.sel({"time": times})

        jittered_noise = self.generate_noise(noise, residuals, date, length)
        start_time = time.time()
        predictions = xr.apply_ufunc(
            predict,
            predictions,
            self.ar_coeff,
            jittered_noise,
            input_core_dims=[["time"], ["coeff"], ["sel_time"]],
            output_core_dims=[["time"]],
            vectorize=True,
        )

        predictions = predictions.groupby("time.month") + self.seasonal_cycle

        print(
            "Generated %i emulations of length %i in ----" % (n_realisations, length),
            time.time() - start_time,
            "s----",
        )

        return predictions
