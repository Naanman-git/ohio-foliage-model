"""Evaluate pre-fall rainfall as a candidate threshold adjustment."""
import numpy as np
import pandas as pd

from ohio_foliage_backtest import (DAYS, OUTPUT, ROOT, SITES, TEST_YEAR,
                                   calibration, draws, load_labels, load_weather)
from ohio_foliage_extensions import summer_threshold_multiplier


def rainfall():
    daily = pd.read_csv(ROOT / "data/noaa_daily_precipitation.csv", parse_dates=["DATE"])
    assert not daily.duplicated(["STATION", "DATE"]).any()
    quality = daily.PRCP_ATTRIBUTES.fillna("").str.split(",").str[1].fillna("")
    daily.loc[quality.ne("") | daily.PRCP.lt(0), "PRCP"] = np.nan
    daily["year"] = daily.DATE.dt.year
    daily = daily.loc[daily.DATE.dt.month.between(6, 8)]
    summary = daily.groupby(["STATION", "year"]).PRCP.agg(
        valid_days="count", mean_daily_mm="mean").reset_index()
    sites = {code: name for name, (_, code, *_) in SITES.items()}
    summary["site_name"] = summary.STATION.map(sites)
    summary["estimated_summer_mm"] = 92 * summary.mean_daily_mm
    return summary


def main():
    weather = load_weather()
    labels = load_labels()
    totals = rainfall()
    totals.to_csv(OUTPUT / "summer_precipitation_station_totals.csv", index=False)
    good = totals.loc[totals.valid_days.ge(88)].copy()
    # Scale to 100 mm units so the regularized fit uses comparable numerical units.
    good["summer_mean_c"] = good.estimated_summer_mm / 100
    observed = good.set_index(["site_name", "year"]).summer_mean_c
    cases = pd.read_csv(OUTPUT / "backtest_and_2026_predictions.csv")
    records = []
    for row in cases.loc[cases["mode"].eq("Preseason")].itertuples():
        key = row.site, row.year
        earlier = labels.loc[labels.year.lt(row.year)]
        train = earlier.loc[[(r.site_name, r.year) in weather for r in earlier.itertuples()]]
        train = train.merge(good[["site_name", "year", "summer_mean_c"]],
                            on=["site_name", "year"], validate="one_to_one")
        if key not in observed.index or train.site_name.eq(row.site).sum() < 2:
            continue
        trained = calibration(train, weather)
        multiplier, slope, anomaly = summer_threshold_multiplier(trained, row.site, observed.loc[key])
        histories = np.array([values for (site, year), values in weather.items()
                              if site == row.site and year < row.year])
        base, _, _ = draws(trained, row.site, histories)
        candidate, _, _ = draws(trained, row.site, histories, threshold_multiplier=multiplier)
        base_day, candidate_day = [np.quantile(x, .5, method="inverted_cdf")
                                   for x in (base, candidate)]
        records.append(dict(site=row.site, year=row.year, anomaly_mm=100*anomaly,
                            slope_per_100_mm=slope, paired_base_day=base_day, rain_day=candidate_day,
                            observed_day=row.observed_day,
                            paired_base_error=abs(base_day-row.observed_day) if base_day < DAYS else np.nan,
                            rain_error=abs(candidate_day-row.observed_day) if candidate_day < DAYS else np.nan))
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / "summer_precipitation_backtest.csv", index=False)
    historical = result.loc[result.year.lt(TEST_YEAR)].dropna(subset=["paired_base_error", "rain_error"])
    print(historical.to_string(index=False))
    print(historical[["paired_base_error", "rain_error"]].mean())


if __name__ == "__main__":
    main()
