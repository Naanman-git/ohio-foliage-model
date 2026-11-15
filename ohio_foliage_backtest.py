"""Backtest autumn canopy-color dates using observed Ohio trees and NOAA Tmin."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.titlesize": 15,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_results"
TEST_YEAR = 2026
WEATHER_CUTOFF = pd.Timestamp("2026-11-10")
N_DRAWS = 5000
SEED = 42
BASE_TEMP_C = 15.0
DAYS = 91
SITES = {
    "Columbus": ([859], "USW00004804", 40.059612, 258),
    "Perry County": ([35743], "USC00335857", 39.700165, 290),
    "Oxford": ([34167, 57318], "USW00053855", 39.4851, 268),
}
SITE_COLORS = {"Columbus": "#006b75", "Perry County": "#936026", "Oxford": "#814781"}
MODE_COLORS = {"Hindcast": "#006b75", "Preseason": "#936026"}


def season_start(year):
    return pd.Timestamp(year=int(year), month=9, day=1)


def datestr(day, year):
    if day >= DAYS:
        return "After Nov 30 / no crossing"
    return (season_start(year) + pd.Timedelta(days=int(np.round(day)))).strftime("%Y-%m-%d")


def load_labels():
    trees = pd.read_csv(ROOT / "data/npn_ohio_tree_crossings.csv",
                        parse_dates=["last_below", "first_at_or_above"])
    names = {site_id: name for name, (ids, *_) in SITES.items() for site_id in ids}
    trees["site_name"] = trees.site.map(names)
    trees = trees.dropna(subset=["site_name"]).copy()
    for field in ("last_below", "first_at_or_above"):
        trees[field + "_day"] = (trees[field] - pd.to_datetime(
            trees.year.astype(str) + "-09-01")).dt.days
    grouped = trees.groupby(["site_name", "year"], as_index=False).agg(
        lo=("last_below_day", "median"), hi=("first_at_or_above_day", "median"),
        trees=("tree", "nunique"),
    )
    grouped["observed"] = (grouped.lo + grouped.hi) / 2
    return grouped


def load_weather(lapse_c_per_m=-0.0065):
    fields = ["STATION", "DATE", "ELEVATION", "TMIN"]
    stations = pd.concat([
        pd.read_csv(ROOT / file, usecols=fields, parse_dates=["DATE"])
        for file in ("ohio_noaa_fall_daily.csv", "data/nearby_noaa_tmin.csv")
    ])
    result = {}
    for name, (_, station_id, _, site_elevation) in SITES.items():
        station = stations[stations.STATION.eq(station_id)].set_index("DATE").sort_index()
        elevation = station.ELEVATION.iloc[0]
        for year in range(2018, TEST_YEAR + 1):
            idx = pd.date_range(season_start(year), periods=DAYS)
            values = station.TMIN.reindex(idx).astype(float)
            if year == TEST_YEAR:
                values = values.where(idx <= WEATHER_CUTOFF)
            values = values.interpolate(limit=2, limit_area="inside")
            finite = values.notna().to_numpy()
            if year != TEST_YEAR and not finite.all():
                continue
            if year == TEST_YEAR and finite.sum() < 60:
                continue
            result[name, year] = values.to_numpy() + lapse_c_per_m * (site_elevation - elevation)
    return result


def stress(temperatures, latitude):
    lat = np.radians(latitude)
    solar_days = np.r_[172, np.arange(244, 244 + DAYS)]
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + solar_days) / 365)
    hours = 24 / np.pi * np.arccos(np.clip(-np.tan(lat) * np.tan(declination), -1, 1))
    shortening = (hours[0] - hours[1:]) / hours[0]
    return np.cumsum(np.maximum(0, BASE_TEMP_C - temperatures) * shortening, axis=-1)


def crossing(cumulative, thresholds):
    crossed = cumulative >= thresholds[..., None]
    return np.where(crossed.any(axis=-1), crossed.argmax(axis=-1), DAYS)


def calibration(labels, weather):
    train = labels.copy()
    for row in train.itertuples():
        cumulative = stress(weather[row.site_name, row.year], SITES[row.site_name][2])
        train.loc[row.Index, ["stress_lo", "stress_hi"]] = np.interp(
            [row.lo, row.hi], np.arange(DAYS), cumulative)
    train["stress_mid"] = (train.stress_lo + train.stress_hi) / 2
    train["center"] = train.groupby("site_name").stress_mid.transform("median")
    return train


def draws(training, name, histories, actual=None, seed=SEED, temperature_shift_c=0.0,
          threshold_multiplier=1.0):
    rng = np.random.default_rng(seed)
    local = training[training.site_name.eq(name)]
    threshold_center = local.stress_mid.median()
    sampled = training.iloc[rng.integers(len(training), size=N_DRAWS)]
    drawn_stress = rng.uniform(sampled.stress_lo.to_numpy(), sampled.stress_hi.to_numpy())
    thresholds = np.maximum(0.01, threshold_multiplier *
                            (threshold_center + drawn_stress - sampled.center.to_numpy()))
    weather = histories[rng.integers(len(histories), size=N_DRAWS)]
    if actual is not None:
        weather = np.where(np.isfinite(actual), actual, weather)
    weather = weather + temperature_shift_c
    latitude = SITES[name][2]
    simulations = crossing(stress(weather, latitude), thresholds)
    warmer = crossing(stress(weather + 1, latitude), thresholds)
    assert simulations.shape == (N_DRAWS,) and np.all((0 <= simulations) & (simulations <= DAYS))
    assert np.all(warmer >= simulations), "Warming advanced a crossing for matched draws."
    weights = np.bincount(simulations, minlength=DAYS + 1) / N_DRAWS
    assert np.isclose(weights.sum(), 1)
    return simulations, weights, warmer


def evaluate(labels, weather):
    results, probabilities = [], []
    labels = labels.loc[[row.year == TEST_YEAR or (row.site_name, row.year) in weather
                         for row in labels.itertuples()]].copy()
    targets = {(row.site_name, row.year): row for row in labels.itertuples()}
    targets.update({(name, TEST_YEAR): targets.get((name, TEST_YEAR)) for name in SITES})
    for (name, year), observed in sorted(targets.items(), key=lambda item: item[0][::-1]):
        prior = labels[labels.year.lt(year)]
        local = prior[prior.site_name.eq(name)]
        histories = np.array([values for (site, y), values in weather.items()
                              if site == name and y < year])
        if len(local) < 2 or len(histories) < 2:
            continue
        training = calibration(prior, weather)
        for mode in ("Hindcast", "Preseason"):
            actual = weather.get((name, year)) if mode == "Hindcast" else None
            if mode == "Hindcast" and actual is None:
                continue
            sims, prob, warmer = draws(training, name, histories, actual)
            if year == TEST_YEAR:
                repeated, _, _ = draws(training, name, histories, actual)
                assert np.array_equal(sims, repeated), "Fixed-seed results changed."
            p10, median, p90 = np.quantile(sims, [.1, .5, .9], method="inverted_cdf")
            stats = dict(site=name, year=year, mode=mode, training_site_years=len(local),
                         weather_years=len(histories), actual_weather_days=int(np.isfinite(actual).sum())
                         if actual is not None else 0, p10=p10, median=median, p90=p90,
                         forecast_date=datestr(median, year), p10_date=datestr(p10, year),
                         p90_date=datestr(p90, year), baseline_day=local.observed.median(),
                         no_crossing_probability=prob[DAYS],
                         warming_median_shift=np.median(warmer) - np.median(sims))
            if observed is not None:
                baseline = stats["baseline_day"]
                stats.update(observed_day=observed.observed, observed_lower=observed.lo,
                             observed_upper=observed.hi, trees=observed.trees,
                             error_days=median - observed.observed if median < DAYS else np.nan,
                             baseline_error_days=baseline - observed.observed,
                             median_in_observed_window=observed.lo <= median <= observed.hi,
                             interval_overlaps=p10 <= observed.hi and p90 >= observed.lo)
            results.append(stats)
            if year == TEST_YEAR:
                probabilities.extend(dict(site=name, mode=mode, day=day, probability=p)
                                     for day, p in enumerate(prob))
    return pd.DataFrame(results), pd.DataFrame(probabilities)


def save_figure(fig, name):
    fig.savefig(OUTPUT / (name + ".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def daily_forecast(probabilities):
    """Export daily percentages and show the highest-probability week at each site."""
    table = probabilities.copy()
    table["date"] = (season_start(TEST_YEAR) + pd.to_timedelta(table.day, unit="D")).where(
        table.day.lt(DAYS)
    ).dt.strftime("%Y-%m-%d")
    table["outcome"] = np.where(table.day.lt(DAYS), "Threshold reached", "No crossing by Nov 30")
    table["chance_percent"] = (100 * table.probability).round(2)
    table[["site", "mode", "date", "outcome", "chance_percent"]].to_csv(
        OUTPUT / f"{TEST_YEAR}_daily_forecast_percent.csv", index=False,
    )

    preseason = table[table["mode"].eq("Preseason") & table.day.lt(DAYS)]
    groups = []
    for name in SITES:
        daily = preseason[preseason.site.eq(name)].sort_values("day").reset_index(drop=True)
        end = daily.probability.rolling(7).sum().idxmax()
        groups.append((name, daily.iloc[end - 6:end + 1]))
    max_percent = max(10, 5 * np.ceil(preseason.chance_percent.max() / 5))
    fig, axes = plt.subplots(1, len(groups), figsize=(10, 4.4), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, top=.77, bottom=.24, wspace=.59)
    fig.suptitle(f"{TEST_YEAR} preseason peak foliage forecast", y=.97)
    fig.text(.5, .87, "Daily chance of first reaching 50% canopy color", ha="center")

    for ax, (name, week) in zip(axes, groups):
        dates = [f"{date:%a %b} {date.day}" for date in pd.to_datetime(week.date)]
        bars = ax.barh(dates, week.chance_percent, color=SITE_COLORS[name], height=.59)
        ax.bar_label(bars, labels=[f"{p:.1f}%" for p in week.chance_percent], padding=3, fontsize=9)
        ax.invert_yaxis()
        ax.set(xlim=(0, max_percent), xticks=[0, max_percent / 2, max_percent],
               title=name, xlabel="Daily probability")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}%")
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.spines[["top", "right"]].set_visible(False)
        chance = 100 * week.probability.sum()
        ax.text(.5, -.26, f"{chance:.1f}% chance over seven days",
                transform=ax.transAxes, ha="center", fontsize=9)
    fig.text(.5, .025, "Each panel shows the seven consecutive days with the highest combined probability.",
             ha="center", fontsize=9)
    save_figure(fig, "06_daily_forecast_percent")


def charts(results, probabilities):
    historical = results[results.year.lt(TEST_YEAR)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, layout="constrained")
    limits = [30, 78]
    for ax, (mode, records) in zip(axes, historical.groupby("mode")):
        ax.plot(limits, limits, color="0.55", ls="--", lw=1)
        for site, rows in records.groupby("site"):
            ax.errorbar(rows.observed_day, rows["median"],
                        xerr=np.vstack([rows.observed_day - rows.observed_lower,
                                        rows.observed_upper - rows.observed_day]),
                        fmt="o", color=SITE_COLORS[site], capsize=2, label=site, alpha=.85)
        ax.set(title=mode, xlabel="Observed window midpoint (days after Sep 1)",
               xlim=limits, ylim=limits)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Predicted median (days after Sep 1)")
    fig.suptitle("Historical backtest using prior years only")
    save_figure(fig, "01_historical_backtest")

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
    start = season_start(TEST_YEAR)
    current = results[results.year.eq(TEST_YEAR)]
    for ax, name in zip(axes, SITES):
        local = current[current.site.eq(name)]
        for mode, values in probabilities[probabilities.site.eq(name)].groupby("mode"):
            daily = values[values.day.lt(DAYS)]
            ax.plot(start + pd.to_timedelta(daily.day, unit="D"), daily.probability,
                    color=MODE_COLORS[mode], label=mode)
        measured = local.dropna(subset=["observed_lower"])
        if not measured.empty:
            row = measured.iloc[0]
            ax.axvspan(start + pd.Timedelta(days=row.observed_lower),
                       start + pd.Timedelta(days=row.observed_upper),
                       color="#8f9d62", alpha=.25, label="Observed tree window")
        ax.set(title=name, ylabel="Daily probability")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[-1].set_xlim(season_start(TEST_YEAR), pd.Timestamp("2026-11-30"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("2026 Monte Carlo peak dates and observed windows")
    save_figure(fig, "02_2026_peak_probabilities")

    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    benchmark = historical.groupby("mode")[["error_days", "baseline_error_days"]].agg(
        lambda x: x.abs().mean())
    benchmark.columns = ["Foliage model", "Historical median date"]
    benchmark.plot(kind="bar", ax=ax, rot=0, color=["#006b75", "#cf9e63"])
    ax.set(title="Mean absolute error on historical holdouts", xlabel="", ylabel="Days")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, loc="upper center", ncols=2)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3)
    save_figure(fig, "03_model_vs_baseline")


def main():
    OUTPUT.mkdir(exist_ok=True)
    labels, weather = load_labels(), load_weather()
    predictions, probabilities = evaluate(labels, weather)
    labels.to_csv(OUTPUT / "observed_site_windows.csv", index=False)
    predictions.to_csv(OUTPUT / "backtest_and_2026_predictions.csv", index=False)
    probabilities.to_csv(OUTPUT / "2026_daily_probabilities.csv", index=False)
    scored = predictions[predictions.year.lt(TEST_YEAR)].copy()
    for field in ("median_in_observed_window", "interval_overlaps"):
        scored[field] = scored[field].astype(float)
    summary = scored.groupby("mode").agg(
        cases=("year", "size"), scored_dates=("error_days", "count"),
        mae_days=("error_days", lambda x: x.abs().mean()),
        rmse_days=("error_days", lambda x: np.sqrt(x.pow(2).mean())),
        baseline_mae_days=("baseline_error_days", lambda x: x.abs().mean()),
        median_in_window=("median_in_observed_window", "mean"),
        interval_overlaps_window=("interval_overlaps", "mean"),
    )
    summary.to_csv(OUTPUT / "historical_metrics.csv")
    charts(predictions, probabilities)
    daily_forecast(probabilities)
    print(summary.round(3).to_string())
    print(predictions.loc[predictions.year.eq(TEST_YEAR),
          ["site", "mode", "forecast_date", "p10_date", "p90_date", "error_days"]].to_string(index=False))
    print(f"Monte Carlo checks passed for {N_DRAWS:,} draws per case. Results in {OUTPUT}")


if __name__ == "__main__":
    main()
