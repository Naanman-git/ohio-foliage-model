"""Reproduce elevation, holdout robustness, and decade scenario checks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, t

from ohio_foliage_backtest import (
    DAYS, OUTPUT, ROOT, SITE_COLORS, SITES, TEST_YEAR, calibration, draws, evaluate,
    load_labels, load_weather, save_figure, season_start,
)

CASE_KEY = ["site", "year", "mode"]


def elevation_check(labels, corrected_weather):
    corrected, _ = evaluate(labels, corrected_weather)
    uncorrected, _ = evaluate(labels, load_weather(lapse_c_per_m=0.0))
    columns = CASE_KEY + ["median", "error_days", "baseline_error_days"]
    held_out = corrected.loc[corrected.year.lt(TEST_YEAR), columns].merge(
        uncorrected.loc[uncorrected.year.lt(TEST_YEAR), columns],
        on=CASE_KEY, suffixes=("_corrected", "_raw"), validate="one_to_one",
    )
    if len(held_out) != corrected.year.lt(TEST_YEAR).sum() or len(held_out) != uncorrected.year.lt(TEST_YEAR).sum():
        raise ValueError("Elevation runs did not score the same historical cases.")
    held_out["prediction_shift_days"] = held_out.median_corrected - held_out.median_raw
    held_out["abs_error_change_days"] = held_out.error_days_corrected.abs() - held_out.error_days_raw.abs()
    held_out.to_csv(OUTPUT / "elevation_paired_cases.csv", index=False)

    summary = held_out.groupby("mode").agg(
        cases=("year", "size"),
        corrected_mae=("error_days_corrected", lambda x: x.abs().mean()),
        raw_mae=("error_days_raw", lambda x: x.abs().mean()),
        max_absolute_prediction_shift=("prediction_shift_days", lambda x: x.abs().max()),
    )
    summary["mae_change"] = summary.corrected_mae - summary.raw_mae
    summary.to_csv(OUTPUT / "elevation_metrics.csv")

    robustness = []
    for mode, cases in held_out.groupby("mode"):
        for omitted_year in sorted(cases.year.unique()):
            remaining = cases[cases.year.ne(omitted_year)]
            model_mae = remaining.error_days_corrected.abs().mean()
            baseline_mae = remaining.baseline_error_days_corrected.abs().mean()
            robustness.append(dict(mode=mode, omitted_year=omitted_year,
                                   cases=len(remaining), model_mae=model_mae,
                                   baseline_mae=baseline_mae,
                                   mae_vs_baseline=model_mae - baseline_mae))
    robustness = pd.DataFrame(robustness)
    robustness.to_csv(OUTPUT / "omit_one_year_score_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].axvline(0, color="0.6", linewidth=1)
    modes = list(summary.index)
    axes[0].scatter(summary.mae_change, modes, color="#006b75", s=60, zorder=3)
    axes[0].set(xlim=(-max(.25, 1.5 * summary.mae_change.abs().max()),
                     max(.25, 1.5 * summary.mae_change.abs().max())),
                title="Effect of adjusting for elevation",
                xlabel="MAE with correction minus MAE without (days)")
    for i, (mode, records) in enumerate(robustness.groupby("mode")):
        axes[1].scatter(records.mae_vs_baseline, np.full(len(records), i), s=45,
                        color="#814781", alpha=.7)
    axes[1].axvline(0, color="0.6", linewidth=1)
    axes[1].set(yticks=range(len(modes)), yticklabels=modes,
                title="Sensitivity to omitting one year from the score",
                xlabel="Model MAE minus historical-median MAE (days)")
    fig.suptitle("Historical accuracy checks")
    save_figure(fig, "04_accuracy_checks")
    return summary, robustness


def accuracy_targets(labels, weather):
    predictions, daily = evaluate(labels, weather)
    history = predictions[predictions.year.lt(TEST_YEAR)].copy()
    history["model_within_7_days"] = history.error_days.abs().le(7)
    history["baseline_within_7_days"] = history.baseline_error_days.abs().le(7)
    targets = history.groupby("mode").agg(
        cases=("year", "size"),
        model_within_7_days=("model_within_7_days", "sum"),
        baseline_within_7_days=("baseline_within_7_days", "sum"),
    )
    targets.to_csv(OUTPUT / "historical_seven_day_accuracy.csv")

    comparison = predictions.loc[predictions.year.eq(TEST_YEAR)].dropna(
        subset=["observed_lower", "observed_upper"],
    )
    rows = []
    for row in comparison.itertuples():
        matching = daily.site.eq(row.site) & daily["mode"].eq(row.mode)
        in_window = daily.day.between(row.observed_lower, row.observed_upper)
        rows.append(dict(site=row.site, mode=row.mode, trees=int(row.trees),
                         probability_in_observed_window=daily.loc[matching & in_window, "probability"].sum()))
    pd.DataFrame(rows).to_csv(OUTPUT / "2026_observed_window_probability.csv", index=False)


def station_trends():
    fields = ["STATION", "DATE", "TMIN"]
    files = ["ohio_noaa_fall_daily.csv", "data/nearby_noaa_tmin.csv",
             "data/longterm_noaa_tmin.csv"]
    records = pd.concat(pd.read_csv(ROOT / path, usecols=fields, parse_dates=["DATE"])
                        for path in files)
    rows = []
    for site, (_, station, _, _) in SITES.items():
        readings = records.loc[records.STATION.eq(station)].set_index("DATE").TMIN
        for year in range(1996, TEST_YEAR):
            date_index = pd.date_range(season_start(year), periods=DAYS)
            values = readings.reindex(date_index).interpolate(limit=2, limit_area="inside")
            if values.notna().all():
                rows.append(dict(site=site, station=station, year=year,
                                 mean_fall_tmin_c=values.mean()))
    return pd.DataFrame(rows)


def decade_scenarios(labels, weather, temperatures):
    historical = labels.loc[[row.year < TEST_YEAR and (row.site_name, row.year) in weather
                             for row in labels.itertuples()]]
    train = calibration(historical, weather)
    trend_rows, scenario_rows = [], []
    for site, years in temperatures.groupby("site"):
        fit = linregress(years.year, years.mean_fall_tmin_c)
        trend = 10 * fit.slope
        margin = 10 * t.ppf(.975, len(years) - 2) * fit.stderr
        trend_rows.append(dict(site=site, years=len(years), first_year=years.year.min(),
                               last_year=years.year.max(), slope_c_per_decade=trend,
                               lower_95_c=trend - margin, upper_95_c=trend + margin))
        histories = np.array([values for (name, year), values in weather.items()
                              if name == site and year < TEST_YEAR])
        base, _, _ = draws(train, site, histories)
        for scenario, change in [("Lower slope bound", trend - margin), ("Station trend", trend),
                                 ("Upper slope bound", trend + margin), ("+1 °C sensitivity", 1)]:
            future, _, _ = draws(train, site, histories, temperature_shift_c=change)
            matched = (base < DAYS) & (future < DAYS)
            shifts = future[matched] - base[matched]
            q10, q50, q90 = np.quantile(shifts, [.1, .5, .9]) if matched.any() else [np.nan] * 3
            scenario_rows.append(dict(site=site, scenario=scenario, delta_tmin_c=change,
                                      date_shift_p10_days=q10, date_shift_median_days=q50,
                                      date_shift_p90_days=q90, matched_crossings=matched.sum(),
                                      no_crossing_probability=np.mean(future == DAYS)))
    trends = pd.DataFrame(trend_rows)
    scenarios = pd.DataFrame(scenario_rows)
    temperatures.to_csv(OUTPUT / "noaa_fall_station_means.csv", index=False)
    trends.to_csv(OUTPUT / "noaa_fall_temperature_trends.csv", index=False)
    scenarios.to_csv(OUTPUT / "decade_scenarios.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for site, years in temperatures.groupby("site"):
        fit = linregress(years.year, years.mean_fall_tmin_c)
        line_years = np.array([years.year.min(), years.year.max()])
        axes[0].scatter(years.year, years.mean_fall_tmin_c, color=SITE_COLORS[site], s=13, alpha=.55)
        axes[0].plot(line_years, fit.intercept + fit.slope * line_years, color=SITE_COLORS[site], label=site)
    axes[0].set(title="Observed NOAA fall temperature", xlabel="Year",
                ylabel="Sep–Nov mean daily minimum (°C)")
    axes[0].legend(frameon=False, fontsize=8)
    for y, site in enumerate(SITES):
        local = scenarios[scenarios.site.eq(site)].set_index("scenario")
        bounds = local.loc[["Lower slope bound", "Upper slope bound"], "date_shift_median_days"]
        center = local.loc["Station trend", "date_shift_median_days"]
        axes[1].hlines(y, bounds.min(), bounds.max(), color=SITE_COLORS[site], linewidth=3)
        axes[1].scatter(center, y, color=SITE_COLORS[site], s=55, zorder=3)
    axes[1].axvline(0, color="0.6", linewidth=1, linestyle="--")
    axes[1].set(yticks=range(len(SITES)), yticklabels=list(SITES),
                title="Sensitivity to ten years of station trend",
                xlabel="Median paired peak-date shift (days)")
    fig.suptitle("Ten-year scenarios based on historical minimum temperatures")
    save_figure(fig, "05_ten_year_scenarios")
    return trends, scenarios


def main():
    OUTPUT.mkdir(exist_ok=True)
    labels = load_labels()
    weather = load_weather()
    summary, sensitivity = elevation_check(labels, weather)
    accuracy_targets(labels, weather)
    trends, scenarios = decade_scenarios(labels, weather, station_trends())
    print(summary.round(3).to_string())
    print(sensitivity.groupby("mode").mae_vs_baseline.agg(["min", "max"]).round(3).to_string())
    print(trends.round(3).to_string(index=False))
    print(scenarios.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
