"""Test a summer temperature predictor and fit a Perry County color curve."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

from ohio_foliage_backtest import (
    DAYS, OUTPUT, ROOT, SITES, SITE_COLORS, TEST_YEAR, calibration, draws,
    evaluate, load_labels, load_weather, save_figure, season_start,
)

PERRY_SITE = 35743
PERRY = "Perry County"
PANEL_YEARS = range(2020, 2025)
PERCENT_DAYS = np.arange(DAYS)
SUMMER_RIDGE = 4.0


def summer_temperatures():
    daily = pd.read_csv(ROOT / "data/noaa_summer_daily.csv", parse_dates=["DATE"])
    daily["year"] = daily.DATE.dt.year
    daily["daily_mean_c"] = (daily.TMIN + daily.TMAX) / 2
    summers = daily.groupby(["STATION", "year"]).daily_mean_c.agg(
        summer_mean_c="mean", valid_days="count",
    ).reset_index()
    station_sites = {station: name for name, (_, station, *_) in SITES.items()}
    summers["site_name"] = summers.STATION.map(station_sites)
    return summers[summers.valid_days.ge(88)]


def summer_threshold_multiplier(training, name, observed_summer):
    """A pooled, regularized slope after removing fixed differences between sites."""
    training = training.assign(log_threshold=np.log(training.stress_mid.clip(lower=.01)))
    centered = training[["summer_mean_c", "log_threshold"]] - training.groupby("site_name")[
        ["summer_mean_c", "log_threshold"]
    ].transform("mean")
    x = centered.summer_mean_c
    slope = x.dot(centered.log_threshold) / (x.dot(x) + SUMMER_RIDGE)
    prior_mean = training.loc[training.site_name.eq(name), "summer_mean_c"].mean()
    anomaly = observed_summer - prior_mean
    return np.exp(slope * anomaly), slope, anomaly


def summer_backtest(labels, weather):
    summers = summer_temperatures()
    cases, _ = evaluate(labels, weather)
    eligible = labels.loc[[(r.site_name, r.year) in weather for r in labels.itertuples()]]
    prior_weather_labels = eligible[eligible.year.lt(TEST_YEAR)].merge(
        summers, on=["site_name", "year"], validate="one_to_one",
    )
    summer_lookup = summers.set_index(["site_name", "year"]).summer_mean_c
    rows = []
    for row in cases.itertuples():
        train = calibration(prior_weather_labels[prior_weather_labels.year.lt(row.year)], weather)
        local = train[train.site_name.eq(row.site)]
        if len(local) < 2 or (row.site, row.year) not in summer_lookup:
            continue
        multiplier, slope, anomaly = summer_threshold_multiplier(
            train, row.site, summer_lookup.loc[row.site, row.year],
        )
        histories = np.array([values for (name, year), values in weather.items()
                              if name == row.site and year < row.year])
        actual = weather[row.site, row.year] if row.mode == "Hindcast" else None
        original, _, _ = draws(train, row.site, histories, actual)
        revised, _, _ = draws(train, row.site, histories, actual, threshold_multiplier=multiplier)
        old, new = (float(np.quantile(values, .5, method="inverted_cdf"))
                    for values in (original, revised))
        if old != row.median:
            raise ValueError("Summer comparison does not reproduce the original paired prediction.")
        rows.append(dict(site=row.site, year=row.year, mode=row.mode, summer_anomaly_c=anomaly,
                         summer_log_stress_slope=slope, observed_day=row.observed_day,
                         original_day=old, with_summer_day=new, baseline_day=row.baseline_day,
                         original_abs_error=abs(old - row.observed_day) if old < DAYS else np.nan,
                         with_summer_abs_error=abs(new - row.observed_day) if new < DAYS else np.nan,
                         baseline_abs_error=abs(row.baseline_day - row.observed_day)))
    paired = pd.DataFrame(rows)
    assert len(paired) == len(cases)
    history = paired[paired.year.lt(TEST_YEAR)]
    summary = history.groupby("mode").agg(
        cases=("year", "size"), original_mae=("original_abs_error", "mean"),
        with_summer_mae=("with_summer_abs_error", "mean"), baseline_mae=("baseline_abs_error", "mean"),
    )
    summers.to_csv(OUTPUT / "summer_station_means.csv", index=False)
    paired.to_csv(OUTPUT / "summer_temperature_cases.csv", index=False)
    summary.to_csv(OUTPUT / "summer_temperature_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    comparison = summary[["original_mae", "with_summer_mae", "baseline_mae"]]
    comparison = comparison.set_axis(["Current model", "With summer temperature", "Historical median"], axis=1)
    comparison.plot.bar(ax=ax, rot=0, color=["#006b75", "#814781", "#cf9e63"])
    ax.set(title="Matched historical test of summer temperature", ylabel="Mean absolute error (days)",
           xlabel="", ylim=(0, comparison.max().max() * 1.28))
    ax.legend(loc="upper center", ncols=3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    save_figure(fig, "07_summer_temperature_backtest")
    return summary


def load_visits(filename):
    visits = pd.read_csv(ROOT / "data" / filename, parse_dates=["date"])
    visits["year"] = visits.date.dt.year
    visits["day"] = (visits.date - pd.to_datetime(visits.year.astype(str) + "-09-01")).dt.days
    return visits


def perry_panel(color, leaf_drop):
    visit_counts = color.groupby(["site", "year"]).agg(
        recorded_visits=("tree", "size"), distinct_trees=("tree", "nunique"),
        distinct_dates=("date", "nunique"),
    ).reset_index()
    repeat = color[["site", "tree", "year"]].drop_duplicates()
    years = repeat.groupby(["site", "tree"]).year.transform("size")
    repeats = repeat.assign(repeated=years.gt(1)).groupby(["site", "year"]).repeated.sum().reset_index()
    visit_counts.merge(repeats, on=["site", "year"], validate="one_to_one").to_csv(
        OUTPUT / "expanded_foliage_coverage.csv", index=False,
    )

    perry = color[color.site.eq(PERRY_SITE) & color.year.isin(PANEL_YEARS)]
    panel_years = perry.groupby("tree").year.nunique()
    trees = panel_years[panel_years.eq(len(PANEL_YEARS))].index
    color_panel = perry[perry.tree.isin(trees)].copy()
    color_panel["mid_percent"] = (color_panel.lower_percent + color_panel.upper_percent) / 2
    daily = color_panel.groupby(["date", "year", "day"], as_index=False).agg(
        color_lower=("lower_percent", "median"), color_upper=("upper_percent", "median"),
        color_mid=("mid_percent", "median"), trees=("tree", "nunique"),
    )
    daily = daily[daily.trees.ge(7)].copy()
    drop = leaf_drop[leaf_drop.site.eq(PERRY_SITE) & leaf_drop.tree.isin(trees) &
                     leaf_drop.year.isin(PANEL_YEARS)].groupby(
        ["date", "year", "day"], as_index=False,
    ).agg(drop_fraction=("status", "mean"), trees=("tree", "nunique"))
    return daily, drop[drop.trees.ge(7)], len(trees)


def color_curve(days_from_crossing, rise_days, extra_color, decline_days):
    """Bounded curve anchored at 50%, with a rise and a later decline."""
    relative = np.asarray(days_from_crossing)
    after = np.maximum(relative, 0)
    before_peak = 100 * expit(relative / rise_days)
    late_season = (50 + extra_color * (-np.expm1(-after / rise_days))) * np.exp(-(after / decline_days)**2)
    return np.where(relative <= 0, before_peak, late_season)


def fit_curve(earlier):
    sizes = earlier.groupby("year").day.transform("size").to_numpy()
    def residual(parameters):
        fit = color_curve(earlier.day - earlier.anchor, *parameters)
        return (fit - earlier.color_mid) / np.sqrt(sizes)

    result = least_squares(residual, x0=[5, 35, 20], bounds=([.5, 0, 3], [30, 50, 70]))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def curve_backtest(labels, weather, color, leaf_drop):
    visits, drops, panel_size = perry_panel(color, leaf_drop)
    labels = labels.loc[[(r.site_name, r.year) in weather for r in labels.itertuples()]]
    anchors = labels[labels.site_name.eq(PERRY)][["year", "observed"]].rename(columns={"observed": "anchor"})
    aligned = visits.merge(anchors, on="year", validate="many_to_one")
    scores, details = [], []
    for year in range(2022, 2025):
        earlier, target = aligned[aligned.year.lt(year)], aligned[aligned.year.eq(year)].copy()
        parameters = fit_curve(earlier)
        train = calibration(labels[labels.year.lt(year)], weather)
        histories = np.array([values for (name, y), values in weather.items() if name == PERRY and y < year])
        sims, _, _ = draws(train, PERRY, histories)
        median_crossing = int(np.quantile(sims, .5, method="inverted_cdf"))
        typical_crossing = anchors.loc[anchors.year.lt(year), "anchor"].median()
        target["forecast_color"] = color_curve(target.day - median_crossing, *parameters)
        target["calendar_color"] = color_curve(target.day - typical_crossing, *parameters)
        details.append(target)
        scores.append(dict(year=year, observed_visit_dates=len(target), monitored_trees=panel_size,
                           forecast_anchor_day=median_crossing, calendar_anchor_day=typical_crossing,
                           observed_anchor_day=target.anchor.iloc[0],
                           forecast_mae_points=(target.forecast_color - target.color_mid).abs().mean(),
                           calendar_mae_points=(target.calendar_color - target.color_mid).abs().mean(),
                           forecast_within_observed_band=target.forecast_color.between(
                               target.color_lower, target.color_upper).mean()))
    scores = pd.DataFrame(scores)
    detail = pd.concat(details, ignore_index=True)
    scores.to_csv(OUTPUT / "perry_color_curve_metrics.csv", index=False)
    detail.to_csv(OUTPUT / "perry_color_curve_observed_visits.csv", index=False)
    drops.to_csv(OUTPUT / "perry_observed_leaf_drop.csv", index=False)

    dates = season_start(year) + pd.to_timedelta(PERCENT_DAYS, unit="D")
    forecast = color_curve(PERCENT_DAYS - median_crossing, *parameters)
    calendar = color_curve(PERCENT_DAYS - typical_crossing, *parameters)
    completed = sims[sims < DAYS]
    if not completed.size:
        raise ValueError(f"No simulated canopy-color crossings in {year}.")
    sampled = color_curve(PERCENT_DAYS[None, :] - completed[:, None], *parameters)
    lo, hi = np.quantile(sampled, [.1, .9], axis=0)
    pd.DataFrame(dict(date=dates, forecast_canopy_percent=forecast, p10_percent=lo,
                      p90_percent=hi, calendar_canopy_percent=calendar)).to_csv(
        OUTPUT / f"perry_{year}_daily_canopy_color.csv", index=False,
    )

    fig, (ax, drop_ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 5.8),
                                     height_ratios=(3, 1), layout="constrained")
    ax.fill_between(dates, lo, hi, color=SITE_COLORS[PERRY], alpha=.15,
                    label="80% timing band given a crossing")
    ax.plot(dates, forecast, color=SITE_COLORS[PERRY], label="Weather model")
    ax.plot(dates, calendar, color="#777777", linestyle="--", label="Historical date")
    ax.errorbar(target.date, target.color_mid, yerr=np.vstack([target.color_mid - target.color_lower,
                                                               target.color_upper - target.color_mid]),
                color="#202020", fmt="o", markersize=3, capsize=2, label="Observed median band")
    ax.axhline(50, color="0.8", linewidth=.7)
    ax.set(ylabel="Colored canopy (%)", ylim=(-3, 103))
    ax.legend(loc="upper left", ncols=2)
    observed_drop = drops[drops.year.eq(year)]
    drop_ax.scatter(observed_drop.date, 100 * observed_drop.drop_fraction, c="#006b75", s=20)
    drop_ax.set(ylabel="Trees with\nleaf drop (%)", ylim=(-5, 105), yticks=(0, 50, 100), xlabel="Observation date")
    drop_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"Perry County daily canopy color · {year} holdout")
    save_figure(fig, "08_perry_daily_canopy_color")
    return scores


def main():
    OUTPUT.mkdir(exist_ok=True)
    labels, weather = load_labels(), load_weather()
    color = load_visits("npn_ohio_color_visits.csv")
    leaf_drop = load_visits("npn_ohio_leaf_drop_visits.csv")
    print(f"Foliage data: {len(color):,} usable color visits and {len(leaf_drop):,} leaf-drop visits.")
    print(summer_backtest(labels, weather).round(3).to_string())
    print(curve_backtest(labels, weather, color, leaf_drop).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
