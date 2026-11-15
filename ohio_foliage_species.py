"""Test whether species offsets improve predictions for monitored individual trees."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ohio_foliage_backtest import OUTPUT, ROOT, SITES, TEST_YEAR, save_figure


def tree_windows():
    trees = pd.read_csv(ROOT / "data/npn_ohio_tree_crossings.csv",
                        parse_dates=["last_below", "first_at_or_above"])
    sites = {key: name for name, (keys, *_) in SITES.items() for key in keys}
    trees["location"] = trees.site.map(sites)
    trees = trees.dropna(subset=["location"]).copy()
    trees["taxon"] = trees.genus + " " + trees.species
    start = pd.to_datetime(trees.year.astype(str) + "-09-01")
    trees["lower"] = (trees.last_below - start).dt.days
    trees["upper"] = (trees.first_at_or_above - start).dt.days
    trees["midpoint"] = (trees.lower + trees.upper) / 2
    return trees


def species_offsets(trees):
    """Compare species with other species observed in the same place and year."""
    groups = trees.groupby(["location", "year"])
    mixed = trees.loc[groups.taxon.transform("nunique").ge(2)].copy()
    mixed["relative_day"] = mixed.midpoint - mixed.groupby(
        ["location", "year"]).midpoint.transform("median")
    yearly = mixed.groupby(["location", "year", "taxon"]).relative_day.median()
    by_species = yearly.groupby("taxon").agg(mean_offset="mean", site_years="size")
    # Shrink toward zero using two site-years' worth of prior weight.
    by_species["offset_days"] = (by_species.mean_offset * by_species.site_years
                                 / (by_species.site_years + 2))
    return by_species


def main():
    trees = tree_windows()
    model = pd.read_csv(OUTPUT / "backtest_and_2026_predictions.csv")
    panels = []
    for row in model[model["mode"].eq("Preseason")].itertuples():
        panel = trees.loc[trees.location.eq(row.site) & trees.year.eq(row.year)].copy()
        offsets = species_offsets(trees.loc[trees.year.lt(row.year)])
        panel["training_site_years"] = panel.taxon.map(offsets.site_years).fillna(0).astype(int)
        panel["offset_days"] = panel.taxon.map(offsets.offset_days).fillna(0)
        panel["pooled_day"] = row.median
        panel["species_day"] = np.clip(panel.pooled_day + panel.offset_days, 0, 90)
        panel["pooled_error"] = abs(panel.pooled_day - panel.midpoint)
        panel["species_error"] = abs(panel.species_day - panel.midpoint)
        panels.append(panel)
    predictions = pd.concat(panels, ignore_index=True)
    predictions.to_csv(OUTPUT / "species_tree_backtest.csv", index=False)
    errors = predictions.groupby(["location", "year"]).agg(
        trees=("tree", "size"), trees_with_prior_species=("training_site_years", lambda x: x.gt(0).sum()),
        pooled_mae=("pooled_error", "mean"), species_mae=("species_error", "mean"),
    ).reset_index()
    errors.to_csv(OUTPUT / "species_site_year_scores.csv", index=False)

    perry = trees.loc[trees.location.eq("Perry County") & trees.year.lt(TEST_YEAR)]
    yearly = perry.groupby(["year", "taxon"]).agg(
        trees=("tree", "nunique"), lower=("lower", "median"),
        midpoint=("midpoint", "median"), upper=("upper", "median"),
    ).reset_index()
    yearly.to_csv(OUTPUT / "perry_species_observed_windows.csv", index=False)
    offsets = species_offsets(trees.loc[trees.year.lt(TEST_YEAR)])
    candidate = model.loc[model["mode"].eq("Preseason") & model.year.eq(TEST_YEAR)].copy()
    roster = trees.loc[trees.year.lt(TEST_YEAR), ["location", "taxon"]].drop_duplicates()
    roster = roster.merge(candidate[["site", "median"]], left_on="location", right_on="site")
    roster["prior_mixed_site_years"] = roster.taxon.map(offsets.site_years).fillna(0).astype(int)
    roster["offset_days"] = roster.taxon.map(offsets.offset_days)
    roster["exploratory_date"] = (pd.Timestamp(f"{TEST_YEAR}-09-01") + pd.to_timedelta(
        (roster["median"] + roster.offset_days).round(), unit="D")).dt.date
    roster.drop(columns="site").to_csv(OUTPUT / "species_2026_exploratory_dates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    colors = ["#006b75", "#814781", "#936026"]
    for i, (taxon, rows) in enumerate(yearly.groupby("taxon")):
        ax.errorbar(rows.year + (i - 1) * .12, rows.midpoint,
                    yerr=[rows.midpoint - rows.lower, rows.upper - rows.midpoint],
                    fmt="o", capsize=3, color=colors[i], label=taxon)
    ax.set(title="Observed 50% color windows by species in Perry County",
           xlabel="Autumn", ylabel="Days after September 1", xticks=sorted(yearly.year.unique()))
    ax.legend()
    save_figure(fig, "09_perry_species_windows")
    past = errors.loc[errors.year.lt(TEST_YEAR)]
    print(past.to_string(index=False))
    print(past[["pooled_mae", "species_mae"]].mean().round(3))


if __name__ == "__main__":
    main()
