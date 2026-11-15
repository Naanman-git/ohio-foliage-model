"""Prepare public source extracts used by the foliage extensions."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GENERA = "Acer Quercus Fagus Carya Juglans Liriodendron Betula Fraxinus Liquidambar Platanus Populus Ulmus Nyssa Tilia Aesculus Prunus Cornus Celtis".split()
COLOR_BANDS = {
    "None observed": (0, 0),
    "Less than 5%": (0, 5),
    "5-24%": (5, 25),
    "25-49%": (25, 50),
    "50-74%": (50, 75),
    "75-94%": (75, 95),
    "95% or more": (95, 100),
}


def autumn_observations(path, phase_id):
    records = pd.read_json(path)
    dates = pd.to_datetime(records.observation_date)
    filtered = records.loc[records.phenophase_id.eq(phase_id) &
                           records.genus.isin(GENERA) & dates.dt.month.between(9, 11)].copy()
    return filtered.rename(columns={"site_id": "site", "individual_id": "tree",
                                    "observation_date": "date", "phenophase_status": "status"})


def prepare_color(path):
    records = autumn_observations(path, 498)
    records["band"] = records.intensity_value.astype(str).where(records.status.eq(1))
    records.loc[records.status.eq(0), "band"] = "None observed"
    records = records[records.band.isin(COLOR_BANDS)].copy()
    records["lower_percent"] = records.band.map(lambda value: COLOR_BANDS[value][0])
    records["upper_percent"] = records.band.map(lambda value: COLOR_BANDS[value][1])
    columns = ["site", "date", "tree", "genus", "species", "band", "lower_percent", "upper_percent"]
    visits = records[columns].drop_duplicates()
    return visits[~visits.duplicated(["site", "date", "tree"], keep=False)].sort_values(columns[:3])


def prepare_leaf_drop(path):
    records = autumn_observations(path, 471)
    columns = ["site", "date", "tree", "genus", "species", "status"]
    visits = records.loc[records.status.isin([0, 1]), columns].drop_duplicates()
    return visits[~visits.duplicated(["site", "date", "tree"], keep=False)].sort_values(columns[:3])


def prepare_summer(path):
    data = pd.read_csv(path, parse_dates=["DATE"])
    data = data[data.DATE.dt.month.between(6, 8)].copy()
    for field in ("TMIN", "TMAX"):
        quality = data[field + "_ATTRIBUTES"].astype("string").str.split(",").str[1].fillna("")
        data.loc[quality.ne(""), field] = float("nan")
    return data[["STATION", "DATE", "TMIN", "TMAX"]].sort_values(["STATION", "DATE"])


def main():
    args = argparse.ArgumentParser(description=__doc__)
    for name in ("color", "leaf_drop", "summer"):
        args.add_argument(name, type=Path)
    paths = args.parse_args()
    for data, name in (
        (prepare_color(paths.color), "npn_ohio_color_visits.csv"),
        (prepare_leaf_drop(paths.leaf_drop), "npn_ohio_leaf_drop_visits.csv"),
        (prepare_summer(paths.summer), "noaa_summer_daily.csv"),
    ):
        data.to_csv(ROOT / "data" / name, index=False)
        print(f"{name}: {len(data):,} rows")


if __name__ == "__main__":
    main()
