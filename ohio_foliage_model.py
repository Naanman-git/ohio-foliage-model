# %% [markdown]
# # Ohio Statewide Fall-Foliage Peak Model (Elevation-Tiered)
# Two-layer model: (1) deterministic senescence accumulation, (2) Monte Carlo wrapper.
# Run all cells for an explicitly synthetic demonstration. Upload ohio_noaa_fall_daily.csv
# alongside the notebook to see an independent real-weather coverage check.
# To model real Ohio foliage, add verified peak observations in CONFIG and disable synthetic mode.

# %%
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from scipy.optimize import differential_evolution
from scipy.stats import linregress

# %% [markdown]
# ## CONFIG (edit this cell)

# %%
USE_SYNTHETIC = True         # True = pipeline test with FAKE data. NEVER use those results in the report.

TEMP_UNIT      = "C"         # unit of TMIN in your NOAA CSVs: "F" or "C"
TEST_YEAR      = 2026
TRAIN_YEARS    = list(range(2016, 2026))     # years with known observed peak dates
TREND_YEARS    = list(range(1996, 2026))     # years used for the warming-trend regression
P_STAR         = 50.0        # peak = first day canopy color >= P_STAR percent
K_LOGISTIC     = 0.35        # logistic steepness (shapes P(t) curve; does not move the peak date at P*=50)
N_SIM          = 1000        # Monte Carlo runs
LAPSE_C_PER_M  = -6.5/1000   # lapse-rate correction, deg C per metre of elevation
LAMBDA_PRECIP  = 0.0         # 0 = ignore precipitation; try 0.2-0.5 as an extension
R_REF_MM       = 25.0        # weekly rain (mm) at which the delay is at its maximum
TOL_DAYS       = 7.0         # forecast-error tolerance used for the "valid for h years" statement
RUN_LOYO       = True        # leave-one-year-out cross-validation of the calibration
DELTAS         = [-2, -1, 0, 1, 2]           # sensitivity shifts (deg C)
SEED           = 42
SIGMA_T_HINDCAST = 1.0       # small noise (deg C) when replaying the observed test-year weather (station representativeness)

# Station elevation groups for the supplied NOAA extract. These are not county terciles.
WEATHER_FILE = Path("ohio_noaa_fall_daily.csv")
TIER_STATIONS = {
    "Low": ["USW00093812", "USW00094830"],    # Cincinnati Lunken, Toledo Express
    "Mid": ["USW00093824", "USW00004804"],    # Zanesville, Ohio State University
    "High": ["USW00014895", "USW00014891"],   # Akron-Canton, Mansfield
}
# Optional: tier mean elevation (m) from county-mean elevation terciles. None = use mean of station elevations.
TIER_ELEV_OVERRIDE = {"Low": None, "Mid": None, "High": None}
# Share of Ohio's area (or counties) in each tier. Must sum to 1. Replace with your real county counts.
TIER_WEIGHTS = {"Low": 1/3, "Mid": 1/3, "High": 1/3}

# Observed peak dates from ODNR weekly reports / news (YYYY-MM-DD), one per (tier, year).
OBS_PEAK = {
    # ("Low", 2016): "2016-10-21",
}

# %% [markdown]
# ## Core model functions

# %%
N_DAYS = 91                  # Sept 1 .. Nov 30
DOY0   = 244                 # day-of-year of Sept 1 (non-leap; 1-day leap error is negligible)
rng    = np.random.default_rng(SEED)

def day_length(lat_deg, doy):
    """Day length (hours) from the standard astronomical (sunrise-equation) formula."""
    phi = np.radians(lat_deg)
    delta = np.radians(23.44) * np.sin(2*np.pi*(284 + doy)/365)
    x = -np.tan(phi)*np.tan(delta)
    return 24/np.pi*np.arccos(np.clip(x, -1, 1))

def photo_factor(lat_deg):
    doy = DOY0 + np.arange(N_DAYS)
    Lmax = day_length(lat_deg, 172)
    return (Lmax - day_length(lat_deg, doy))/Lmax

def rain_weight(prcp):
    if prcp is None or LAMBDA_PRECIP == 0:
        return np.ones(N_DAYS)
    r7 = pd.Series(prcp).rolling(7, min_periods=1).sum().values
    return 1 - LAMBDA_PRECIP*np.minimum(1, r7/R_REF_MM)

def cumulative_stress(Tmin, Tbase, F, W):
    """Tmin: (N,T) or (T,). Returns C(t) = cumulative senescence units."""
    SU = np.maximum(0, Tbase - Tmin) * F * W
    return np.cumsum(SU, axis=-1)

def peak_day(C, Ccrit):
    """First (fractional) day index where C >= Ccrit. Never reached -> N_DAYS (censored)."""
    C = np.atleast_2d(C)
    Ccrit = np.broadcast_to(np.asarray(Ccrit, float), (C.shape[0],))
    hit = C >= Ccrit[:, None]
    anyhit = hit.any(1)
    idx = hit.argmax(1)
    prev = np.take_along_axis(C, np.maximum(idx-1, 0)[:, None], 1)[:, 0]
    cur  = np.take_along_axis(C, idx[:, None], 1)[:, 0]
    denom = np.where(cur-prev > 0, cur-prev, 1.0)
    frac = np.where(idx > 0, idx-1 + (Ccrit-prev)/denom, 0.0)
    return np.where(anyhit, frac, float(C.shape[1]))

# %% [markdown]
# ## Data loading (NOAA CSVs or synthetic test data)

# %%
def _station_year_array(df, year, allow_trailing_nan):
    idx = pd.date_range(f"{year}-09-01", f"{year}-11-30")
    s = df["TMIN"].reindex(idx)
    s = s.interpolate(limit=3, limit_area="inside")
    p = df["PRCP"].reindex(idx).fillna(0).values if "PRCP" in df else None
    arr = s.values.astype(float)
    nan = np.isnan(arr)
    if nan.any():
        if allow_trailing_nan and (~nan).any() and not nan[:np.flatnonzero(~nan)[-1]+1].any():
            return arr, p            # only trailing NaNs (season still in progress)
        return None, None
    return arr, p

def load_real():
    if not WEATHER_FILE.is_file():
        raise FileNotFoundError(f"Upload {WEATHER_FILE} to the runtime working directory.")
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["DATE"])
    tiers = {}
    for tier, station_ids in TIER_STATIONS.items():
        st = []
        for station_id in station_ids:
            df = weather.loc[weather["STATION"] == station_id].copy()
            if df.empty:
                raise ValueError(f"Missing weather for {station_id} in {tier}.")
            df = df.set_index("DATE").sort_index()
            if TEMP_UNIT == "F":
                df["TMIN"] = (df["TMIN"] - 32) * 5 / 9
                if "PRCP" in df:
                    df["PRCP"] *= 25.4
            st.append(dict(df=df, lat=float(df["LATITUDE"].iloc[0]), z=float(df["ELEVATION"].iloc[0])))
        lat = np.mean([s["lat"] for s in st])
        override = TIER_ELEV_OVERRIDE.get(tier)
        z = np.mean([s["z"] for s in st]) if override is None else override
        years = sorted(set(TRAIN_YEARS) | set(TREND_YEARS) | {TEST_YEAR})
        T, P = {}, {}
        for y in years:
            arrs, prc = [], []
            for s in st:
                a, p = _station_year_array(s["df"], y, allow_trailing_nan=(y == TEST_YEAR))
                if a is not None:
                    arrs.append(a + LAPSE_C_PER_M*(z - s["z"]))
                    if p is not None: prc.append(p)
            if arrs:
                stacked = np.vstack(arrs)
                count = np.isfinite(stacked).sum(axis=0)
                T[y] = np.divide(np.nansum(stacked, axis=0), count,
                                 out=np.full(N_DAYS, np.nan), where=count > 0)
                P[y] = np.mean(np.vstack(prc), axis=0) if prc else None
        tiers[tier] = dict(lat=lat, z=z, T=T, P=P, n_stations=len(st))
        print(f"{tier}: {len(st)} stations, lat={lat:.2f}, elev={z:.0f} m, usable years={len(T)}")
    return tiers

TRUE_SYN = dict(Tbase=10.0, a=18.0, b=1.5, c=-0.01)
def load_synthetic():
    print("="*70, "\n SYNTHETIC DATA - FOR PIPELINE TESTING ONLY - NOT FOR THE REPORT\n", "="*70)
    spec = {"Low": (41.4, 190), "Mid": (40.0, 300), "High": (39.3, 420)}
    tiers = {}
    years = sorted(set(TRAIN_YEARS) | set(TREND_YEARS) | {TEST_YEAR})
    for tier, (lat, z) in spec.items():
        T = {}
        for y in years:
            base = 11.5 - 0.105*np.arange(N_DAYS) - 0.35*(lat-40) - 0.004*(z-300) + 0.04*(y-2016)
            e = np.zeros(N_DAYS)
            for t in range(N_DAYS):
                e[t] = (0.6*e[t-1] if t else 0) + rng.normal(0, 2.0)
            T[y] = base + e
        tiers[tier] = dict(lat=lat, z=z, T=T, P={y: None for y in years}, n_stations=1)
    return tiers

if WEATHER_FILE.is_file():
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["DATE"])
    coverage = weather.groupby("STATION").agg(
        First=("DATE", "min"), Last=("DATE", "max"),
        Elevation_m=("ELEVATION", "first"), Valid_Tmin_days=("TMIN", "count"),
    )
    print("NOAA weather coverage (Sep–Nov only):")
    print(coverage.to_string())

if not USE_SYNTHETIC and not OBS_PEAK:
    raise ValueError("Add verified foliage peak observations in CONFIG before fitting the real model.")

tiers = load_synthetic() if USE_SYNTHETIC else load_real()

# centred covariates keep the regression well conditioned
LAT0 = np.mean([v["lat"] for v in tiers.values()])
Z0   = np.mean([v["z"] for v in tiers.values()])
def ccrit_of(tier, theta):
    _, a, b, c = theta
    return a + b*(tiers[tier]["lat"]-LAT0) + c*(tiers[tier]["z"]-Z0)

if USE_SYNTHETIC:   # build fake "observed" peaks from the true model + 2-day noise
    th = (TRUE_SYN["Tbase"], TRUE_SYN["a"], TRUE_SYN["b"], TRUE_SYN["c"])
    for tier, v in tiers.items():
        F = photo_factor(v["lat"])
        for y in list(TRAIN_YEARS) + [TEST_YEAR]:
            if y in v["T"]:
                C = cumulative_stress(v["T"][y], th[0], F, np.ones(N_DAYS))
                d = peak_day(C, ccrit_of(tier, th))[0] + rng.normal(0, 2)
                OBS_PEAK[(tier, y)] = str(pd.Timestamp(f"{y}-09-01") + pd.Timedelta(days=int(round(d))))[:10]

def obs_idx(tier, year):
    return (pd.Timestamp(OBS_PEAK[(tier, year)]) - pd.Timestamp(f"{year}-09-01")).days

# climatology to fill the unfinished tail of the test season
for tier, v in tiers.items():
    complete = [v["T"][y] for y in TRAIN_YEARS if y in v["T"] and not np.isnan(v["T"][y]).any()]
    if not complete:
        raise ValueError(f"{tier}: no complete training weather seasons.")
    full = np.vstack(complete)
    v["clim"] = full.mean(0)
    if TEST_YEAR in v["T"]:
        miss = np.isnan(v["T"][TEST_YEAR])
        if miss.any():
            print(f"{tier}: {miss.sum()} trailing {TEST_YEAR} days filled with climatology; late predictions depend on imputed weather")
            v["T"][TEST_YEAR] = np.where(miss, v["clim"], v["T"][TEST_YEAR])

# %% [markdown]
# ## Layer 1 calibration (fit Tbase, a, b, c on prior years only)

# %%
def _pairs(years):
    return [(t, y) for t in tiers for y in years if (t, y) in OBS_PEAK and y in tiers[t]["T"]]

def predict_det(tier, year, theta):
    v = tiers[tier]
    W = rain_weight(v["P"].get(year))
    C = cumulative_stress(v["T"][year], theta[0], photo_factor(v["lat"]), W)
    return peak_day(C, ccrit_of(tier, theta))[0]

def objective(theta, pairs):
    err = [(predict_det(t, y, theta) - obs_idx(t, y)) for t, y in pairs]
    return float(np.sum(np.square(err)) + 0.01*(theta[2]**2 + (100*theta[3])**2))   # mild ridge on b, c

BOUNDS = [(5, 15), (0, 80), (-6, 6), (-0.05, 0.05)]   # Tbase, a, b, c  (widen if the optimiser hits an edge)

def calibrate(years, maxiter=200):
    pairs = _pairs(years)
    if len(pairs) < len(BOUNDS) or {tier for tier, _ in pairs} != set(tiers):
        raise ValueError("Calibration needs weather-matched peak observations across all tiers.")
    res = differential_evolution(objective, BOUNDS, args=(pairs,), seed=SEED, maxiter=maxiter,
                                 popsize=15, tol=1e-6, polish=True)
    return res.x, pairs

theta_hat, train_pairs = calibrate(TRAIN_YEARS)
print("Calibrated theta (Tbase, a, b, c):", np.round(theta_hat, 4))
for (lo, hi), nm, v in zip(BOUNDS, ["Tbase", "a", "b", "c"], theta_hat):
    if abs(v-lo) < 1e-3*(hi-lo) or abs(v-hi) < 1e-3*(hi-lo):
        print(f"  WARNING: {nm} sits on its bound - widen BOUNDS")

res_days = np.array([predict_det(t, y, theta_hat) - obs_idx(t, y) for t, y in train_pairs])
print(f"In-sample: MAE={np.mean(np.abs(res_days)):.2f} d, RMSE={np.sqrt(np.mean(res_days**2)):.2f} d, n={len(res_days)}")

# %% [markdown]
# ## Noise parameters estimated from data (sigma_T, rho, sigma_C)

# %%
def estimate_noise():
    an = []
    for v in tiers.values():
        M = np.vstack([v["T"][y] for y in TRAIN_YEARS if y in v["T"] and not np.isnan(v["T"][y]).any()])
        an.append(M - M.mean(0))
    A = np.vstack(an)
    sigT = A.std()
    correlations = [np.corrcoef(a[:-1], a[1:])[0, 1] for a in A
                    if a[:-1].std() > 0 and a[1:].std() > 0]
    rho = np.mean(correlations) if correlations else 0.0
    return float(sigT), float(np.clip(rho, 0, 0.95))

sigma_T, rho = estimate_noise()
# slope of C near the peak converts a day-error into threshold units
slopes = []
for t, y in train_pairs:
    v = tiers[t]
    C = cumulative_stress(v["T"][y], theta_hat[0], photo_factor(v["lat"]), rain_weight(v["P"].get(y)))
    i = int(np.clip(obs_idx(t, y), 1, N_DAYS-2))
    slopes.append((C[i+1]-C[i-1])/2)
sigma_C = float(max(np.std(res_days), 1.0) * np.mean(slopes))
print(f"sigma_T={sigma_T:.2f} C, rho={rho:.2f}, sigma_C={sigma_C:.2f} (threshold units)")

# %% [markdown]
# ## Layer 2: Monte Carlo

# %%
def monte_carlo(tier, year, theta, delta=0.0, N=N_SIM, sT=None, sC=None, r=None, seed=None, use_clim=False):
    """use_clim=False: hindcast on OBSERVED weather (small noise). use_clim=True: pre-season forecast on
    CLIMATOLOGY + full anomaly noise (this is what you publish before the season)."""
    g = np.random.default_rng(seed if seed is not None else SEED)
    sT = (sigma_T if use_clim else SIGMA_T_HINDCAST) if sT is None else sT
    sC = sigma_C if sC is None else sC
    r  = rho if r is None else r
    v = tiers[tier]
    xi = g.normal(0, sT, (N, N_DAYS))
    eps = np.zeros_like(xi); eps[:, 0] = xi[:, 0]
    for t in range(1, N_DAYS):
        eps[:, t] = r*eps[:, t-1] + np.sqrt(1-r**2)*xi[:, t]
    base = v["clim"] if use_clim else v["T"][year]
    Tm = base[None, :] + delta + eps
    rain = None if use_clim else v["P"].get(year)
    C = cumulative_stress(Tm, theta[0], photo_factor(v["lat"]), rain_weight(rain))
    cc = ccrit_of(tier, theta) + g.normal(0, sC, N)
    return peak_day(C, cc)

def week_label(day_idx, year):
    if day_idx >= N_DAYS: return "No peak by Nov 30"
    d = pd.Timestamp(f"{year}-09-01") + pd.Timedelta(days=int(day_idx))
    return "Wk of " + (d - pd.Timedelta(days=d.weekday())).strftime("%b %d")

def summarize(pk, year):
    d = pd.Timestamp(f"{year}-09-01")
    q10, q50, q90 = np.percentile(pk, [10, 50, 90])
    def dt(q):
        if q > N_DAYS - 1:
            return "Beyond Nov 30"
        return (d + pd.Timedelta(days=float(q))).strftime("%b %d")
    return dict(median=q50, lo=q10, hi=q90, median_date=dt(q50), lo_date=dt(q10), hi_date=dt(q90))

def weekly_probs(pk, year):
    labs = pd.Series([week_label(x, year) for x in pk])
    vc = labs.value_counts(normalize=True)
    key = lambda x: (x.startswith("No"), pd.to_datetime(x[6:]+f" {year}", format="%b %d %Y") if not x.startswith("No") else pd.Timestamp.max)
    return vc.reindex(sorted(vc.index, key=key))

# %% [markdown]
# ## Out-of-sample prediction for the test year + validation

# %%
rows, probs, mc_test = [], {}, {}
for tier in tiers:
    if TEST_YEAR not in tiers[tier]["T"]:
        print(f"{tier}: no usable {TEST_YEAR} weather; skipping hindcast.")
        continue
    pk = monte_carlo(tier, TEST_YEAR, theta_hat)
    mc_test[tier] = pk
    s = summarize(pk, TEST_YEAR)
    probs[tier] = weekly_probs(pk, TEST_YEAR)
    row = dict(Tier=tier, Pred_median=s["median_date"], P10=s["lo_date"], P90=s["hi_date"])
    if (tier, TEST_YEAR) in OBS_PEAK:
        o = obs_idx(tier, TEST_YEAR)
        row.update(Observed=OBS_PEAK[(tier, TEST_YEAR)], Error_days=round(s["median"]-o, 1),
                   Inside_80pct=bool(s["lo"] <= o <= s["hi"]))
    rows.append(row)
val = pd.DataFrame(rows); print("HINDCAST on observed", TEST_YEAR, "weather (out-of-sample parameters):"); print(val.to_string(index=False))
# Pre-season style forecast: climatology + full anomaly noise (what the newsroom would publish in August)
fc_rows, fc_probs = [], {}
for tier in tiers:
    pk = monte_carlo(tier, TEST_YEAR, theta_hat, use_clim=True); s_ = summarize(pk, TEST_YEAR)
    fc_probs[tier] = weekly_probs(pk, TEST_YEAR)
    fc_rows.append(dict(Tier=tier, Pred_median=s_["median_date"], P10=s_["lo_date"], P90=s_["hi_date"]))
print("\nPRE-SEASON-STYLE FORECAST (climatology baseline):"); print(pd.DataFrame(fc_rows).to_string(index=False))
print("\nPre-season weekly probabilities:"); print(pd.DataFrame(fc_probs).fillna(0).round(3))
print("\nWeekly peak probabilities:")
prob_df = pd.DataFrame(probs).fillna(0); print(prob_df.round(3))

# statewide mix
if set(prob_df) == set(tiers):
    state = sum(TIER_WEIGHTS[t] * prob_df[t] for t in tiers)
    print("\nCombined tier probabilities (configured weights):")
    print(state.round(3))

# %% [markdown]
# ## Cross-validation of calibration (leave one year out)

# %%
if RUN_LOYO and len(TRAIN_YEARS) >= 4:
    errs = []
    for hold in sorted({year for _, year in train_pairs}):
        remaining = [y for y in TRAIN_YEARS if y != hold]
        fold_pairs = _pairs(remaining)
        if len(fold_pairs) < len(BOUNDS) or {t for t, _ in fold_pairs} != set(tiers):
            print(f"LOYO {hold}: insufficient remaining observations; skipping fold.")
            continue
        th, _ = calibrate(remaining, maxiter=80)
        for t in tiers:
            if (t, hold) in OBS_PEAK and hold in tiers[t]["T"]:
                errs.append(predict_det(t, hold, th) - obs_idx(t, hold))
    errs = np.array(errs)
    if errs.size:
        print(f"LOYO: MAE={np.mean(np.abs(errs)):.2f} d, RMSE={np.sqrt(np.mean(errs**2)):.2f} d, bias={errs.mean():+.2f} d, n={len(errs)}")
        print("Weekly reporting may introduce several days of timing uncertainty.")

# %% [markdown]
# ## Sensitivity: temperature shifts and parameter perturbations

# %%
sens = {t: [np.median(monte_carlo(t, TEST_YEAR, theta_hat, delta=d, use_clim=True)) for d in DELTAS] for t in tiers}
sens_df = pd.DataFrame(sens, index=DELTAS); sens_df.index.name = "delta_C"
shift = sens_df - sens_df.loc[0]; print("Peak shift (days) vs baseline:"); print(shift.round(1))
kappa = {t: np.polyfit(DELTAS, sens_df[t].values, 1)[0] for t in tiers}
print("Sensitivity kappa (days per deg C):", {k: round(v, 2) for k, v in kappa.items()})

pert = []
for name, kw in [("sigma_T x0.8", dict(sT=sigma_T*0.8)), ("sigma_T x1.2", dict(sT=sigma_T*1.2)),
                 ("sigma_C x0.8", dict(sC=sigma_C*0.8)), ("sigma_C x1.2", dict(sC=sigma_C*1.2)),
                 ("rho = 0", dict(r=0.0))]:
    for t in tiers:
        pk = monte_carlo(t, TEST_YEAR, theta_hat, use_clim=True, **kw)
        pert.append(dict(Case=name, Tier=t, Median_shift_d=round(np.median(pk)-np.median(monte_carlo(t, TEST_YEAR, theta_hat, use_clim=True)), 2),
                         Spread_P10_P90_d=round(np.percentile(pk, 90)-np.percentile(pk, 10), 1)))
print(pd.DataFrame(pert).to_string(index=False))

# %% [markdown]
# ## 10-year outlook: data-driven warming trend + scenarios + validity horizon

# %%
out10 = {}
for t in tiers:
    ys = [y for y in TREND_YEARS if y in tiers[t]["T"] and not np.isnan(tiers[t]["T"][y]).any()]
    if len(ys) < 10:
        print(f"{t}: only {len(ys)} complete years - trend unreliable, use NOAA Climate at a Glance instead"); continue
    means = [tiers[t]["T"][y].mean() for y in ys]
    lr = linregress(ys, means)
    d10 = 10*lr.slope; se10 = 10*lr.stderr
    scen = {"trend": d10, "2x trend": 2*d10, "upper (trend+2SE)": d10 + 2*se10}
    base = np.median(monte_carlo(t, TEST_YEAR, theta_hat, use_clim=True))
    out10[t] = dict(slope_per_decade=d10, se=se10,
                    shifts={k: float(np.median(monte_carlo(t, TEST_YEAR, theta_hat, delta=v, use_clim=True)) - base) for k, v in scen.items()})
    print(f"{t}: Sep-Nov Tmin trend = {d10:+.2f} +/- {se10:.2f} C/decade (n={len(ys)} yrs); shifts (days): "
          f"{ {k: round(v,1) for k, v in out10[t]['shifts'].items()} }")

# Exploratory error proxy. It does not establish forecast validity.
yrs = np.array([y for t, y in train_pairs]); 
if len(set(yrs)) >= 5:
    drift = linregress(yrs, res_days).slope                       # days of bias per year
    sig_w = float(np.mean([np.std(monte_carlo(t, TEST_YEAR, theta_hat, use_clim=True)) for t in tiers]))   # weather-driven spread (days)
    sig_th = float(np.std(res_days))
    print(f"\nResidual drift = {drift:+.3f} d/yr (n={len(yrs)} obs; treat as indicative only)")
    for h in range(0, 11):
        sp = np.sqrt(sig_w**2 + sig_th**2 + (h*drift)**2)
        print(f"  h={h:2d} yr: heuristic spread = {sp:.1f} d" + ("   <-- exceeds tolerance" if sp > TOL_DAYS else ""))

# %% [markdown]
# ## Figures

# %%
os.makedirs("figs", exist_ok=True)
# Fig 1: pre-season weekly probability per tier
prob_df = pd.DataFrame(fc_probs).fillna(0)
ax = prob_df.plot(kind="bar", figsize=(8, 4)); ax.set_ylabel("Probability of peak"); ax.set_title(f"{TEST_YEAR} weekly peak probability by elevation tier")
plt.tight_layout(); plt.savefig("figs/fig1_weekly_prob.png", dpi=200); plt.show()
# Fig 2: calibration scatter
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for t in tiers:
    pr = [(obs_idx(tt, y), predict_det(tt, y, theta_hat)) for tt, y in train_pairs if tt == t]
    if pr: ax.scatter(*zip(*pr), label=t)
lim = [0, N_DAYS]; ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(20, 80); ax.set_ylim(20, 80)
ax.set_xlabel("Observed peak (days after Sep 1)"); ax.set_ylabel("Predicted"); ax.legend(); ax.set_title("Calibration")
plt.tight_layout(); plt.savefig("figs/fig2_calibration.png", dpi=200); plt.show()
# Fig 3: sensitivity
ax = sens_df.plot(marker="o", figsize=(6, 4)); ax.set_xlabel("Temperature shift (deg C)"); ax.set_ylabel("Median peak (days after Sep 1)")
ax.set_title("Sensitivity of peak date to Tmin shift"); plt.tight_layout(); plt.savefig("figs/fig3_sensitivity.png", dpi=200); plt.show()
# Fig 4: 10-year shifts
if out10:
    pd.DataFrame({t: v["shifts"] for t, v in out10.items()}).plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Peak delay (days)"); plt.title("10-year warming scenarios"); plt.tight_layout()
    plt.savefig("figs/fig4_10yr.png", dpi=200); plt.show()
print("Figures saved in ./figs")
