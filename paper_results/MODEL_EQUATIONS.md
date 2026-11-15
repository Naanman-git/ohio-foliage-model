# Model equations and code map

The main report uses the observed-data workflow. The original notebook and reference pack describe a separate tier prototype. Equations below distinguish computations that produced the reported Ohio results from prototype features and designs considered during planning.

## A. Observed-data timing model

**Field target.** For tree $j$, let $l_j$ and $u_j$ denote the dates bracketing the first qualifying 50% crossing. Dates are days after September 1. For site $s$ and year $y$,

$$
 L_{sy}^{obs}=\operatorname{med}_j(l_j),\qquad
 U_{sy}^{obs}=\operatorname{med}_j(u_j),\qquad
 O_{sy}=\frac{L_{sy}^{obs}+U_{sy}^{obs}}{2}.
$$

**Weather and daylight.** Use station minimum temperature corrected to the site elevation. The fixed solar calendar uses ordinal day $244+d$ in every year and day 172 as a solstice reference. Latitude and declination are in radians in the trigonometric functions.

$$
 T_d=T_{\mathrm{NOAA},d}-0.0065(z_{\mathrm{site}}-z_{\mathrm{station}}),
 \qquad \delta(j)=\frac{23.44\pi}{180}\sin\frac{2\pi(284+j)}{365}.
$$

$$
 H(j)=\frac{24}{\pi}\arccos\bigl[\operatorname{clip}(-\tan\phi\tan\delta(j),-1,1)\bigr],
 \qquad F_d=1-\frac{H(244+d)}{H(172)}.
$$

$$
 S_d=\sum_{k=0}^{d}\max(0,15-T_k)F_k,\qquad
 D=\min\{d:S_d\ge C\}.
$$

Use a separate no-crossing outcome $D=91$ when the threshold is still unmet after November 30. With temperature entered in °C and elevations in meters, the adjustment has units °C per meter. The core model uses integer date indices.

**Calibration and simulation.** Interpolate $S$ to each training window's two bounds, giving $c_i^-=S(L_i^{obs})$ and $c_i^+=S(U_i^{obs})$. Compute a center for each site with earlier matched observations only.

$$
 \mu_s=\operatorname{med}_{i:s_i=s}\frac{c_i^-+c_i^+}{2},\qquad
 U_b\mid i_b\sim\mathcal U(c_{i_b}^-,c_{i_b}^+).
$$

$$
 C_b=\max\{0.01,m(\mu_s+U_b-\mu_{s_{i_b}})\},\qquad m=1\ \text{in the core model}.
$$

Here $i_b$ is sampled uniformly from the earlier training site-years. Each of 5,000 draws also samples a whole earlier autumn from the target site's weather series. For a hindcast, replace days with valid target-season station observations before evaluating $S$. The 2026 runs stop using observed weather after November 10. This keeps temporal temperature structure by resampling seasons and uses calibration variation to represent threshold uncertainty.

**Daily percentages and summary dates.** Let $F_N$ denote the empirical cumulative distribution of simulated crossing dates, including the no-crossing outcome. The simulation reports $Q_p=\inf\{d:F_N(d)\ge p\}$ for $p=0.10,0.50,0.90$. For the display, pick seven consecutive days with the largest sum.

$$
 \widehat p(d)=\frac{1}{5000}\sum_{b=1}^{5000}\mathbf{1}\{D_b=d\},
 \qquad \widehat W=\arg\max_w\sum_{d=w}^{w+6}\widehat p(d).
$$

The daily chart shows $100\widehat p(d)$. It gives simulation probabilities for reaching a color threshold. The Perry curve in B estimates the amount of canopy in color.

**Validation.** Forecast error is $e_i=\widehat D_i-O_i$, where $\widehat D_i=Q_{0.50}$. The baseline is the median of earlier $O_{sy}$ at that site. For finite predictions, $\mathrm{MAE}=n^{-1}\sum_i|e_i|$ and $\mathrm{RMSE}=\sqrt{n^{-1}\sum_i e_i^2}$. Seven-day accuracy is the fraction with $|e_i|\le7$. A simulation band overlaps a field window when $Q_{.10}\le U_i^{obs}$ and $Q_{.90}\ge L_i^{obs}$. Probability within the observed window is the fraction of draws with $L_i^{obs}\le D_b\le U_i^{obs}$. The horizon is unverified. These calculations are in [the core backtest](../ohio_foliage_backtest.py) and [diagnostics](../ohio_foliage_diagnostics.py).

## B. Observed-data extensions and sensitivity

**Elevation and summer tests.** The elevation sensitivity reruns the backtest with the lapse correction set to zero. Temperature and rainfall are separate candidate predictors. For summer temperature use the mean of daily $(T_{min}+T_{max})/2$. For rainfall use $92\overline{P}_{\mathrm{daily}}/100$, in 100 mm units, after screening out quality flags. Both require at least 88 valid days.

Let $X_i$ denote either summer predictor and $Z_i=\log \max[0.01,(c_i^-+c_i^+)/2]$. Demean $X$ and $Z$ by training site to get $x_i$ and $z_i$. With earlier years only,

$$
 \widehat\beta=\frac{\sum_i x_i z_i}{\sum_i x_i^2+4},
 \qquad m=\exp\left[\widehat\beta(X_{sy}-\overline X_{s,\mathrm{prior}})\right].
$$

This $m$ multiplies the sampled threshold before clipping, using the same random draws for each paired comparison. [Summer temperature code](../ohio_foliage_extensions.py) · [Rainfall code](../ohio_foliage_precipitation.py)

**Species.** Demean against the median tree midpoint in each earlier mixed-species site-year. Let $r_{ig}$ be the median residual for taxon $g$ in its earlier mixed site-year $i$, and let $n_g$ be the number of such site-years. Pooling uses equal weight per represented site-year. With no eligible match, use zero offset in the backtest.

$$
 a_g=\frac{\sum_i r_{ig}}{n_g+2},
 \qquad \widehat D_g=\operatorname{clip}(\widehat D+a_g,0,90).
$$

Score individual tree midpoint errors within each site-year, then average across site-years. The resulting scores apply to monitored trees. [Species code](../ohio_foliage_species.py)

**Perry County color curve.** Let $t=d-D$, days relative to the crossing. Parameters $r,A,q$ have bounds $[0.5,30]$, $[0,50]$, and $[3,70]$ respectively. The curve reaches 50% at $t=0$. For $t<0$ it is below 50%, and on both sides it stays within 0–100%.

$$
 G(t)=
 \begin{cases}
 100/(1+e^{-t/r}), & t\le 0,\\
 [50+A(1-e^{-t/r})]e^{-(t/q)^2}, & t>0.
 \end{cases}
$$

$$
 \min_{r,A,q}\sum_{y\in\mathrm{prior}}\frac{1}{n_y}
 \sum_{i=1}^{n_y}\bigl[G(d_i-O_y)-M_i\bigr]^2 .
$$

$M_i$ is the median across panel trees of the color-category midpoint on the visit date. We require at least seven observed trees. The plotted timing band takes percentiles of $G(d-D_b)$ among crossing draws with shape held fixed. The recorded leaf-drop fraction averages yes/no reports across observed panel trees. It does not measure the percentage of leaves lost. [Curve and panel code](../ohio_foliage_extensions.py)

**Spring budbreak.** Match autumn trees to the same site, tree ID, and year. Spring intervals run from the last observed absence before the first observed presence to the first observed presence in January–June. The audit counts pairs with a spring width of at most 14 days. Oxford had zero qualifying matches, so this analysis does not fit a spring coefficient. [Spring audit code](../ohio_foliage_spring.py)

**Long-term weather scenarios.** For each station, fit $\overline T_y=\alpha+\beta y+\epsilon_y$ to complete fall means. For $n$ fall seasons and slope standard error $s_\beta$, use

$$
 \Delta_0=10\widehat\beta,\qquad
 \Delta_\pm=10\left(\widehat\beta\pm t_{.975,n-2}s_\beta\right).
$$

Add the shift to every day of each paired weather draw, fixing its threshold and random sampling. Report the 10th, 50th, and 90th percentiles of $D_b(\Delta)-D_b(0)$ among simulations that cross in both runs. Also test +1 °C. Since $\max(0,15-T-\Delta)$ cannot increase under warming and $F_d\ge0$ over the modeled season, matched warming runs cannot cross earlier. This is a property of the model's stress equation. [Scenario code](../ohio_foliage_diagnostics.py)

## C. Original notebook and reference-pack design

The original [Jupyter notebook](../Ohio_Foliage_Model.ipynb) and [Python prototype](../ohio_foliage_model.py) demonstrate the initial three-tier design. Their default mode generates synthetic weather and calibrated synthetic target dates. The reported Ohio site forecasts and validation use A–B. The planning pack proposed county elevation terciles and measured area shares. The notebook's real-weather configuration groups stations by station elevation and supplies illustrative equal tier weights.

**Tier inputs and accumulation.** Average available station temperatures after correction to the tier elevation. $R_7$ is trailing seven-day rain. The prototype sets $R_{ref}=25\,\mathrm{mm}$ and disables the rainfall factor by default with $\lambda=0$.

$$
 T_{m,d}=\operatorname{mean}_{s\in m}\bigl[T_{s,d}+\gamma(z_m-z_s)\bigr],
 \qquad W_d=1-\lambda\min(1,R_7(d)/R_{ref}).
$$

$$
 C_m(d)=\sum_{k=0}^{d}(T_{base}-T_{m,k})_+F_{m,k}W_k,
 \qquad c_m=a+b(\phi_m-\overline\phi)+c(z_m-\overline z).
$$

The planning pack also proposed $P_m(d)=100/[1+\exp(-k(C_m(d)-c_m))]$ with positive $k$. At 50%, its first crossing is equivalent to $C_m(d)\ge c_m$. The notebook computes the stress crossing directly. The [planning checks module](../ohio_foliage_planning_checks.py) implements the optional logistic conversion separately. This proposed stress-to-color logistic is separate from the implemented Perry County rise-and-decline curve.

For the first crossing day $d>0$, the prototype linearly interpolates

$$
 D_m=d-1+\frac{c_m-C_m(d-1)}{C_m(d)-C_m(d-1)}.
$$

It returns zero if already crossed on day zero and the season-length sentinel if no crossing occurs. Its differential-evolution objective over the configured parameter bounds is

$$
 \min_{T_{base},a,b,c}\sum_{m,y\in\mathrm{train}}(D_{my}-O_{my})^2
       +0.01\bigl[b^2+(100c)^2\bigr].
$$

**Prototype simulation.** The preseason base $B_d$ is the mean of earlier complete temperature seasons for each day. The hindcast base uses available season weather and fills missing weather with climatology. A simulated temperature is $B_d+\Delta+\eta_d$. The weather noise uses

$$
 \eta_0=\xi_0,\qquad \eta_d=\rho\eta_{d-1}+\sqrt{1-\rho^2}\,\xi_d,
 \qquad \xi_d\sim\mathcal N(0,\sigma_T^2).
$$

The preseason $\sigma_T$ is the standard deviation of daily residuals from climatology. The hindcast default is 1 °C. The average lag-one anomaly correlation is clipped to [0, 0.95]. Threshold simulations use $c_m+\varepsilon_C$, with $\varepsilon_C\sim\mathcal N(0,\sigma_C^2)$ and

$$
 \sigma_C=\max(\operatorname{sd}(e),1)\,\operatorname{mean}_{m,y}
 \frac{C_m(i+1)-C_m(i-1)}{2},
$$

where $i$ is the observed date clipped to interior days. The notebook estimates week probabilities by the fraction of 1,000 simulations crossing in the displayed week bin and combines tier probabilities with $p_{mix,w}=\sum_m\omega_m p_{m,w}$. Default weights are each $1/3$; this illustrative output has no measured statewide coverage. The convex cold-stress term means adding zero-mean temperature noise can raise expected accumulation. This is a property of the prototype's climatology-plus-noise construction.

**Prototype validation, sensitivity and horizon.** Leave-one-year-out calibration reports MAE, RMSE, and signed bias $\overline e$. Temperature perturbations $\Delta=-2,-1,0,1,2$ °C give $\operatorname{med}D(\Delta)-\operatorname{med}D(0)$; regression of the median against $\Delta$ gives the days-per-degree slope. Separate sensitivity runs vary $\sigma_T$ and $\sigma_C$ by ±20% and set $\rho=0$. For small warming $\Delta$ and an unchanged set $I$ of days below the base temperature, the reference pack's first-order approximation is
$$
 \Delta D\approx\frac{\Delta\sum_{d\in I,\, d\le D} F_{m,d}W_d}
 {C_m(D)-C_m(D-1)}.
$$
This approximation is implemented in the [planning checks module](../ohio_foliage_planning_checks.py). The notebook computes its sensitivity numerically.

The prototype fits a fall temperature slope and compares its decadal estimate, twice that estimate, and the estimate plus two standard errors. It also computes an exploratory spread

$$
 \sigma_{proxy}(h)=\sqrt{\sigma_w^2+\sigma_{res}^2+(h\widehat\delta)^2},
$$

using mean simulation spread, calibration residual spread, and a fitted residual drift. The planning pack proposed finding the first $h$ for which this proxy exceeds seven days. It may count some uncertainty twice and lacks decade-scale validation. The report consequently uses the observed-data scenarios in B and gives no accuracy horizon.
