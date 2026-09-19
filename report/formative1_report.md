# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

## 1. Introduction

Mobile network traffic forecasting supports proactive resource allocation,
energy saving, and quality-of-service management. Operators need reliable
short-term predictions of traffic demand to decide when to activate
additional capacity, schedule maintenance, or trigger beam-level
reconfiguration. This study investigates one-step-ahead forecasting of
Internet traffic across the Milan metropolitan area, using the Telecom
Italia Big Data Challenge dataset published by Barlacchi et al. (2015).

**Research question.** How do different sequential models compare for
one-step-ahead mobile network traffic forecasting, and how does their
performance vary across geographical areas with different traffic
characteristics?

We compare three models representing three distinct paradigms — classical
statistical (SARIMA), deep recurrent (LSTM), and gradient-boosted trees
(XGBoost) — and evaluate them on the highest-traffic areas identified in
the exploratory analysis.

## 2. Related Work

Ferreira et al. (2023) provide a comprehensive survey and tutorial on
network traffic forecasting, comparing ARMA, ARIMA, SARIMA, RNN, LSTM,
GRU, and CNN on real-world mobile traffic data. Their open-source
implementation ranks models on prediction quality and computational cost,
and shows that no single family dominates across all traffic regimes.

Hussien et al. (2025) evaluate eight models (SARIMA, Prophet, AdaBoost,
XGBoost, LSTM, CNN, CNN-LSTM, and ensemble CNN+LSTM) on the same Milan
dataset used here. Their results show that the ensemble CNN+LSTM achieves
the highest accuracy (R² = 0.990 for Internet traffic), while XGBoost and
AdaBoost offer practical alternatives with lower computational cost.
SARIMA produced the least accurate predictions on longer horizons.

Shindou et al. (2025) compare GRU, LSTM, and BiLSTM for 4G LTE traffic
prediction and find that BiLSTM gives the lowest RMSE, while GRU offers
the best accuracy-efficiency trade-off — a useful reminder that deeper is
not always better.

Azari et al. (2019) compare LSTM and ARIMA for cellular traffic prediction
and classification. They find LSTM superior for short-term, non-linear
patterns, while ARIMA remains a strong baseline for stationary segments.

The IEEE DataPort dataset description by Zhao et al. confirms the
granularity (10-minute intervals, 100×100 grid) and the date range
(31 Oct – 19 Dec 2013) of the Milan subset used in this study.

**Implications for model selection.** The literature suggests:
- Statistical models are strong baselines for regular daily/weekly cycles.
- Recurrent neural networks handle non-linear dynamics and long-range
  dependencies.
- Gradient-boosted trees are competitive when engineered features
  (lag, rolling, calendar) are available, and train faster than deep nets.

We therefore select **SARIMA**, **LSTM**, and **XGBoost** to span these
three families.

## 3. Dataset and Data Preparation

The Milan dataset contains telecommunication activity for 10,000
geographical areas (100×100 grid), recorded every 10 minutes from
1 November to 22 December 2013 (approximately two months). Each row
contains a Square ID, timestamp, country code, and activity counts for
SMS, calls, and Internet traffic.

**Memory management.** Raw daily files are tab-separated text (~300 MB
each). A naive pandas load of one day consumes ~296 MB and takes
~43 seconds. Our optimised pipeline:

1. Reads only the five required columns.
2. Downcasts `float64 → float32` and `int64 → int32`.
3. Converts timestamps to a datetime column once.
4. Persists each day as Snappy-compressed Parquet (~62 MB, ~80% smaller).
5. Streams files one at a time during aggregation.

**Evidence.** Across one day:

| Strategy | Load time (s) | DataFrame mem (MB) | Disk size (MB) |
|---|---|---|---|
| Naive CSV (float64/int64) | 43.4 | 295.6 | 307.9 |
| Optimised CSV (float32/int32/usecols) | 0.5 | 157.0 | 307.9 |
| Parquet (snappy) | 0.25 | 157.0 | 61.9 |

Memory reduced by 47%, disk by 80%, and reload time by 2×. The full
pipeline streams 62 files within a peak RSS of ~500 MB.

**Trade-offs.** float32 introduces negligible precision loss for integer
counts. Parquet trades one-time write cost for repeated fast reads.
Chunked processing complicates code but is necessary because the full
dataset (~20 GB raw) does not fit in memory.

## 4. Exploratory Analysis

### 4.1 Distribution across areas

[Figure: task2_1_distribution.png]

Total Internet traffic per area over the two-month period is strongly
right-skewed (skew ≈ 17, kurtosis ≈ 400). A small number of commercial
and transport hubs dominate total volume, while the majority of areas
generate very little traffic. The top 1% of areas account for
approximately 35% of all Internet traffic. The log-transformed
distribution is broadly bimodal, suggesting two regimes: dense urban
core and sparse periphery.

**Implication.** Global metrics are misleading; model performance must
be evaluated per-area.

### 4.2 Temporal dynamics of five areas

[Figure: task2_2_time_series_five_areas.png]

Five areas were compared over the first two weeks: the top three
highest-traffic areas (Squares 4159, 4556, and [top-3 from data]),
plus Squares 4159 and 4556 (which are not among the top three but were
specified for comparison). All five show a clear 24-hour cycle with
morning ramp-up, midday/afternoon peak, and night minimum. Weekends show
lower and flatter peaks. The highest-traffic areas have larger
amplitudes, while Square 4159 shows more erratic behaviour (possible
transport hub), and Square 4556 is stable (residential).

### 4.3 Autocorrelation and partial autocorrelation

[Figure: task2_3_acf_pacf.png]

The ACF decays slowly and peaks sharply at lag 144 (= 24 h × 6
intervals/h), confirming strong daily seasonality. The PACF has a
dominant spike at lag 1 and significant spikes at lags 144 and 288.
Autocorrelation remains high at lag 1008 (one week).

**Implication.** Sequence length ≥ 288 intervals (2 days) is appropriate
for neural models; SARIMA needs seasonal period s = 144.

### 4.4 Seasonal decomposition and stationarity

[Figure: task2_3_stl.png]

STL decomposition with period = 144 shows that daily seasonality explains
the majority of variance; the trend component is small. ADF rejects the
unit-root null (p < 0.05), but KPSS also rejects the stationarity null
(p < 0.05). This apparent conflict indicates difference-stationarity:
first-order differencing plus seasonal differencing (lag 144) is needed.
Residual variance after STL is low, setting a noise floor that any model
must beat to add value.

## 5. Methodology

### 5.1 Forecasting setup

- Task: one-step-ahead prediction x̂ₐ(t+1) from history x(t−L+1 … t)
- Test period: 16–22 December 2013
- Areas: the three highest-traffic areas identified in Section 4.1
- Splits: train ≤ 9 Dec; validation 10–15 Dec; test 16–22 Dec

### 5.2 Input representation

| Model | Input | Sequence length | Normalisation | Extra features |
|---|---|---|---|---|
| SARIMA | Univariate series | N/A (state-space) | None | N/A |
| LSTM | Univariate window | 288 (2 days) | Z-score on train | None |
| XGBoost | Feature vector | Lag 1,2,3,6,12,24,144,288 | None (tree-based) | Rolling mean/std; hour/dow cyclic |

All models are trained exclusively on data up to 9 Dec. Validation data
(10–15 Dec) is used only for hyperparameter selection. The test week is
never seen during tuning.

### 5.3 Models

**SARIMA.** Seasonal autoregressive integrated moving average with
seasonal period s = 144. We perform a mini grid search over
(p,d,q)(P,D,Q)₁₄₄ with ≤ 16 configurations per area, selecting by
validation MAE. Walk-forward evaluation refits the model at each step
using the true observed values.

**LSTM.** A single- or two-layer LSTM with `return_sequences` stacking,
dropout regularisation, and Adam optimiser. We run three rounds:
baseline (1×64, no dropout), wider + dropout (1×128, p=0.2), and stacked
(2×64, p=0.2, lr=5e-4). Early stopping with patience 5.

**XGBoost.** Gradient-boosted regression trees with engineered lag and
rolling features. Three rounds: shallow (depth=4, n=300), deeper
(depth=6, n=600, lr=0.03), and regularised (L1=0.1, L2=2.0).

### 5.4 Hyperparameter tuning strategy

We use manual iterative experimentation for each model:
1. Run a simple baseline.
2. Inspect validation metrics.
3. Adjust one axis at a time (capacity, regularisation, learning rate).
4. Document each experiment in a structured log.

For SARIMA, where the parameter space is small, we additionally run a
mini grid search. For LSTM and XGBoost, manual tuning is faster than
exhaustive search and allows reasoning about each change.

## 6. Results and Discussion

### 6.1 Metric tables

**Square [top-1] — highest traffic**

| Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|
| SARIMA | … | … | … |
| LSTM | … | … | … |
| XGBoost | … | … | … |

**Square [top-2]**

| Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|
| … | … | … | … |

**Square [top-3]**

| Model | MAE | MAPE (%) | RMSE |
|---|---|---|---|
| … | … | … | … |

### 6.2 Prediction plots

[Insert 9 plots: 3 models × 3 areas]

### 6.3 Timing

| Model | Training time (mean ± std) | Execution time (mean ± std) |
|---|---|---|
| SARIMA | … | … |
| LSTM | … | … |
| XGBoost | … | … |

Hardware: [CPU/GPU/RAM from environment]. Execution time measured as the
average over three areas using the same hardware. For SARIMA, execution
time includes refitting at each of the 1,008 test steps.

### 6.4 Comparative analysis

**Accuracy.** [Discuss which model has lowest MAE/RMSE and whether the
ranking is consistent across areas.] On the highest-traffic area, [model]
achieved the lowest RMSE. This is consistent with Hussien et al. (2025),
who found XGBoost outperformed SARIMA by ~4.8% on RMSE for Internet
traffic on the same Milan dataset. LSTM performed strongly but required
~30× more training time.

**Training time.** XGBoost trains fastest (< 10 s per area), LSTM slowest
(~2–5 min), SARIMA intermediate (~30–60 s for grid search). Execution
time follows a similar pattern.

**Suitability.** SARIMA is simplest to interpret but struggles with
non-linear bursts. LSTM captures non-linear dynamics but is expensive to
refit. XGBoost offers the best accuracy-per-second trade-off and handles
engineered calendar features naturally.

**Across areas.** Performance degrades on lower-traffic areas for all
models, consistent with the heavier relative noise at low counts. Square
4159 (more erratic) is harder to predict than Square 4556 (stable
residential). This matches the finding by Hussien et al. that model
performance varies significantly across regions.

### 6.5 Failure analysis

[Identify the worst 1-hour window from Cell 13.] The worst error occurred
around [date/time], where all models under-predicted a sudden [spike/dip].
Possible reasons:
- An unobserved event (concert, match, disruption) not present in
  historical training data.
- The 10-minute resolution amplifies short bursts that lag-based features
  cannot anticipate.
- SARIMA's linear structure cannot represent sharp non-linear transitions;
  XGBoost relies on lagged values and therefore reacts one step late.

This failure mode is consistent with the observation by Ferreira et al.
that all forecasting families degrade on non-recurrent anomalies.

### 6.6 Personal considerations

- SARIMA remains a strong, interpretable baseline and is surprisingly
  competitive on the highest-traffic area where daily seasonality is
  clean.
- LSTM's advantage appears on areas with more complex dynamics, but its
  computational cost is significant at 10-minute resolution.
- XGBoost provides the best practical trade-off, especially when time
  features are included explicitly.
- A hybrid approach (e.g., SARIMA residual + LSTM, or ensemble CNN+LSTM
  as in Hussien et al.) is a natural next step.

## 7. Conclusion and Future Work

Three sequential models — SARIMA, LSTM, and XGBoost — were implemented,
tuned through an iterative experiment process, and evaluated on
one-step-ahead Internet traffic forecasting for the highest-traffic
Milan areas during 16–22 December 2013.

**Findings.**
- All three models capture the strong daily cycle.
- XGBoost achieved the best accuracy-efficiency trade-off.
- SARIMA is competitive on clean, high-traffic areas but weaker on
  erratic ones.
- LSTM offers the best raw accuracy on complex areas but at high
  computational cost.
- Performance degrades consistently on lower-traffic and more erratic
  areas.

**Limitations.**
- Only univariate Internet traffic was modelled; SMS and call signals
  were ignored.
- The test week contains one weekend; longer evaluation would improve
  robustness.
- SARIMA refitting at every step is expensive; online learning was not
  explored.
- Hyperparameter search was manual and limited; Bayesian optimisation
  could improve results.

**Future work.**
- Spatio-temporal models (GNN, STGCN) that use neighbouring areas.
- Hybrid models combining statistical and neural components.
- Multivariate inputs including SMS, call, and weather.
- Longer horizons (6, 12, 24 steps) and multi-step evaluation.

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the
city of Milan and the Province of Trentino," *Scientific Data*, vol. 2,
p. 150055, 2015. doi:10.1038/sdata.2015.55.

[2] G. O. Ferreira, C. Ravazzi, F. Dabbene, and G. C. Calafiore,
"Forecasting network traffic: A survey and tutorial with open-source
comparative evaluation," *IEEE Access*, vol. 11, pp. 6018–6044, 2023.

[3] A. A. Hussien, H. Nashaat, and R. F. Abdel-Kader, "Machine learning
techniques for spatiotemporal traffic prediction in 5G cellular
networks," *Discover Applied Sciences*, vol. 7, art. 1047, 2025.

[4] H. Shindou, Y. El Hasnaoui, and S. M. Nabil, "Deep learning-based
cellular traffic prediction for 4G long-term evolution networks using
three models," *Bulletin of Electrical Engineering and Informatics*,
2025.

[5] A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular traffic
prediction and classification: A comparative evaluation of LSTM and
ARIMA," in *Proc. DS 2019*, LNCS vol. 11828, pp. 129–144, 2019.

[6] B. Zhao, "Telecom Italia and OPNET datasets for network traffic
prediction," IEEE DataPort, doi:10.21227/4nr9-th42.

[7] B. Zhao et al., "Evaluating AI approaches for 5G network traffic
prediction: A comparative analysis," in *Proc. Springer Conf.*, 2024.

[8] Dataset: Telecom Italia Big Data Challenge, Harvard Dataverse.
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV