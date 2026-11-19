# Can Simple SDEs Describe Equity-Market Time Series?

Educational mini-research by Dimitri Bolt, May 2026. Compares textbook stochastic differential equation benchmarks against daily equity-market prices.

## Research question and methods

- Itô diffusion as a benchmark for daily log returns.
- Geometric Brownian motion as a benchmark for raw log prices.
- Ornstein–Uhlenbeck dynamics for OLS-detrended log prices, with AR(1)-based parameter estimation.
- Autocorrelation decay, conditional-mean diagnostics, rolling volatility, and stochastic simulation.

## Reported findings

The presentation covers 94 large-cap US stocks (January 2023–April 2026) and focuses on NVDA, META and GOOG as illustrative cases. Estimated OU half-lives for those cases are 56.4–57.4 trading days; one-step AR(1) fit R² ranges from 0.976 to 0.984. Rolling volatility varies materially, so constant-volatility SDE benchmarks do not capture all observed behavior. These are educational diagnostic results, not proof of stationarity or a profitable trading strategy.

## Start here

- [Research presentation (PDF)](presentation/OU_slides.pdf)
- [Presentation source](presentation/OU_slides.tex)
- [Study notebook](OU_in_Equity_Markets_Study.ipynb)
- [Dependencies](requirements.txt)

The presentation is the source for the numerical summary above. Results have not been independently rerun as part of this documentation update.
