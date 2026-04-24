# AGENTS.md: AI Guide for Ornstein-Uhlenbeck in Equity Markets

## Project Overview
Educational mini-research on stochastic differential equations (Ornstein-Uhlenbeck processes) applied to daily equity market data. Single Jupyter notebook deliverable, Python data pipeline supporting the analysis. English notebook with Russian inline comments.

## Architecture & Data Flow
**Core Pipeline** (`log_returns.py`):
- `LogReturns` class: download daily NASDAQ prices → transform to log-prices → returns → detrended versions
- Each stage cached internally via lazy-loading (private `_build_*` methods compute once; public `get_*()` methods return copies)
- Data validation drops symbols with gaps or non-positive prices (tracked separately for debugging)
- Configurable exclusion list for known problematic tickers (e.g., `BRK/A`, `BRK/B` require special handling)

**Visualization & Simulation** (`notebook_support.py`):
- Heavy functions for diagnostics, rolling statistics, autocorrelation, AR(1) parameter fitting
- GIF animation generation (Brownian motion, OU paths) offloaded from notebook to keep it presentation-friendly
- All plotting functions expect `Sequence[str]` symbol lists (work on subsets for clarity)

**Notebook Output**:
- Single `OU_in_Equity_Markets_Study.ipynb` is the deliverable
- Structure: Intro → Theory → Data → Raw vs Detrended Comparison → OU Analysis → Simulations → Conclusion
- ~34 cells, kept light on heavy computation; CSV caches pre-computed outside

## Project-Specific Patterns & Conventions

### 1. **Formula & Mathematical References** (Critical)
- Every display formula (`$$...$$`) must match Øksendal textbook exactly
- Include `\tag{4.1.3}` style annotations referencing chapter/formula/exercise number
- Check LaTeX **twice**: raw markdown source must have single backslashes (`\theta`, not `\\theta`)
- See `NOTEBOOK_FORMULA_RULES.md` for escaping rules when editing `.ipynb` JSON
- References: Ch. 2.2 (Brownian), Ch. 4.1-4.2 (Itô), Ch. 5.1 (GBM), Ch. 5/12.3 (OU mean-reversion)

### 2. **Representative Stocks Strategy**
- Use 1–3 "representative stocks" for detailed plots; aggregate stats for the rest
- Current working set: **NVDA** (strong trend, large amplitude), **META** (growth-style), **GOOG** (cleaner, still expressive)
- Selection criteria: High raw-trend R², strong detrended variance, high lag-1 autocorrelation (≈0.988), interpretable for visual narrative
- Diagnostic table in notebook helps justify stock choices to reader

### 3. **Detrending & OU Interpretation**
- Linear trend removal via sklearn.LinearRegression(X=time_index, y=log_prices) required to unlock OU-like mean reversion
- After detrending: AR(1) fit on residuals gives phi_hat ≈0.988, translates to half-life ≈56-57 days
- OU slow mean-reversion by design—visualizes as gentle oscillation around zero, not tight bounce-back
- Narrative: "OU is more meaningful after detrending"

### 4. **Data Validation & Diagnostics**
- Dropped symbols tracked in two categories: missing history and non-positive prices
- Each LogReturns object computes a `get_summary()` dict showing:
  - Input symbol count → filtered count → retained count after price validation
  - Shape of each pipeline stage (prices, log_prices, detrended_log_prices, etc.)
  - List of dropped symbols and why
- Use this for sanity-checking data quality before analysis

### 5. **Simulation & Animation Design**
- GIF generation separated from notebook: helper functions in `notebook_support.py` called with `regenerate_animation_assets=True/False` flag
- Animations include **embedded captions, legends, frame counters** for clarity (not bare lines)
- Brownian motion: each colored line = one independent path starting at 0
- OU animation: colored lines are independent OU paths; dashed line = estimated long-run mean
- Reference: `simulate_ou_paths()` uses **exact Gaussian transition** physics (not naive Euler), ensures numerics are defensible

### 6. **Notebook Style Guidelines**
- Short markdown blocks → short code cells (usually 2–5 lines) → plots/results
- Avoid heavy computation or long helper code in cells; use notebook_support imports
- All formulas in markdown, not code comments
- Code comments allowed in Russian for clarity on edge cases
- Export-friendly: no large data outputs in notebook cells (use CSV caches)

## Critical Developer Workflows

### Running the Full Pipeline
```python
from log_returns import LogReturns
import pandas as pd

# Load symbol list
df_symbols = pd.read_csv("nasdaq_screener_100.csv")
symbols = df_symbols["Symbol"].astype(str).tolist()

# Create pipeline object with date window
obj = LogReturns(symbols_local=symbols, start_date_local='2023-01-01', end_date_local='2026-04-13')

# Access data (lazy-loaded & cached)
prices = obj.get_prices()
detrended_log_prices = obj.get_detrended_log_prices()
summary = obj.get_summary()  # inspect data quality

# Save to CSV for notebook
obj.save_all_to_csv('.')
```

### Generating Diagnostics for Representative Stock Selection
- Use `build_representative_diagnostics()` to compute: trend R², lag-1 autocorr, zero-crossings, volatility
- Sort by these metrics to identify stocks matching the narrative
- Record chosen stocks in `PROJECT_PROGRESS_LOG.md` for reproducibility

### Testing Before Notebook Execution
- Validate date window has sufficient history (>500 observations recommended for stable AR(1) fit)
- Check that at least 50–100 symbols survive filtering (data quality varies by period)
- Run `log_returns.py` as `__main__` to confirm pipeline produces outputs to `.csv`

## External Dependencies & Quirks
- **yfinance**: Download heavy; includes 20-second timeout and progress bar. Occasionally returns inconsistent column formats (code normalizes this)
- **websockets**: Required by yfinance in some environments; in requirements.txt
- **sklearn LinearRegression**: Fast for multi-target OLS (all detrending in one fit); used for speed in batch context
- **matplotlib FuncAnimation + PillowWriter**: GIF generation; slow for 100+ frames, but keeps output size reasonable

## Integration Points & Common Modifications

**Adding New Detrending Method**:  
In `LogReturns._build_detrended_log_prices()`, replace LinearRegression fit with alternative. Ensure method is deterministic and handles entire panel at once.

**Expanding to Other Markets**:  
`LogReturns` is symbol/date-agnostic; feed any yfinance-compatible tickers and date range. Be prepared for data quality issues (gaps, delists, corporate actions).

**Comparing New OU Models**:  
Current AR(1) fit is meant as **teaching approximation only**. For rigorous model comparison, use dedicated SDEs library; keep notebook narrative simple.

**Updating Formulas from Øksendal**:  
Always retrieve PDF exact text, verify chapter/section numbers, and test LaTeX rendering in Jupyter before committing `.ipynb`.

## Related Reference Files
- `NOTEBOOK_FORMULA_RULES.md`: Technical rules for LaTeX in `.ipynb`
- `PROJECT_PROGRESS_LOG.md`: Dated log of empirical choices (window, stock selection, findings)
- `RESEARCH_REQUIREMENTS_AND_TODO.md`: Detailed scope, narrative structure, and methodology
- `TextBooks/978-3-662-03620-4.pdf`: Øksendal textbook (source for all formula references)

## What This Project Is *Not* Trying To Do
- Provide production-grade parameter calibration or backtesting
- Test statistical hypothesis of OU-model fit (comparison is visual/heuristic)
- Include jump models or advanced volatility clustering
- Build trading strategy or market microstructure model

Emphasize in any explanations that analysis is **educational mini-research** with pedagogical goals, not industry-grade quantitative finance.

