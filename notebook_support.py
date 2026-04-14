from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from matplotlib.animation import FuncAnimation, PillowWriter
from pandas import DataFrame, Series


def configure_notebook_style() -> None:
    """Apply a consistent visual style for the study notebook."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10


def build_representative_diagnostics(
    log_prices_df: DataFrame,
    detrended_log_prices_df: DataFrame,
    log_returns_df: DataFrame,
    focus_symbols: Sequence[str],
) -> DataFrame:
    """Build a compact diagnostic table for representative-stock selection."""
    diagnostics_rows: list[dict[str, float | int | str]] = []

    for symbol in focus_symbols:
        raw_series = log_prices_df[symbol]
        detrended_series = detrended_log_prices_df[symbol]
        time_index = np.arange(len(raw_series))

        diagnostics_rows.append(
            {
                "symbol": symbol,
                "raw_trend_r2": float(np.corrcoef(time_index, raw_series.values)[0, 1] ** 2),
                "lag1_autocorr_detrended_level": float(detrended_series.autocorr(lag=1)),
                "zero_crossings_detrended_level": int(((detrended_series.shift(1) * detrended_series) < 0).sum()),
                "raw_return_std": float(log_returns_df[symbol].std()),
                "detrended_level_std": float(detrended_series.std()),
            }
        )

    return pd.DataFrame(diagnostics_rows).sort_values(
        by=["raw_trend_r2", "lag1_autocorr_detrended_level"],
        ascending=False,
    )


def plot_raw_vs_detrended(
    log_prices_df: DataFrame,
    detrended_log_prices_df: DataFrame,
    symbols: Sequence[str],
) -> None:
    """Plot raw and detrended log-price levels side by side."""
    fig, axes = plt.subplots(
        nrows=len(symbols),
        ncols=2,
        figsize=(16, 4 * len(symbols)),
        sharex=True,
    )

    if len(symbols) == 1:
        axes = np.array([axes])

    for row_index, symbol in enumerate(symbols):
        axes[row_index, 0].plot(log_prices_df.index, log_prices_df[symbol], color="steelblue", linewidth=1.8)
        axes[row_index, 0].set_title(f"{symbol}: raw log-prices")
        axes[row_index, 0].set_ylabel("log-price")

        axes[row_index, 1].plot(
            detrended_log_prices_df.index,
            detrended_log_prices_df[symbol],
            color="darkorange",
            linewidth=1.8,
        )
        axes[row_index, 1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[row_index, 1].set_title(f"{symbol}: detrended log-prices")
        axes[row_index, 1].set_ylabel("detrended log-price")

    axes[-1, 0].set_xlabel("date")
    axes[-1, 1].set_xlabel("date")
    plt.tight_layout()
    plt.show()


def plot_return_distributions(
    log_returns_df: DataFrame,
    detrended_returns_df: DataFrame,
    symbols: Sequence[str],
) -> None:
    """Plot raw and detrended return distributions for selected symbols."""
    fig, axes = plt.subplots(
        nrows=len(symbols),
        ncols=2,
        figsize=(16, 4 * len(symbols)),
    )

    if len(symbols) == 1:
        axes = np.array([axes])

    for row_index, symbol in enumerate(symbols):
        sns.histplot(log_returns_df[symbol], kde=True, ax=axes[row_index, 0], color="steelblue")
        axes[row_index, 0].axvline(log_returns_df[symbol].mean(), color="black", linestyle="--", linewidth=1.0)
        axes[row_index, 0].set_title(f"{symbol}: raw log-returns")
        axes[row_index, 0].set_xlabel("return")

        sns.histplot(detrended_returns_df[symbol], kde=True, ax=axes[row_index, 1], color="darkorange")
        axes[row_index, 1].axvline(
            detrended_returns_df[symbol].mean(),
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axes[row_index, 1].set_title(f"{symbol}: detrended log-returns")
        axes[row_index, 1].set_xlabel("return")

    plt.tight_layout()
    plt.show()


def plot_rolling_diagnostics(
    log_returns_df: DataFrame,
    symbols: Sequence[str],
    rolling_window: int = 21,
) -> None:
    """Plot rolling mean and rolling volatility for raw log-returns."""
    fig, axes = plt.subplots(
        nrows=len(symbols),
        ncols=2,
        figsize=(16, 4 * len(symbols)),
        sharex=True,
    )

    if len(symbols) == 1:
        axes = np.array([axes])

    for row_index, symbol in enumerate(symbols):
        rolling_mean = log_returns_df[symbol].rolling(rolling_window).mean()
        rolling_volatility = log_returns_df[symbol].rolling(rolling_window).std()

        axes[row_index, 0].plot(rolling_mean.index, rolling_mean, color="steelblue", linewidth=1.6)
        axes[row_index, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[row_index, 0].set_title(f"{symbol}: {rolling_window}-day rolling mean of raw log-returns")
        axes[row_index, 0].set_ylabel("rolling mean")

        axes[row_index, 1].plot(rolling_volatility.index, rolling_volatility, color="firebrick", linewidth=1.6)
        axes[row_index, 1].set_title(f"{symbol}: {rolling_window}-day rolling volatility of raw log-returns")
        axes[row_index, 1].set_ylabel("rolling std")

    axes[-1, 0].set_xlabel("date")
    axes[-1, 1].set_xlabel("date")
    plt.tight_layout()
    plt.show()


def compute_autocorrelation_table(
    detrended_log_prices_df: DataFrame,
    symbols: Sequence[str],
    max_lag: int = 20,
) -> DataFrame:
    """Compute lag autocorrelations for detrended log-prices."""
    autocorrelation_df = pd.DataFrame(index=range(1, max_lag + 1))

    for symbol in symbols:
        autocorrelation_df[symbol] = [
            detrended_log_prices_df[symbol].autocorr(lag=lag)
            for lag in autocorrelation_df.index
        ]

    return autocorrelation_df


def plot_autocorrelation_table(autocorrelation_df: DataFrame) -> None:
    """Visualize lag autocorrelation decay."""
    ax = autocorrelation_df.plot(marker="o", linewidth=1.8, figsize=(12, 6))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Lag autocorrelation of detrended log-prices")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    plt.tight_layout()
    plt.show()


def plot_detrended_levels_with_rolling_mean(
    detrended_log_prices_df: DataFrame,
    symbols: Sequence[str],
    window: int = 63,
) -> None:
    """Plot detrended levels together with a rolling mean."""
    fig, axes = plt.subplots(
        nrows=len(symbols),
        ncols=1,
        figsize=(14, 4 * len(symbols)),
        sharex=True,
    )

    if len(symbols) == 1:
        axes = [axes]

    for row_index, symbol in enumerate(symbols):
        detrended_series = detrended_log_prices_df[symbol]
        rolling_level_mean = detrended_series.rolling(window).mean()

        axes[row_index].plot(
            detrended_series.index,
            detrended_series,
            color="darkorange",
            linewidth=1.2,
            alpha=0.8,
            label="detrended level",
        )
        axes[row_index].plot(
            rolling_level_mean.index,
            rolling_level_mean,
            color="navy",
            linewidth=2.0,
            label=f"{window}-day rolling mean",
        )
        axes[row_index].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axes[row_index].set_title(f"{symbol}: detrended level and {window}-day rolling mean")
        axes[row_index].set_ylabel("detrended log-price")
        axes[row_index].legend(loc="upper right")

    axes[-1].set_xlabel("date")
    plt.tight_layout()
    plt.show()


def fit_simple_ou_parameters(
    detrended_log_prices_df: DataFrame,
    symbols: Sequence[str],
) -> DataFrame:
    """Estimate a simple AR(1)-style proxy for OU mean reversion."""
    parameter_rows: list[dict[str, float | str]] = []

    for symbol in symbols:
        detrended_series = detrended_log_prices_df[symbol].dropna()
        x_prev = detrended_series.shift(1).dropna()
        x_curr = detrended_series.loc[x_prev.index]

        design_matrix = np.column_stack([np.ones(len(x_prev)), x_prev.values])
        alpha_hat, phi_hat = np.linalg.lstsq(design_matrix, x_curr.values, rcond=None)[0]
        residuals = x_curr.values - (alpha_hat + phi_hat * x_prev.values)
        sigma_hat = residuals.std(ddof=1)

        if 0 < phi_hat < 1:
            theta_hat = -np.log(phi_hat)
            half_life_days = np.log(2) / theta_hat
        else:
            theta_hat = np.nan
            half_life_days = np.nan

        if abs(1 - phi_hat) > 1e-8:
            long_run_mean_hat = alpha_hat / (1 - phi_hat)
        else:
            long_run_mean_hat = np.nan

        parameter_rows.append(
            {
                "symbol": symbol,
                "alpha_hat": float(alpha_hat),
                "phi_hat": float(phi_hat),
                "theta_hat": float(theta_hat) if np.isfinite(theta_hat) else np.nan,
                "half_life_days": float(half_life_days) if np.isfinite(half_life_days) else np.nan,
                "long_run_mean_hat": float(long_run_mean_hat) if np.isfinite(long_run_mean_hat) else np.nan,
                "sigma_hat": float(sigma_hat),
            }
        )

    return pd.DataFrame(parameter_rows)


def plot_simulated_model_paths(
    brownian_time: np.ndarray,
    brownian_paths: np.ndarray,
    gbm_time: np.ndarray,
    gbm_paths: np.ndarray,
    ou_time: np.ndarray,
    ou_paths: np.ndarray,
    ou_mean_level: float,
    ou_reference_symbol: str,
) -> None:
    """Plot benchmark paths for Brownian motion, GBM, and OU."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    for path in brownian_paths:
        axes[0].plot(brownian_time, path, linewidth=1.2, alpha=0.85)
    axes[0].set_title("Simulated Brownian motion paths")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("level")

    for path in gbm_paths:
        axes[1].plot(gbm_time, path, linewidth=1.2, alpha=0.85)
    axes[1].set_title("Simulated geometric Brownian motion paths")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("level")

    for path in ou_paths:
        axes[2].plot(ou_time, path, linewidth=1.2, alpha=0.85)
    axes[2].axhline(ou_mean_level, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_title(f"Simulated OU paths calibrated to {ou_reference_symbol}")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("level")

    plt.tight_layout()
    plt.show()


def plot_empirical_vs_simulated_ou(
    empirical_series: Series,
    simulated_series: Series,
    symbol: str,
) -> None:
    """Compare an empirical detrended path with a simulated OU benchmark."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        empirical_series.index,
        empirical_series,
        color="darkorange",
        linewidth=1.6,
        label=f"Empirical detrended {symbol}",
    )
    ax.plot(
        simulated_series.index,
        simulated_series,
        color="navy",
        linewidth=1.4,
        alpha=0.9,
        label="Simulated OU benchmark",
    )
    ax.axhline(
        float(empirical_series.mean()),
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Empirical mean level",
    )
    ax.set_title(f"Empirical detrended path vs simulated OU benchmark: {symbol}")
    ax.set_xlabel("date")
    ax.set_ylabel("level")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def simulate_brownian_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate standard Brownian paths on an equidistant grid."""
    time_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
    increments = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    paths = np.concatenate([np.zeros((n_paths, 1)), increments.cumsum(axis=1)], axis=1)
    return time_grid, paths


def simulate_gbm_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    x0: float,
    drift: float,
    volatility: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate geometric Brownian motion via the closed-form representation."""
    time_grid, brownian_paths = simulate_brownian_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        rng=rng,
    )
    drift_term = (drift - 0.5 * volatility**2) * time_grid[None, :]
    diffusion_term = volatility * brownian_paths
    paths = x0 * np.exp(drift_term + diffusion_term)
    return time_grid, paths


def simulate_ou_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    x0: float,
    theta: float,
    mean_level: float,
    diffusion_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate OU paths using the exact Gaussian transition."""
    time_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0

    exp_term = np.exp(-theta * dt)
    innovation_std = diffusion_scale * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))

    for step_index in range(1, n_steps + 1):
        noise = rng.standard_normal(n_paths)
        paths[:, step_index] = (
            mean_level
            + (paths[:, step_index - 1] - mean_level) * exp_term
            + innovation_std * noise
        )

    return time_grid, paths


def discrete_residual_std_to_ou_diffusion_scale(
    theta_hat: float,
    sigma_hat: float,
    dt: float = 1.0,
) -> float:
    """Convert discrete AR(1)-style residual std to a continuous-time OU diffusion scale."""
    denominator = 1.0 - np.exp(-2.0 * theta_hat * dt)
    return sigma_hat * np.sqrt((2.0 * theta_hat) / denominator)


def animation_output_paths(
    output_dir: str | Path = "animation_outputs",
) -> dict[str, Path]:
    output_root = Path(output_dir)
    return {
        "brownian": output_root / "brownian_motion.gif",
        "ou": output_root / "ou_process.gif",
    }


def save_path_animation(
    time_grid: np.ndarray,
    paths: np.ndarray,
    title: str,
    output_path: str | Path,
    ylabel: str,
    xlabel: str = "time",
    path_label_prefix: str = "Path",
    caption_text: str | None = None,
    reference_level: float | None = None,
    reference_label: str | None = None,
    fps: int = 15,
) -> Path:
    """Save an annotated line animation to GIF."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, paths.shape[0]))
    lines = [
        ax.plot([], [], color=color, linewidth=1.8, label=f"{path_label_prefix} {index + 1}")[0]
        for index, color in enumerate(colors)
    ]

    ax.set_xlim(time_grid[0], time_grid[-1])
    y_min = float(paths.min())
    y_max = float(paths.max())
    padding = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    reference_artist = None
    if reference_level is not None:
        reference_artist = ax.axhline(
            reference_level,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=reference_label or "reference level",
        )

    if caption_text is None:
        caption_text = f"Each colored line denotes one independently simulated path ({paths.shape[0]} paths)."

    frame_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.text(0.02, 0.02, caption_text, ha="left", va="bottom", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)
    fig.subplots_adjust(right=0.78, bottom=0.17)

    def update(frame_index: int):
        for line_index, line in enumerate(lines):
            line.set_data(time_grid[: frame_index + 1], paths[line_index, : frame_index + 1])
        frame_text.set_text(f"Frame {frame_index + 1}/{len(time_grid)}")
        artists = [*lines, frame_text]
        if reference_artist is not None:
            artists.append(reference_artist)
        return artists

    animation = FuncAnimation(fig, update, frames=len(time_grid), interval=60, blit=True)
    animation.save(target_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return target_path


def ensure_animation_assets(
    brownian_time: np.ndarray,
    brownian_paths: np.ndarray,
    ou_time: np.ndarray,
    ou_paths: np.ndarray,
    ou_mean_level: float,
    ou_reference_symbol: str,
    regenerate: bool = False,
    output_dir: str | Path = "animation_outputs",
) -> dict[str, Path]:
    """Return animation paths and optionally regenerate the GIF assets."""
    output_paths = animation_output_paths(output_dir=output_dir)

    if regenerate:
        save_path_animation(
            time_grid=brownian_time,
            paths=brownian_paths[:5],
            title="Brownian motion animation",
            output_path=output_paths["brownian"],
            ylabel="level",
            xlabel="time",
            path_label_prefix="Brownian path",
            caption_text="Each colored line denotes one independently simulated Brownian path starting at 0.",
        )
        save_path_animation(
            time_grid=ou_time,
            paths=ou_paths[:5],
            title=f"OU animation calibrated to {ou_reference_symbol}",
            output_path=output_paths["ou"],
            ylabel="level",
            xlabel="step",
            path_label_prefix="OU path",
            caption_text=(
                "Each colored line denotes one independently simulated OU path; "
                "the dashed line marks the estimated long-run mean."
            ),
            reference_level=ou_mean_level,
            reference_label="long-run mean",
        )

    return output_paths


def display_available_animations(animation_paths: dict[str, Path]) -> None:
    """Display GIF files that already exist on disk."""
    descriptions = {
        "brownian": "Brownian GIF: each colored line is one independently simulated Brownian path.",
        "ou": "OU GIF: each colored line is one independently simulated OU path; the dashed line is the long-run level.",
    }
    for label, path in animation_paths.items():
        print(f"{label}: {path}")
        if label in descriptions:
            print(descriptions[label])
        if path.exists():
            display(Image(filename=str(path)))
        else:
            print("file not found")
