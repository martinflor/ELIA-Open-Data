import os
from datetime import datetime

import numpy as np
import pandas as pd

import elia
from bid_acceptance_model import (
    build_capacity_periods,
    acceptance_probability_offers,
    prepare_offer_stacks,
)


START_DATE = "2025-01-01"
END_DATE = "2025-01-11"
DIR = "Up"  # "Up" or "Down"
PRICE_POINTS = 60
VOLUME_POINTS = 50
PRICE_MAX = 150.0
EXAMPLE_BLOCK_START = "2025-01-10 04:00:00"  # e.g., "2025-01-01 00:00:00"
EXPORT_MERIT_ORDER_DATA = False
SAVE_MERIT_ORDER_PNG = False
OUTPUT_DIR = "outputs"
BID_PRICE = 40.0
BID_VOLUME = 10.0


def _safe_quantiles(series: pd.Series, q_low: float, q_high: float, default_min: float, default_max: float):
    if series is None or series.empty:
        return default_min, default_max
    return float(series.quantile(q_low)), float(series.quantile(q_high))


def _plot_acceptance_surface(grid_df: pd.DataFrame, title: str):
    """Plot 3D surface where z = acceptance percentage."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt

    if grid_df is None or grid_df.empty:
        return

    # Truncate to plot limits so surface doesn't extend beyond axes
    plot_df = grid_df[grid_df["bid_price"] <= PRICE_MAX].copy()
    if plot_df.empty:
        return

    prices = np.sort(plot_df["bid_price"].unique())
    volumes = np.sort(plot_df["bid_volume"].unique())
    pivot = plot_df.pivot(index="bid_volume", columns="bid_price", values="accepted_percent")
    pivot = pivot.reindex(index=volumes, columns=prices)

    X, Y = np.meshgrid(prices, volumes)
    Z = pivot.values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Bid Price (EUR/MW/h)")
    ax.set_ylabel("Bid Volume (MW)")
    ax.set_zlabel("Accepted Blocks (%)")
    ax.set_xlim(0, PRICE_MAX)
    plt.tight_layout()
    plt.show()


def _plot_merit_order_example(stacks, block_start: pd.Timestamp | None = None):
    """Plot merit order curve for a single block with accepted-volume limit."""
    import matplotlib.pyplot as plt

    if not stacks:
        return

    if block_start is None:
        block_start = sorted(stacks.keys())[0]
    elif not isinstance(block_start, pd.Timestamp):
        block_start = pd.to_datetime(block_start)

    stack = stacks.get(block_start)
    if stack is None:
        return

    cumulative = stack.cumulative
    prices = stack.prices
    clearing_price = stack.orig_clearing_price

    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()
    ax.set_facecolor("#000000")

    # Build step arrays so each rectangle aligns with the curve segment
    cum_with_zero = np.concatenate(([0.0], cumulative))
    prices_step = np.concatenate(([prices[0]], prices))

    plt.step(
        cum_with_zero,
        prices_step,
        where="post",
        color="#ffffff",
        linewidth=2.5,
        label="Merit order (offer stack)",
        zorder=3,
    )
    # Red vertical line at volume implied by ELIA last accepted bid price
    accepted_volume_at_price = None
    if not np.isnan(clearing_price):
        accepted_volume_at_price = float(
            stack.cumulative[np.searchsorted(stack.prices, clearing_price, side="right") - 1]
        )
        plt.axvline(
            accepted_volume_at_price,
            color="#d62728",
            linestyle="--",
            linewidth=2.5,
            label="ELIA last accepted bid volume",
            zorder=4,
        )
        # Fill bids: accepted (green) vs not accepted (orange)
        for i in range(len(prices)):
            x0 = float(cum_with_zero[i])
            x1 = float(cum_with_zero[i + 1])
            y = float(prices_step[i])
            is_accepted = x1 <= accepted_volume_at_price
            fill_color = "#dff0d8" if is_accepted else "#fde0c5"
            ax.fill_between(
                [x0, x1],
                [0, 0],
                [y, y],
                color=fill_color,
                alpha=0.85,
                zorder=1,
            )
    else:
        # Fallback: no accepted volume line, fill all as not accepted
        for i in range(len(prices)):
            x0 = float(cum_with_zero[i])
            x1 = float(cum_with_zero[i + 1])
            y = float(prices_step[i])
            ax.fill_between(
                [x0, x1],
                [0, 0],
                [y, y],
                color="#fde0c5",
                alpha=0.85,
                zorder=1,
            )
    # Plot the curve after inserting our bid
    if BID_PRICE is not None and BID_VOLUME is not None and BID_VOLUME > 0:
        pos = int(np.searchsorted(prices, BID_PRICE, side="left"))
        new_prices = prices.copy()
        new_volumes = stack.volumes.copy()
        same_price = pos < len(new_prices) and np.isclose(new_prices[pos], BID_PRICE)
        if same_price:
            orig_vol_at_price = float(new_volumes[pos])
            new_volumes[pos] += float(BID_VOLUME)
        else:
            orig_vol_at_price = 0.0
            new_prices = np.insert(new_prices, pos, BID_PRICE)
            new_volumes = np.insert(new_volumes, pos, BID_VOLUME)

        new_cumulative = np.cumsum(new_volumes)
        new_cum_with_zero = np.concatenate(([0.0], new_cumulative))
        new_prices_step = np.concatenate(([new_prices[0]], new_prices))

        plt.step(
            new_cum_with_zero,
            new_prices_step,
            where="post",
            color="#1f77b4",
            linewidth=2.2,
            linestyle="--",
            label="Merit order + our bid",
            zorder=3,
        )

        # Highlight our bid rectangle at the correct price segment
        cumulative_before_price = float(new_cum_with_zero[pos])
        bid_x0 = cumulative_before_price + orig_vol_at_price
        bid_x1 = bid_x0 + float(BID_VOLUME)
        ax.fill_between(
            [bid_x0, bid_x1],
            [0, 0],
            [BID_PRICE, BID_PRICE],
            color="#cfe2f3",
            alpha=0.9,
            zorder=2,
        )

    plt.title(
        f"Merit Order Example - Block {block_start}",
        fontsize=14,
        weight="bold",
        pad=12,
        color="#000000",
    )
    plt.xlabel("Cumulative Offered Volume (MW)", fontsize=12, color="#000000")
    plt.ylabel("Price (EUR/MW/h)", fontsize=12, color="#000000")

    ax.grid(True, which="major", color="#2b2b2b", linewidth=1, alpha=0.9)
    ax.tick_params(axis="both", labelsize=10, colors="#000000")

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    plt.legend(frameon=False, fontsize=10, labelcolor="#000000")
    plt.tight_layout()
    plt.show()

    # Build export table for the step rectangles
    rows = []
    for i in range(len(prices)):
        x0 = float(cum_with_zero[i])
        x1 = float(cum_with_zero[i + 1])
        y = float(prices_step[i])
        is_accepted = False
        if accepted_volume_at_price is not None:
            is_accepted = x1 <= accepted_volume_at_price
        rows.append(
            {
                "block_start": block_start,
                "price_eur_mwh": y,
                "volume_start_mw": x0,
                "volume_end_mw": x1,
                "volume_width_mw": x1 - x0,
                "accepted": is_accepted,
                "elia_last_accepted_price": clearing_price,
                "elia_last_accepted_volume": accepted_volume_at_price,
            }
        )
    return pd.DataFrame(rows)


def _plot_merit_order_base(stacks, block_start: pd.Timestamp | None = None):
    """Plot merit order curve for a single block without adding our bid."""
    import matplotlib.pyplot as plt

    if not stacks:
        return

    if block_start is None:
        block_start = sorted(stacks.keys())[0]
    elif not isinstance(block_start, pd.Timestamp):
        block_start = pd.to_datetime(block_start)

    stack = stacks.get(block_start)
    if stack is None:
        return

    cumulative = stack.cumulative
    prices = stack.prices
    clearing_price = stack.orig_clearing_price

    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()
    ax.set_facecolor("#000000")

    cum_with_zero = np.concatenate(([0.0], cumulative))
    prices_step = np.concatenate(([prices[0]], prices))

    plt.step(
        cum_with_zero,
        prices_step,
        where="post",
        color="#ffffff",
        linewidth=2.5,
        label="Merit order (offer stack)",
        zorder=3,
    )

    accepted_volume_at_price = None
    if not np.isnan(clearing_price):
        accepted_volume_at_price = float(
            stack.cumulative[np.searchsorted(stack.prices, clearing_price, side="right") - 1]
        )
        plt.axvline(
            accepted_volume_at_price,
            color="#d62728",
            linestyle="--",
            linewidth=2.5,
            label="ELIA last accepted bid volume",
            zorder=4,
        )
        for i in range(len(prices)):
            x0 = float(cum_with_zero[i])
            x1 = float(cum_with_zero[i + 1])
            y = float(prices_step[i])
            is_accepted = x1 <= accepted_volume_at_price
            fill_color = "#dff0d8" if is_accepted else "#fde0c5"
            ax.fill_between(
                [x0, x1],
                [0, 0],
                [y, y],
                color=fill_color,
                alpha=0.85,
                zorder=1,
            )
    else:
        for i in range(len(prices)):
            x0 = float(cum_with_zero[i])
            x1 = float(cum_with_zero[i + 1])
            y = float(prices_step[i])
            ax.fill_between(
                [x0, x1],
                [0, 0],
                [y, y],
                color="#fde0c5",
                alpha=0.85,
                zorder=1,
            )

    plt.title(
        f"Merit Order Example (No Bid) - Block {block_start}",
        fontsize=14,
        weight="bold",
        pad=12,
        color="#000000",
    )
    plt.xlabel("Cumulative Offered Volume (MW)", fontsize=12, color="#000000")
    plt.ylabel("Price (EUR/MW/h)", fontsize=12, color="#000000")

    ax.grid(True, which="major", color="#2b2b2b", linewidth=1, alpha=0.9)
    ax.tick_params(axis="both", labelsize=10, colors="#000000")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    plt.legend(frameon=False, fontsize=10, labelcolor="#000000")
    plt.tight_layout()
    plt.show()






def main():
    print(f"Fetching capacity data from {START_DATE} to {END_DATE} ...")
    cap_raw = elia.fetch_afrr_capacity_range_all_chunked(START_DATE, END_DATE)
    if cap_raw is None or cap_raw.empty:
        raise SystemExit("No capacity data returned for the selected range.")

    cap_df = build_capacity_periods(cap_raw)
    stacks, price_col, offered_col, awarded_col = prepare_offer_stacks(cap_df, DIR)
    if not stacks:
        raise SystemExit("No valid stacks could be prepared from capacity data.")

    prices = cap_df[price_col].dropna()
    volumes = cap_df[offered_col].dropna()
    p_min, p_max = _safe_quantiles(prices, 0.01, 0.99, 0.0, 200.0)
    v_min, v_max = _safe_quantiles(volumes, 0.01, 0.99, 1.0, 100.0)

    price_grid = np.linspace(p_min, p_max, PRICE_POINTS)
    volume_grid = np.linspace(max(1.0, v_min), max(1.0, v_max), VOLUME_POINTS)

    print("Computing 3D acceptance surface (count of accepted blocks)...")
    grid_rows = []
    for price in price_grid:
        for vol in volume_grid:
            prob_accept = acceptance_probability_offers(
                stacks=stacks,
                bid_price=float(price),
                bid_volume=float(vol),
                accept_mode="any",
            )
            accepted_percent = prob_accept * 100.0
            grid_rows.append(
                {
                    "bid_price": float(price),
                    "bid_volume": float(vol),
                    "accepted_percent": float(accepted_percent),
                }
            )
    grid_df = pd.DataFrame(grid_rows)

    try:
        _plot_acceptance_surface(
            grid_df,
            f"Accepted Blocks Count Surface (any accept) - {DIR}",
        )
    except Exception as e:
        print(f"3D plot skipped (matplotlib not available): {e}")

    # Merit order example for a single block (no bid, then with bid)
    try:
        _plot_merit_order_base(stacks, EXAMPLE_BLOCK_START)
        merit_df = _plot_merit_order_example(stacks, EXAMPLE_BLOCK_START)
        if EXPORT_MERIT_ORDER_DATA and merit_df is not None and not merit_df.empty:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            csv_path = os.path.join(OUTPUT_DIR, "merit_order_example.csv")
            merit_df.to_csv(csv_path, index=False)
            try:
                xlsx_path = os.path.join(OUTPUT_DIR, "merit_order_example.xlsx")
                merit_df.to_excel(xlsx_path, index=False)
            except Exception:
                pass
    except Exception as e:
        print(f"Merit order example plot skipped: {e}")




if __name__ == "__main__":
    main()
