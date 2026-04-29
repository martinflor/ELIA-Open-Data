import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class BlockStack:
    prices: np.ndarray
    volumes: np.ndarray
    cumulative: np.ndarray
    total_volume: float
    clearing_price: float


@dataclass
class OfferStack:
    prices: np.ndarray
    volumes: np.ndarray
    cumulative: np.ndarray
    required_volume: float
    orig_clearing_price: float
    computed_clearing_price: float


def build_capacity_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 4-hour period start timestamps from deliverydate and capacitybiddeliveryperiod.

    Expects columns: deliverydate (YYYY-MM-DD) and capacitybiddeliveryperiod like '0 - 4'.
    Returns DataFrame indexed by period_start (DatetimeIndex).
    """
    if df is None or df.empty:
        return df
    tmp = df.copy()
    tmp["deliverydate"] = pd.to_datetime(tmp["deliverydate"], errors="coerce").dt.date

    def parse_bounds(s: str) -> Tuple[int, int]:
        try:
            parts = str(s).replace("\u2212", "-").split("-")
            start_h = int(parts[0].strip())
            end_h = int(parts[1].strip())
            return start_h, end_h
        except Exception:
            return 0, 4

    bounds = tmp["capacitybiddeliveryperiod"].apply(parse_bounds)
    tmp["start_hour"] = bounds.apply(lambda x: x[0])
    tmp["period_start_dt"] = tmp.apply(
        lambda r: datetime.combine(r["deliverydate"], datetime.min.time())
        + timedelta(hours=int(r["start_hour"])),
        axis=1,
    )
    tmp = tmp.drop(columns=["start_hour"])
    tmp = tmp.sort_values("period_start_dt")
    tmp.set_index(pd.to_datetime(tmp["period_start_dt"]), inplace=True)
    tmp.index.name = "period_start"
    tmp = tmp.drop(columns=["period_start_dt"])
    return tmp


def _compute_clearing_price(prices: np.ndarray, volumes: np.ndarray, required_volume: float) -> float:
    if required_volume <= 0 or prices.size == 0:
        return float("nan")
    cumulative = np.cumsum(volumes)
    idx = int(np.searchsorted(cumulative, required_volume, side="left"))
    if idx >= len(prices):
        return float(prices[-1])
    return float(prices[idx])


def _prepare_block_stack(df_block: pd.DataFrame, price_col: str, volume_col: str) -> BlockStack | None:
    df_block = df_block[[price_col, volume_col]].dropna()
    df_block = df_block[df_block[volume_col] > 0]
    if df_block.empty:
        return None

    grouped = df_block.groupby(price_col, as_index=False)[volume_col].sum()
    grouped = grouped.sort_values(price_col)

    prices = grouped[price_col].to_numpy(dtype=float)
    volumes = grouped[volume_col].to_numpy(dtype=float)
    cumulative = np.cumsum(volumes)
    total_volume = float(cumulative[-1])
    clearing_price = float(prices[-1])
    return BlockStack(
        prices=prices,
        volumes=volumes,
        cumulative=cumulative,
        total_volume=total_volume,
        clearing_price=clearing_price,
    )


def _prepare_offer_stack(
    df_block: pd.DataFrame,
    price_col: str,
    offered_volume_col: str,
    awarded_volume_col: str,
) -> OfferStack | None:
    df_block = df_block[[price_col, offered_volume_col, awarded_volume_col]].copy()
    df_block = df_block.dropna(subset=[price_col])

    # Prefer offered volume, fallback to awarded when offered is missing
    df_block["offer_volume"] = df_block[offered_volume_col]
    missing_offer = df_block["offer_volume"].isna()
    df_block.loc[missing_offer, "offer_volume"] = df_block.loc[missing_offer, awarded_volume_col]
    df_block = df_block.dropna(subset=["offer_volume"])
    df_block = df_block[df_block["offer_volume"] > 0]

    if df_block.empty:
        return None

    grouped = df_block.groupby(price_col, as_index=False)["offer_volume"].sum()
    grouped = grouped.sort_values(price_col)

    prices = grouped[price_col].to_numpy(dtype=float)
    volumes = grouped["offer_volume"].to_numpy(dtype=float)
    cumulative = np.cumsum(volumes)

    required_volume = float(df_block[awarded_volume_col].fillna(0).sum())
    if required_volume <= 0:
        return None

    accepted = df_block[df_block[awarded_volume_col].fillna(0) > 0]
    orig_clearing_price = float(accepted[price_col].max()) if not accepted.empty else float("nan")

    computed_clearing_price = _compute_clearing_price(prices, volumes, required_volume)

    return OfferStack(
        prices=prices,
        volumes=volumes,
        cumulative=cumulative,
        required_volume=required_volume,
        orig_clearing_price=orig_clearing_price,
        computed_clearing_price=computed_clearing_price,
    )


def prepare_block_stacks(
    cap_df: pd.DataFrame, direction: str
) -> Tuple[Dict[pd.Timestamp, BlockStack], str, str]:
    """Prepare price-volume stacks per 4h block from accepted capacity bids."""
    if direction not in {"Up", "Down"}:
        raise ValueError("direction must be 'Up' or 'Down'")
    price_col = "priceupmwh" if direction == "Up" else "pricedownmwh"
    volume_col = "afrrawardedvolumeupmw" if direction == "Up" else "afrrawardedvolumedownmw"

    if not isinstance(cap_df.index, pd.DatetimeIndex):
        raise ValueError("cap_df must be indexed by period_start (DatetimeIndex)")

    stacks: Dict[pd.Timestamp, BlockStack] = {}
    for block, df_block in cap_df.groupby(level=0):
        stack = _prepare_block_stack(df_block, price_col, volume_col)
        if stack is not None:
            stacks[block] = stack
    return stacks, price_col, volume_col


def prepare_offer_stacks(
    cap_df: pd.DataFrame, direction: str
) -> Tuple[Dict[pd.Timestamp, OfferStack], str, str, str]:
    """Prepare full offer stacks per 4h block from all bids."""
    if direction not in {"Up", "Down"}:
        raise ValueError("direction must be 'Up' or 'Down'")
    price_col = "priceupmwh" if direction == "Up" else "pricedownmwh"
    offered_col = "afrrofferedvolumeupmw" if direction == "Up" else "afrrofferedvolumedownmw"
    awarded_col = "afrrawardedvolumeupmw" if direction == "Up" else "afrrawardedvolumedownmw"

    if not isinstance(cap_df.index, pd.DatetimeIndex):
        raise ValueError("cap_df must be indexed by period_start (DatetimeIndex)")

    stacks: Dict[pd.Timestamp, OfferStack] = {}
    for block, df_block in cap_df.groupby(level=0):
        stack = _prepare_offer_stack(df_block, price_col, offered_col, awarded_col)
        if stack is not None:
            stacks[block] = stack
    return stacks, price_col, offered_col, awarded_col


def simulate_bid_for_block(
    stack: BlockStack,
    bid_price: float,
    bid_volume: float,
    tie_policy: str = "after",
) -> Dict[str, float]:
    """Simulate bid acceptance against a single block stack.

    tie_policy: "after" means bids at same price are cleared before ours (conservative).
    """
    if stack is None or stack.total_volume <= 0 or bid_volume <= 0 or math.isnan(bid_price):
        return {
            "accepted_volume": 0.0,
            "acceptance_ratio": 0.0,
            "full_accept": 0.0,
            "partial_accept": 0.0,
            "any_accept": 0.0,
            "new_clearing_price": stack.clearing_price if stack else float("nan"),
            "orig_clearing_price": stack.clearing_price if stack else float("nan"),
            "price_in_clearing": 0.0,
        }

    prices = stack.prices
    cumulative = stack.cumulative
    total = stack.total_volume

    pos_left = int(np.searchsorted(prices, bid_price, side="left"))
    pos_right = int(np.searchsorted(prices, bid_price, side="right"))
    volume_before = float(cumulative[pos_left - 1]) if pos_left > 0 else 0.0
    volume_at_price = (
        float(cumulative[pos_right - 1]) - volume_before if pos_left < pos_right else 0.0
    )
    volume_at_price_before_ours = volume_at_price if tie_policy == "after" else 0.0

    remaining = max(0.0, total - volume_before - volume_at_price_before_ours)
    accepted = min(float(bid_volume), remaining)
    acceptance_ratio = 0.0 if bid_volume <= 0 else accepted / float(bid_volume)

    price_in_clearing = 1.0 if bid_price <= stack.clearing_price else 0.0

    if bid_price > stack.clearing_price:
        new_clearing = stack.clearing_price
    else:
        target = total - float(bid_volume)
        if target <= 0:
            q = float(prices[0])
        else:
            idx = int(np.searchsorted(cumulative, target, side="left"))
            q = float(prices[idx])
        new_clearing = max(float(bid_price), q)

    return {
        "accepted_volume": accepted,
        "acceptance_ratio": acceptance_ratio,
        "full_accept": 1.0 if accepted >= float(bid_volume) - 1e-9 else 0.0,
        "partial_accept": 1.0 if 0.0 < accepted < float(bid_volume) else 0.0,
        "any_accept": 1.0 if accepted > 0.0 else 0.0,
        "new_clearing_price": new_clearing,
        "orig_clearing_price": stack.clearing_price,
        "price_in_clearing": price_in_clearing,
    }


def simulate_bid_for_block_offers(
    stack: OfferStack,
    bid_price: float,
    bid_volume: float,
    tie_policy: str = "after",
) -> Dict[str, float]:
    """Simulate bid acceptance against a full offer stack with required volume."""
    if stack is None or stack.required_volume <= 0 or bid_volume <= 0 or math.isnan(bid_price):
        return {
            "accepted_volume": 0.0,
            "acceptance_ratio": 0.0,
            "full_accept": 0.0,
            "partial_accept": 0.0,
            "any_accept": 0.0,
            "new_clearing_price": stack.orig_clearing_price if stack else float("nan"),
            "orig_clearing_price": stack.orig_clearing_price if stack else float("nan"),
            "price_in_clearing": 0.0,
        }

    prices = stack.prices
    cumulative = stack.cumulative
    required = stack.required_volume

    pos_left = int(np.searchsorted(prices, bid_price, side="left"))
    pos_right = int(np.searchsorted(prices, bid_price, side="right"))
    volume_before = float(cumulative[pos_left - 1]) if pos_left > 0 else 0.0
    volume_at_price = (
        float(cumulative[pos_right - 1]) - volume_before if pos_left < pos_right else 0.0
    )
    volume_at_price_before_ours = volume_at_price if tie_policy == "after" else 0.0

    remaining = max(0.0, required - volume_before - volume_at_price_before_ours)
    accepted = min(float(bid_volume), remaining)
    acceptance_ratio = 0.0 if bid_volume <= 0 else accepted / float(bid_volume)

    # Build new stack with our bid inserted (grouped at same price)
    new_prices = prices.copy()
    new_volumes = stack.volumes.copy()
    if pos_left < len(new_prices) and math.isclose(new_prices[pos_left], bid_price, rel_tol=1e-9, abs_tol=1e-9):
        new_volumes[pos_left] += float(bid_volume)
    else:
        new_prices = np.insert(new_prices, pos_left, bid_price)
        new_volumes = np.insert(new_volumes, pos_left, float(bid_volume))

    new_clearing = _compute_clearing_price(new_prices, new_volumes, required)
    orig_clearing = stack.orig_clearing_price
    price_in_clearing = 1.0 if not math.isnan(orig_clearing) and bid_price <= orig_clearing else 0.0

    return {
        "accepted_volume": accepted,
        "acceptance_ratio": acceptance_ratio,
        "full_accept": 1.0 if accepted >= float(bid_volume) - 1e-9 else 0.0,
        "partial_accept": 1.0 if 0.0 < accepted < float(bid_volume) else 0.0,
        "any_accept": 1.0 if accepted > 0.0 else 0.0,
        "new_clearing_price": new_clearing,
        "orig_clearing_price": orig_clearing,
        "price_in_clearing": price_in_clearing,
    }


def simulate_bid_over_year(
    stacks: Dict[pd.Timestamp, BlockStack],
    bid_price: float,
    bid_volume: float,
    tie_policy: str = "after",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Simulate a single bid across all blocks and return per-block and summary results."""
    rows = []
    for block, stack in stacks.items():
        res = simulate_bid_for_block(stack, bid_price, bid_volume, tie_policy=tie_policy)
        res["block_start"] = block
        res["block_hour"] = int(pd.Timestamp(block).hour)
        rows.append(res)

    df = pd.DataFrame(rows).set_index("block_start").sort_index()
    if df.empty:
        summary = pd.Series(dtype="float64")
        return df, summary

    summary = pd.Series(
        {
            "prob_full_accept": df["full_accept"].mean(),
            "prob_any_accept": df["any_accept"].mean(),
            "avg_acceptance_ratio": df["acceptance_ratio"].mean(),
            "avg_new_clearing_price": df["new_clearing_price"].mean(),
            "avg_orig_clearing_price": df["orig_clearing_price"].mean(),
            "avg_clearing_price_delta": (df["new_clearing_price"] - df["orig_clearing_price"]).mean(),
            "prob_price_in_clearing": df["price_in_clearing"].mean(),
        }
    )
    return df, summary


def acceptance_probability(
    stacks: Dict[pd.Timestamp, BlockStack],
    bid_price: float,
    bid_volume: float,
    tie_policy: str = "after",
    accept_mode: str = "full",
    block_hours: Iterable[int] | None = None,
) -> float:
    """Probability of acceptance across blocks (full or any)."""
    if not stacks:
        return 0.0
    if accept_mode not in {"full", "any"}:
        raise ValueError("accept_mode must be 'full' or 'any'")
    allowed_hours = set(block_hours) if block_hours is not None else None
    accept_vals = []
    for block, stack in stacks.items():
        if allowed_hours is not None and int(pd.Timestamp(block).hour) not in allowed_hours:
            continue
        res = simulate_bid_for_block(stack, bid_price, bid_volume, tie_policy=tie_policy)
        accept_vals.append(res["full_accept"] if accept_mode == "full" else res["any_accept"])
    return float(np.mean(accept_vals)) if accept_vals else 0.0


def acceptance_probability_offers(
    stacks: Dict[pd.Timestamp, OfferStack],
    bid_price: float,
    bid_volume: float,
    tie_policy: str = "after",
    accept_mode: str = "any",
    block_hours: Iterable[int] | None = None,
) -> float:
    """Acceptance probability across blocks using full offer stacks."""
    if not stacks:
        return 0.0
    if accept_mode not in {"full", "any"}:
        raise ValueError("accept_mode must be 'full' or 'any'")
    allowed_hours = set(block_hours) if block_hours is not None else None
    accept_vals = []
    for block, stack in stacks.items():
        if allowed_hours is not None and int(pd.Timestamp(block).hour) not in allowed_hours:
            continue
        res = simulate_bid_for_block_offers(stack, bid_price, bid_volume, tie_policy=tie_policy)
        accept_vals.append(res["full_accept"] if accept_mode == "full" else res["any_accept"])
    return float(np.mean(accept_vals)) if accept_vals else 0.0


def generate_slide2_3_4_examples(
    cap_df: pd.DataFrame,
    direction: str,
    block_start: str,
    out_dir: str = "plotly_exports",
    bid_price: float = 40.0,
    bid_volume: float = 10.0,
    dark_mode: bool = True,
) -> List[str]:
    """Generate HTML plots for slides 2, 3, and 4 showing merit order examples.
    
    Args:
        cap_df: DataFrame with capacity data (indexed by period_start)
        direction: "Up" or "Down"
        block_start: Block start timestamp string (e.g., "2025-01-10 04:00:00")
        out_dir: Output directory for HTML files
        bid_price: Price for the test bid (EUR/MW/h)
        bid_volume: Volume for the test bid (MW)
        dark_mode: Whether to use dark mode styling
        
    Returns:
        List of paths to generated HTML files [slide2_path, slide3_path, slide4_path]
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")
    
    # Prepare stacks
    stacks, price_col, offered_col, awarded_col = prepare_offer_stacks(cap_df, direction)
    if not stacks:
        raise ValueError("No valid stacks could be prepared from capacity data.")
    
    # Convert block_start to Timestamp
    block_ts = pd.to_datetime(block_start)
    stack = stacks.get(block_ts)
    if stack is None:
        raise ValueError(f"No stack found for block_start: {block_start}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Determine if bid is in or out of merit
    clearing_price = stack.orig_clearing_price
    bid_in_merit = not math.isnan(clearing_price) and bid_price <= clearing_price
    
    # Generate the three plots
    paths = []
    
    # Slide 2: Base merit order (no bid)
    slide2_path = os.path.join(out_dir, f"slide2_merit_order_base_{direction.lower()}.html")
    _create_merit_order_plot_html(
        stack=stack,
        block_start=block_ts,
        direction=direction,
        output_path=slide2_path,
        bid_price=None,
        bid_volume=None,
        dark_mode=dark_mode,
    )
    paths.append(slide2_path)
    
    # Slide 3: Merit order with bid (in merit)
    slide3_path = os.path.join(out_dir, f"slide3_merit_order_bid_in_merit_{direction.lower()}.html")
    _create_merit_order_plot_html(
        stack=stack,
        block_start=block_ts,
        direction=direction,
        output_path=slide3_path,
        bid_price=bid_price,
        bid_volume=bid_volume,
        dark_mode=dark_mode,
    )
    paths.append(slide3_path)
    
    # Slide 4: Merit order with bid (out of merit) - use a higher price
    out_of_merit_price = clearing_price + 20.0 if not math.isnan(clearing_price) else bid_price + 50.0
    slide4_path = os.path.join(out_dir, f"slide4_merit_order_bid_out_of_merit_{direction.lower()}.html")
    _create_merit_order_plot_html(
        stack=stack,
        block_start=block_ts,
        direction=direction,
        output_path=slide4_path,
        bid_price=out_of_merit_price,
        bid_volume=bid_volume,
        dark_mode=dark_mode,
    )
    paths.append(slide4_path)
    
    return paths


def generate_monthly_capacity_price_plot(
    cap_df: pd.DataFrame,
    out_path: str = "plotly_exports/monthly_avg_capacity_prices_dark.html",
    dark_mode: bool = True,
) -> str:
    """Generate a dark-themed monthly average aFRR capacity price bar chart (HTML).

    Args:
        cap_df: Capacity DataFrame indexed by period_start (DatetimeIndex),
            with columns ``priceupmwh`` and ``pricedownmwh``.
        out_path: Path of the HTML file to write.
        dark_mode: Whether to use dark styling consistent with other dark plots.

    Returns:
        The path to the generated HTML file.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")

    if not isinstance(cap_df.index, pd.DatetimeIndex):
        raise ValueError("cap_df must be indexed by period_start (DatetimeIndex)")

    for col in ("priceupmwh", "pricedownmwh"):
        if col not in cap_df.columns:
            raise ValueError(f"cap_df must contain column '{col}'")

    tmp = cap_df.copy()
    tmp["year"] = tmp.index.year
    tmp["month"] = tmp.index.month
    tmp["month_name"] = tmp.index.month_name()

    monthly = (
        tmp.groupby(["year", "month", "month_name"])[["priceupmwh", "pricedownmwh"]]
        .mean()
        .reset_index()
        .sort_values(["year", "month"])
    )
    monthly["year_month"] = (
        monthly["year"].astype(str) + "-" + monthly["month_name"]
    )

    # Colors and styling
    if dark_mode:
        bg_color = "#000000"
        grid_color = "#2b2b2b"
        text_color = "#ffffff"
    else:
        bg_color = "#ffffff"
        grid_color = "#e0e0e0"
        text_color = "#000000"

    up_color = "#1f77b4"
    down_color = "#7fb3ff"

    fig = go.Figure()
    fig.add_bar(
        x=monthly["year_month"],
        y=monthly["priceupmwh"],
        name="aFRR capacity UP (EUR/MW/h)",
        marker_color=up_color,
    )
    fig.add_bar(
        x=monthly["year_month"],
        y=monthly["pricedownmwh"],
        name="aFRR capacity DOWN (EUR/MW/h)",
        marker_color=down_color,
    )

    fig.update_layout(
        autosize=True,
        barmode="group",
        bargap=0.2,
        title=dict(
            text="Monthly average aFRR capacity prices",
            font=dict(size=16, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text="year_month",
                font=dict(size=12, color=text_color),
            ),
            tickfont=dict(size=10, color=text_color),
            tickangle=-45,
            gridcolor=grid_color,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(
                text="Average capacity price (EUR/MW/h)",
                font=dict(size=12, color=text_color),
            ),
            tickfont=dict(size=10, color=text_color),
            gridcolor=grid_color,
            showgrid=True,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(
            font=dict(size=10, color=text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        width=None,
        height=520,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(
        out_path,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )
    return out_path


def generate_monthly_capacity_price_plot_from_csv(
    csv_path: str = "monthly average prices of aFRR capacity.csv",
    html_out_path: str = "plotly_exports/monthly_avg_capacity_prices_dark_from_csv.html",
    svg_out_path: str = "outputs/monthly_avg_capacity_prices_dark.svg",
    dark_mode: bool = True,
) -> tuple[str, str]:
    """Generate a dark-themed monthly average aFRR capacity price bar chart from a CSV.

    The CSV is expected to have at least the columns:
    ``year_month``, ``priceupmwh``, ``pricedownmwh``.

    Only data from **December 2024 onwards** is plotted (later dates in the file
    are kept; earlier rows are filtered out).
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"year_month", "priceupmwh", "pricedownmwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file must contain columns: {required_cols}, missing: {missing}")

    # Start from the raw CSV
    monthly = df.copy()

    # Derive year and month number from the `year_month` string so that we can
    # filter from 2024-December onwards.
    # Expected format: "<year>-<MonthName>", e.g. "2024-December"
    parts = monthly["year_month"].astype(str).str.split("-", n=1, expand=True)
    monthly["year"] = parts[0].astype(int)
    monthly["month_name"] = parts[1]
    month_order = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    monthly["month"] = monthly["month_name"].map(month_order)

    # Filter to keep only rows from December 2024 and later
    monthly = monthly[
        (monthly["year"] > 2024)
        | ((monthly["year"] == 2024) & (monthly["month"] >= 12))
    ]

    # Colors and styling (same as generate_monthly_capacity_price_plot)
    if dark_mode:
        bg_color = "#000000"
        grid_color = "#2b2b2b"
        text_color = "#ffffff"
    else:
        bg_color = "#ffffff"
        grid_color = "#e0e0e0"
        text_color = "#000000"

    up_color = "#1f77b4"
    down_color = "#7fb3ff"

    fig = go.Figure()
    fig.add_bar(
        x=monthly["year_month"],
        y=monthly["priceupmwh"],
        name="aFRR capacity UP (EUR/MW/h)",
        marker_color=up_color,
    )
    fig.add_bar(
        x=monthly["year_month"],
        y=monthly["pricedownmwh"],
        name="aFRR capacity DOWN (EUR/MW/h)",
        marker_color=down_color,
    )

    fig.update_layout(
        autosize=True,
        barmode="group",
        bargap=0.2,
        title=dict(
            text="Monthly average aFRR capacity prices",
            font=dict(size=16, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text="year_month",
                font=dict(size=12, color=text_color),
            ),
            tickfont=dict(size=10, color=text_color),
            tickangle=-45,
            gridcolor=grid_color,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(
                text="Average capacity price (EUR/MW/h)",
                font=dict(size=12, color=text_color),
            ),
            tickfont=dict(size=10, color=text_color),
            gridcolor=grid_color,
            showgrid=True,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(
            font=dict(size=10, color=text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        width=None,
        height=520,
    )

    # Ensure output directories exist
    if html_out_path:
        os.makedirs(os.path.dirname(html_out_path), exist_ok=True)
    if svg_out_path:
        os.makedirs(os.path.dirname(svg_out_path), exist_ok=True)

    # Save HTML (interactive)
    if html_out_path:
        fig.write_html(html_out_path)

    # Save SVG (static, high-res) – requires kaleido
    if svg_out_path:
        try:
            fig.write_image(svg_out_path, format="svg")
        except Exception as e:
            raise RuntimeError(
                "Failed to write SVG image. Make sure 'kaleido' is installed "
                "(pip install -r requirements.txt)."
            ) from e

    return html_out_path, svg_out_path


def generate_monthly_capacity_price_year_comparison_from_csv(
    csv_path: str = "monthly average prices of aFRR capacity.csv",
    direction: str = "Up",
    html_out_path: str | None = None,
    svg_out_path: str | None = None,
    dark_mode: bool = True,
) -> tuple[str | None, str | None]:
    """Bar chart per month with one bar per year to compare capacity prices.

    Args:
        csv_path: CSV file with columns ``year_month``, ``priceupmwh``, ``pricedownmwh``.
        direction: ``\"Up\"`` or ``\"Down\"`` to choose which price column to plot.
        html_out_path: Optional HTML path (interactive Plotly).
        svg_out_path: Optional SVG path (static, e.g. for slides).
        dark_mode: Whether to use dark styling consistent with other plots.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required. Install with: pip install plotly")

    if direction not in {"Up", "Down"}:
        raise ValueError("direction must be 'Up' or 'Down'")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"year_month", "priceupmwh", "pricedownmwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV file must contain columns: {required_cols}, missing: {missing}")

    monthly = df.copy()

    # Parse year and month name from year_month
    parts = monthly["year_month"].astype(str).str.split("-", n=1, expand=True)
    monthly["year"] = parts[0].astype(int)
    monthly["month_name"] = parts[1]
    month_order = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    monthly["month"] = monthly["month_name"].map(month_order)

    # Sort by month for nicer left-to-right display
    monthly = monthly.sort_values(["month", "year"])

    # Choose column / labels based on direction
    if direction == "Up":
        val_col = "priceupmwh"
        title_prefix = "aFRR capacity UP"
    else:
        val_col = "pricedownmwh"
        title_prefix = "aFRR capacity DOWN"

    # Colors and styling
    if dark_mode:
        bg_color = "#000000"
        grid_color = "#2b2b2b"
        text_color = "#ffffff"
    else:
        bg_color = "#ffffff"
        grid_color = "#e0e0e0"
        text_color = "#000000"

    # Distinct colors per year (cycle if more years)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
    years = sorted(monthly["year"].unique())

    fig = go.Figure()
    for i, year in enumerate(years):
        year_df = monthly[monthly["year"] == year]
        fig.add_bar(
            x=year_df["month_name"],
            y=year_df[val_col],
            name=str(year),
            marker_color=palette[i % len(palette)],
        )

    fig.update_layout(
        autosize=True,
        barmode="group",
        bargap=0.15,
        title=dict(
            text=f"Monthly average {title_prefix} prices by year",
            font=dict(size=16, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Month", font=dict(size=12, color=text_color)),
            tickfont=dict(size=10, color=text_color),
            categoryorder="array",
            categoryarray=list(month_order.keys()),
            gridcolor=grid_color,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(
                text="Average capacity price (EUR/MW/h)",
                font=dict(size=12, color=text_color),
            ),
            tickfont=dict(size=10, color=text_color),
            gridcolor=grid_color,
            showgrid=True,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(
            title="Year",
            font=dict(size=10, color=text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        width=None,
        height=520,
    )

    # Default output paths if not provided
    if html_out_path is None:
        suffix = "up" if direction == "Up" else "down"
        html_out_path = f"plotly_exports/monthly_avg_capacity_prices_year_comp_{suffix}_dark_from_csv.html"
    if svg_out_path is None:
        suffix = "up" if direction == "Up" else "down"
        svg_out_path = f"outputs/monthly_avg_capacity_prices_year_comp_{suffix}_dark.svg"

    # Ensure output directories exist and save
    if html_out_path:
        os.makedirs(os.path.dirname(html_out_path), exist_ok=True)
        fig.write_html(
            html_out_path,
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True},
        )
    if svg_out_path:
        os.makedirs(os.path.dirname(svg_out_path), exist_ok=True)
        try:
            fig.write_image(svg_out_path, format="svg")
        except Exception as e:
            raise RuntimeError(
                "Failed to write SVG image for year-comparison plot. "
                "Make sure 'kaleido' is installed."
            ) from e

    return html_out_path, svg_out_path


def _create_merit_order_plot_html(
    stack: OfferStack,
    block_start: pd.Timestamp,
    direction: str,
    output_path: str,
    bid_price: float | None = None,
    bid_volume: float | None = None,
    dark_mode: bool = False,
):
    """Create a merit order plot as HTML using plotly."""
    cumulative = stack.cumulative
    prices = stack.prices
    clearing_price = stack.orig_clearing_price
    
    # Build step arrays
    cum_with_zero = np.concatenate(([0.0], cumulative))
    prices_step = np.concatenate(([prices[0]], prices))
    
    # Determine accepted volume
    accepted_volume_at_price = None
    if not math.isnan(clearing_price):
        accepted_volume_at_price = float(
            stack.cumulative[np.searchsorted(stack.prices, clearing_price, side="right") - 1]
        )
    
    # Color scheme
    if dark_mode:
        bg_color = "#000000"
        grid_color = "#2b2b2b"
        text_color = "#ffffff"
        accepted_color = "#dff0d8"
        rejected_color = "#fde0c5"
        merit_line_color = "#ffffff"
        clearing_line_color = "#d62728"
        bid_line_color = "#1f77b4"
        bid_fill_color = "#cfe2f3"
        axis_line_color = "#444444"
    else:
        bg_color = "#ffffff"
        grid_color = "#e0e0e0"
        text_color = "#000000"
        accepted_color = "#dff0d8"
        rejected_color = "#fde0c5"
        merit_line_color = "#000000"
        clearing_line_color = "#d62728"
        bid_line_color = "#1f77b4"
        bid_fill_color = "#cfe2f3"
        axis_line_color = "#444444"
    
    fig = go.Figure()
    
    # Add filled rectangles for accepted/rejected bids
    for i in range(len(prices)):
        x0 = float(cum_with_zero[i])
        x1 = float(cum_with_zero[i + 1])
        y = float(prices_step[i])
        is_accepted = accepted_volume_at_price is not None and x1 <= accepted_volume_at_price
        
        fill_color = accepted_color if is_accepted else rejected_color
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[0, 0, y, y, 0],
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=fill_color, width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
    
    # Add merit order step line
    fig.add_trace(go.Scatter(
        x=cum_with_zero,
        y=prices_step,
        mode="lines",
        line=dict(color=merit_line_color, width=2.5),
        name="Merit order (offer stack)",
        hovertemplate="Volume: %{x:.2f} MW<br>Price: %{y:.2f} EUR/MWh<extra></extra>",
    ))
    
    # Add clearing price vertical line
    if accepted_volume_at_price is not None:
        fig.add_trace(go.Scatter(
            x=[accepted_volume_at_price, accepted_volume_at_price],
            y=[0, float(prices.max())],
            mode="lines",
            line=dict(color=clearing_line_color, width=2.5, dash="dash"),
            name="ELIA last accepted bid volume",
            hovertemplate=f"Volume: {accepted_volume_at_price:.2f} MW<extra></extra>",
        ))
    
    # Add bid if provided
    if bid_price is not None and bid_volume is not None and bid_volume > 0:
        # Find position for bid
        pos = int(np.searchsorted(prices, bid_price, side="left"))
        new_prices = prices.copy()
        new_volumes = stack.volumes.copy()
        same_price = pos < len(new_prices) and np.isclose(new_prices[pos], bid_price, rtol=1e-9, atol=1e-9)
        
        if same_price:
            orig_vol_at_price = float(new_volumes[pos])
            new_volumes[pos] += float(bid_volume)
        else:
            orig_vol_at_price = 0.0
            new_prices = np.insert(new_prices, pos, bid_price)
            new_volumes = np.insert(new_volumes, pos, bid_volume)
        
        new_cumulative = np.cumsum(new_volumes)
        new_cum_with_zero = np.concatenate(([0.0], new_cumulative))
        new_prices_step = np.concatenate(([new_prices[0]], new_prices))
        
        # Add merit order + bid line
        fig.add_trace(go.Scatter(
            x=new_cum_with_zero,
            y=new_prices_step,
            mode="lines",
            line=dict(color=bid_line_color, width=2.2, dash="dash"),
            name="Merit order + our bid",
            hovertemplate="Volume: %{x:.2f} MW<br>Price: %{y:.2f} EUR/MWh<extra></extra>",
        ))
        
        # Highlight our bid rectangle
        cumulative_before_price = float(new_cum_with_zero[pos])
        bid_x0 = cumulative_before_price + orig_vol_at_price
        bid_x1 = bid_x0 + float(bid_volume)
        fig.add_trace(go.Scatter(
            x=[bid_x0, bid_x1, bid_x1, bid_x0, bid_x0],
            y=[0, 0, bid_price, bid_price, 0],
            fill="toself",
            fillcolor=bid_fill_color,
            line=dict(color=bid_fill_color, width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
    
    # Update layout
    title_suffix = ""
    if bid_price is not None:
        if not math.isnan(clearing_price) and bid_price <= clearing_price:
            title_suffix = " (Bid In Merit)"
        else:
            title_suffix = " (Bid Out of Merit)"
    elif accepted_volume_at_price is None:
        title_suffix = " (No Accepted Volume)"
    
    fig.update_layout(
        autosize=True,
        title=dict(
            text=f"Merit Order Example - Block {block_start}{title_suffix}",
            font=dict(size=16, color=text_color),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="Cumulative Offered Volume (MW)", font=dict(size=12, color=text_color)),
            tickfont=dict(size=10, color=text_color),
            gridcolor=grid_color,
            showgrid=True,
            linecolor=axis_line_color,
            showline=True,
        ),
        yaxis=dict(
            title=dict(text="Price (EUR/MW/h)", font=dict(size=12, color=text_color)),
            tickfont=dict(size=10, color=text_color),
            gridcolor=grid_color,
            showgrid=True,
            linecolor=axis_line_color,
            showline=True,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend=dict(
            font=dict(size=10, color=text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        width=None,
        height=520,
    )
    
    # Save as HTML (responsive)
    fig.write_html(
        output_path,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )


if __name__ == "__main__":
    """Convenience CLI: running this file directly will generate the HTML plots.

    Uses `ods125.csv` if present in the current directory; otherwise falls back
    to fetching data from ELIA for a fixed demo range that includes the example
    block used in the slides.
    """
    try:
        import os as _os
        import pandas as _pd
        """

        csv_path = "ods125.csv"
        if _os.path.exists(csv_path):
            print(f"Loading data from {csv_path}...")
            df_raw = _pd.read_csv(csv_path)
        else:
            print("CSV file not found. Fetching data from ELIA API...")
            import elia as _elia

            df_raw = _elia.fetch_afrr_capacity_range_all_chunked(
                "2025-01-01", "2025-01-11"
            )
            if df_raw is None or df_raw.empty:
                raise SystemExit(
                    "No data fetched from API. Provide ods125.csv or check ELIA API."
                )
            print(f"Fetched {len(df_raw)} rows from API")

        print("Building capacity periods...")
        cap_df_cli = build_capacity_periods(df_raw)
        print(f"Built {len(cap_df_cli)} capacity-period rows")

        print("Generating merit-order HTML plots (dark mode)...")
        paths_cli = generate_slide2_3_4_examples(
            cap_df=cap_df_cli,
            direction="Up",
            block_start="2025-01-10 04:00:00",
            out_dir="plotly_exports",
            dark_mode=True,
        )

        print("\nMerit-order HTML plots saved:")
        for p in paths_cli:
            print(f" - {p}")

        """
        print("\nGenerating monthly average capacity price plot (dark mode)...")
        html_path, svg_path = generate_monthly_capacity_price_plot_from_csv()
        print(f" - HTML: {html_path}")
        print(f" - SVG : {svg_path}")
    except Exception as _e:
        print(f"Error while generating plots from bid_acceptance_model.py: {_e}")