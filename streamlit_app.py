import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
import pytz
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple
import importlib

# Local module import (avoid symbol import to be resilient to partial init errors)
import elia
from pandas import Timestamp


@st.cache_data(show_spinner=False)
def load_afrr_prices_for_range(start_date: date, end_date: date) -> pd.DataFrame:
    """Load aFRR prices for the full date range via a single paginated API request.

    Returns a DataFrame indexed by datetime with columns including 'afrrpriceup' and 'afrrpricedown'.
    """
    # Ensure we have latest version if file changed during dev session
    try:
        importlib.reload(elia)
    except Exception:
        pass

    # Prefer chunked range fetching to avoid API result size limits
    try:
        df = elia.fetch_afrr_energy_price_range_chunked(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    except Exception:
        df = elia.fetch_afrr_energy_price_range(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return pd.DataFrame(columns=["afrrpriceup", "afrrpricedown"]).astype({"afrrpriceup": "float64", "afrrpricedown": "float64"})

    # Normalize columns if missing (safety)
    for col in ["afrrpriceup", "afrrpricedown"]:
        if col not in df.columns:
            df[col] = pd.NA
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["ts"] = tmp.index
    tmp["date"] = tmp["ts"].dt.date
    tmp["year"] = tmp["ts"].dt.year
    tmp["month"] = tmp["ts"].dt.month
    tmp["month_name"] = tmp["ts"].dt.month_name()
    tmp["day"] = tmp["ts"].dt.day
    tmp["weekday"] = tmp["ts"].dt.day_name()
    tmp["hour"] = tmp["ts"].dt.hour
    return tmp


def _iter_month_ranges(start_date: date, end_date: date) -> List[Tuple[date, date]]:
    """Yield (month_start, month_end) tuples covering [start_date, end_date]."""
    ranges: List[Tuple[date, date]] = []
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        # month end: next month first day minus one day
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = min(next_month - timedelta(days=1), end_date)
        month_start = max(cursor, start_date)
        ranges.append((month_start, month_end))
        cursor = next_month
    return ranges


def fetch_energy_with_progress(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch energy data month-by-month with a visible progress bar."""
    month_ranges = _iter_month_ranges(start_date, end_date)
    progress = st.progress(0, text="Fetching aFRR energy data...")
    status = st.empty()
    frames: List[pd.DataFrame] = []
    total = len(month_ranges)
    for i, (m_start, m_end) in enumerate(month_ranges, start=1):
        status.write(f"Energy: fetching {m_start} to {m_end} ({i}/{total})")
        try:
            df_part = elia.fetch_afrr_energy_price_range(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            if df_part is not None and not df_part.empty:
                frames.append(df_part)
        except Exception as e:
            status.warning(f"Energy fetch failed for {m_start}–{m_end}: {e}")
        progress.progress(i / total, text=f"Fetching aFRR energy data... ({i}/{total})")
    status.empty()
    progress.empty()
    if not frames:
        return pd.DataFrame(columns=["afrrpriceup", "afrrpricedown"]).astype({"afrrpriceup": "float64", "afrrpricedown": "float64"})
    df = pd.concat(frames, axis=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Ensure required columns exist
    for col in ["afrrpriceup", "afrrpricedown"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def fetch_capacity_with_progress(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch capacity data month-by-month with a visible progress bar."""
    month_ranges = _iter_month_ranges(start_date, end_date)
    progress = st.progress(0, text="Fetching aFRR capacity data...")
    status = st.empty()
    frames: List[pd.DataFrame] = []
    total = len(month_ranges)
    for i, (m_start, m_end) in enumerate(month_ranges, start=1):
        status.write(f"Capacity: fetching {m_start} to {m_end} ({i}/{total})")
        try:
            part = elia.fetch_afrr_capacity_range(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            if part is not None and not part.empty:
                frames.append(part)
        except Exception as e:
            status.warning(f"Capacity fetch failed for {m_start}–{m_end}: {e}")
        progress.progress(i / total, text=f"Fetching aFRR capacity data... ({i}/{total})")
    status.empty()
    progress.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).reset_index(drop=True)


def build_capacity_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 4-hour period start/end timestamps from deliverydate and capacitybiddeliveryperiod.

    Expects columns: deliverydate (YYYY-MM-DD), capacitybiddeliveryperiod like '0 - 4', '4 - 8', etc.
    Returns DataFrame with index=datetime period_start and columns including period_end, prices, etc.
    """
    if df is None or df.empty:
        return df
    tmp = df.copy()
    # Ensure correct types
    tmp['deliverydate'] = pd.to_datetime(tmp['deliverydate'], errors='coerce').dt.date
    # Parse 'a - b' hours
    def parse_bounds(s: str) -> tuple[int, int]:
        try:
            parts = str(s).replace('\u2212', '-').split('-')  # handle unicode minus
            start_h = int(parts[0].strip())
            end_h = int(parts[1].strip())
            return start_h, end_h
        except Exception:
            return 0, 4
    bounds = tmp['capacitybiddeliveryperiod'].apply(parse_bounds)
    tmp['start_hour'] = bounds.apply(lambda x: x[0])
    tmp['end_hour'] = bounds.apply(lambda x: x[1])
    # Build tz-naive periods in Europe/Brussels local time
    tmp['period_start_dt'] = tmp.apply(lambda r: datetime.combine(r['deliverydate'], datetime.min.time()) + timedelta(hours=int(r['start_hour'])), axis=1)
    tmp['period_end'] = tmp.apply(lambda r: datetime.combine(r['deliverydate'], datetime.min.time()) + timedelta(hours=int(r['end_hour'])), axis=1)
    tmp = tmp.drop(columns=['start_hour', 'end_hour'])
    tmp = tmp.sort_values('period_start_dt')
    # Set index to period_start (avoid column with same name)
    tmp.set_index(pd.to_datetime(tmp['period_start_dt']), inplace=True)
    tmp.index.name = 'period_start'
    tmp = tmp.drop(columns=['period_start_dt'])
    return tmp


def describe_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[["afrrpriceup", "afrrpricedown"]].describe().T


def agg_by_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[["afrrpriceup", "afrrpricedown"]].resample(freq).mean()


def render_time_series(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("No data to plot.")
        return
    fig = px.line(
        df.reset_index(),
        x=df.index.name or "index",
        y=["afrrpriceup", "afrrpricedown"],
        title=title,
        labels={"value": "EUR/MWh", "variable": "Series", "index": "Datetime"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_step_series_with_outliers(
    df_normal: pd.DataFrame,
    df_outliers: pd.DataFrame,
    mode: str,
    title: str
):
    if mode not in {"Exclude outliers", "Only outliers", "Together"}:
        mode = "Together"

    if mode == "Exclude outliers":
        ts = df_normal[["afrrpriceup", "afrrpricedown"]].resample("15T").mean()
        if ts.empty:
            st.info("No data to plot.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts["afrrpriceup"], mode="lines", name="aFRR up (15m)", line_shape="hv"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts["afrrpricedown"], mode="lines", name="aFRR down (15m)", line_shape="hv"))
        fig.update_layout(title=title, yaxis_title="EUR/MWh")
        st.plotly_chart(fig, use_container_width=True)
        return

    if mode == "Only outliers":
        if df_outliers.empty:
            st.info("No outliers to plot.")
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_outliers.index, y=df_outliers["afrrpriceup"], mode="markers", name="Outliers up", marker=dict(color="#d62728", size=6)))
        fig.add_trace(go.Scatter(x=df_outliers.index, y=df_outliers["afrrpricedown"], mode="markers", name="Outliers down", marker=dict(color="#1f77b4", size=6)))
        fig.update_layout(title=title + " (outliers only)", yaxis_title="EUR/MWh")
        st.plotly_chart(fig, use_container_width=True)
        return

    # Together: step line for normal, markers for outliers
    ts = df_normal[["afrrpriceup", "afrrpricedown"]].resample("15T").mean()
    fig = go.Figure()
    if not ts.empty:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["afrrpriceup"], mode="lines", name="aFRR up (15m)", line_shape="hv"))
        fig.add_trace(go.Scatter(x=ts.index, y=ts["afrrpricedown"], mode="lines", name="aFRR down (15m)", line_shape="hv"))
    if not df_outliers.empty:
        fig.add_trace(go.Scatter(x=df_outliers.index, y=df_outliers["afrrpriceup"], mode="markers", name="Outliers up", marker=dict(color="#d62728", size=6)))
        fig.add_trace(go.Scatter(x=df_outliers.index, y=df_outliers["afrrpricedown"], mode="markers", name="Outliers down", marker=dict(color="#1f77b4", size=6)))
    fig.update_layout(title=title, yaxis_title="EUR/MWh")
    st.plotly_chart(fig, use_container_width=True)


def render_distributions(df: pd.DataFrame):
    if df.empty:
        st.info("No data to plot.")
        return
    melted = df.melt(value_vars=["afrrpriceup", "afrrpricedown"], var_name="series", value_name="price")
    fig = px.histogram(melted, x="price", color="series", nbins=100, barmode="overlay", opacity=0.6, title="Distribution of aFRR Prices")
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(df: pd.DataFrame):
    if df.empty:
        st.info("No data to plot.")
        return
    feat = add_time_features(df)
    pivot_up = feat.pivot_table(index="weekday", columns="hour", values="afrrpriceup", aggfunc="mean")
    pivot_down = feat.pivot_table(index="weekday", columns="hour", values="afrrpricedown", aggfunc="mean")

    # Order weekdays
    ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_up = pivot_up.reindex(ordered)
    pivot_down = pivot_down.reindex(ordered)

    fig_up = px.imshow(pivot_up, aspect="auto", color_continuous_scale="RdBu", origin="lower", title="Mean aFRR Up Price by Weekday/Hour (EUR/MWh)")
    fig_down = px.imshow(pivot_down, aspect="auto", color_continuous_scale="RdBu", origin="lower", title="Mean aFRR Down Price by Weekday/Hour (EUR/MWh)")
    st.plotly_chart(fig_up, use_container_width=True)
    st.plotly_chart(fig_down, use_container_width=True)


def render_aggregations(df: pd.DataFrame):
    if df.empty:
        st.info("No data for aggregations.")
        return
    feat = add_time_features(df)
    # Monthly mean
    monthly = feat.groupby(["year", "month_name"], sort=False)[["afrrpriceup", "afrrpricedown"]].mean().reset_index()
    monthly["year_month"] = monthly["year"].astype(str) + "-" + monthly["month_name"]
    fig_m = px.bar(monthly, x="year_month", y=["afrrpriceup", "afrrpricedown"], barmode="group", title="Monthly Mean aFRR Prices")
    st.plotly_chart(fig_m, use_container_width=True)

    # Yearly mean
    yearly = feat.groupby(["year"])[["afrrpriceup", "afrrpricedown"]].mean().reset_index()
    fig_y = px.bar(yearly, x="year", y=["afrrpriceup", "afrrpricedown"], barmode="group", title="Yearly Mean aFRR Prices")
    st.plotly_chart(fig_y, use_container_width=True)


def main():
    st.set_page_config(page_title="ELIA aFRR Price Explorer", layout="wide")
    st.title("ELIA aFRR Price Explorer")
    st.caption("Interactive exploration of aFRR energy and capacity prices from ELIA Open Data.")

    # Sidebar controls
    st.sidebar.header("Controls")
    with st.sidebar.expander("Diagnostics"):
        test_date = st.date_input("Test single-day fetch", value=date(2024, 7, 22), key="diag_date")
        if st.button("Run fetch test", key="diag_btn"):
            with st.spinner("Testing single-day fetch..."):
                try:
                    df_test = elia.fetch_afrr_energy_price(test_date.strftime("%Y-%m-%d"))
                    if df_test is None or df_test.empty:
                        st.warning("No rows returned for the selected test date.")
                    else:
                        cols = [c for c in ["afrrpriceup", "afrrpricedown", "imbalanceprice"] if c in df_test.columns]
                        st.success(f"Fetched {len(df_test)} rows. Showing first 5:")
                        st.dataframe(df_test[cols].head(5))
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

        st.markdown("---")
        rs = st.date_input("Range start (range test)", value=date(2024, 7, 1), key="range_start")
        re = st.date_input("Range end (range test)", value=date(2024, 7, 31), key="range_end")
        if st.button("Run range fetch test", key="range_diag_btn"):
            with st.spinner("Testing range fetch..."):
                try:
                    df_r = elia.fetch_afrr_energy_price_range(rs.strftime("%Y-%m-%d"), re.strftime("%Y-%m-%d"))
                    if df_r is None or df_r.empty:
                        st.warning("No rows returned for the selected range.")
                    else:
                        cols_r = [c for c in ["afrrpriceup", "afrrpricedown", "imbalanceprice"] if c in df_r.columns]
                        nrows = len(df_r)
                        tmin = df_r.index.min()
                        tmax = df_r.index.max()
                        st.success(f"Fetched {nrows} rows from {tmin} to {tmax}.")
                        st.dataframe(df_r[cols_r].head(10))
                except Exception as e:
                    st.error(f"Range fetch failed: {e}")
    today = datetime.now(pytz.timezone("Europe/Brussels")).date()

    default_end = today - timedelta(days=2)  # allow for ELIA delay
    default_start = default_end - timedelta(days=90)

    preset = st.sidebar.selectbox(
        "Preset range",
        ["Custom", "Last 30 days", "Last 90 days", "Last 6 months", "Year to date", "Last calendar year"],
        index=2,
    )

    if preset == "Last 30 days":
        start_date = default_end - timedelta(days=30)
        end_date = default_end
    elif preset == "Last 90 days":
        start_date = default_end - timedelta(days=90)
        end_date = default_end
    elif preset == "Last 6 months":
        start_date = (default_end.replace(day=1) - timedelta(days=1)).replace(day=1)
        # start_date now at first day of previous month; step back 5 more months
        for _ in range(5):
            start_date = (start_date.replace(day=1) - timedelta(days=1)).replace(day=1)
        end_date = default_end
    elif preset == "Year to date":
        start_date = date(default_end.year, 1, 1)
        end_date = default_end
    elif preset == "Last calendar year":
        start_date = date(default_end.year - 1, 1, 1)
        end_date = date(default_end.year - 1, 12, 31)
    else:
        start_date, end_date = st.sidebar.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=date(2020, 1, 1),
            max_value=default_end,
        )

    # Outlier settings (hidden defaults)
    iqr_k = 1.5
    outlier_view = "Together"
    exclude_outliers = False

    st.sidebar.write("\n")
    fetch_button = st.sidebar.button("Fetch data", type="primary")

    # Load from session if available
    df_energy = st.session_state.get("df_energy")
    cap_df_raw = st.session_state.get("cap_df_raw")

    if fetch_button:
        # Fetch energy first with progress
        df_energy = fetch_energy_with_progress(start_date, end_date)
        st.session_state["df_energy"] = df_energy
        st.session_state["energy_range"] = (start_date, end_date)

        # Fetch capacity with progress
        try:
            importlib.reload(elia)
        except Exception:
            pass
        cap_df_raw = fetch_capacity_with_progress(start_date, end_date)
        st.session_state["cap_df_raw"] = cap_df_raw
        st.session_state["capacity_range"] = (start_date, end_date)

    # Build tabs always, using session data when present
    tab1, tab2, tab3 = st.tabs(["aFRR Energy Prices", "aFRR Capacity Prices", "aFRR Capacity Volume"])

    with tab1:
            if df_energy is None or df_energy.empty:
                st.warning("No energy data returned for the selected range.")
            else:
                st.caption("Data source: aFRR energy prices from ELIA Open Data — ods134 (historical) and ods166 (from 2024-05-01).")
                # Confirmation of fetch
                feat = add_time_features(df_energy)
                counts_by_month = feat.groupby(["year", "month_name"]).size().reset_index(name="rows")
                total_rows = len(df_energy)
                st.success(f"Fetched {total_rows} rows across {feat['year'].nunique()} year(s) and {feat['month'].nunique()} month(s).")
                st.dataframe(counts_by_month)

                st.subheader("Raw aFRR Energy Prices")
                st.dataframe(df_energy[["afrrpriceup", "afrrpricedown"]])

                # Outlier analysis
                def iqr_bounds(series: pd.Series, k: float) -> tuple[float, float]:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    return (q1 - k * iqr, q3 + k * iqr)

                up_low, up_high = iqr_bounds(df_energy["afrrpriceup"].dropna(), iqr_k)
                dn_low, dn_high = iqr_bounds(df_energy["afrrpricedown"].dropna(), iqr_k)

                mask_normal = (
                    df_energy["afrrpriceup"].between(up_low, up_high, inclusive="both") &
                    df_energy["afrrpricedown"].between(dn_low, dn_high, inclusive="both")
                )

                outlier_count = int((~mask_normal).sum())
                total_count = int(len(df_energy))
                st.info(f"Outlier filter (IQR k={iqr_k:.1f}): {outlier_count}/{total_count} points flagged as outliers.")

                df_normal = df_energy[mask_normal]
                df_outliers = df_energy[~mask_normal]
                df_used = df_normal if exclude_outliers else df_energy

                # Descriptive statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(describe_prices(df_used))

                # Time series
                st.subheader("Time Series (15-minute aFRR energy price)")
                render_step_series_with_outliers(df_normal=df_normal, df_outliers=df_outliers, mode=outlier_view, title="aFRR energy prices (15-minute)")
                ts = df_normal[["afrrpriceup", "afrrpricedown"]].resample("15T").mean()

                # Distributions
                st.subheader("Distributions")
                render_distributions(df_used)

                # Heatmaps
                st.subheader("Heatmaps")
                render_heatmap(df_used)

                # Aggregations
                st.subheader("Aggregations")
                render_aggregations(df_used)

                # Downloads
                st.subheader("Downloads")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download raw aFRR energy prices (CSV)",
                        data=df_energy.to_csv(index=True).encode("utf-8"),
                        file_name=f"afrr_energy_{start_date}_{end_date}.csv",
                        mime="text/csv",
                    )
                with col2:
                    st.download_button(
                        label="Download 15-min series (CSV)",
                        data=ts.to_csv(index=True).encode("utf-8"),
                        file_name=f"afrr_energy_15min_{start_date}_{end_date}.csv",
                        mime="text/csv",
                    )

    with tab2:
            if cap_df_raw is None or cap_df_raw.empty:
                st.warning("No capacity bids found for the selected range.")
            else:
                st.caption("Data source: aFRR capacity prices and awarded volumes from ELIA Open Data — ods125.")
                cap_df = build_capacity_periods(cap_df_raw)
                # Enrich for aggregations
                cap_df['period_hour'] = cap_df.index.hour
                cap_df['weekday'] = cap_df.index.day_name()
                cap_df['year'] = cap_df.index.year
                cap_df['month'] = cap_df.index.month
                cap_df['month_name'] = cap_df.index.month_name()

                # Overall descriptive stats
                st.subheader("Overall Capacity Price Statistics")
                st.dataframe(cap_df[['priceupmwh', 'pricedownmwh']].describe().T)

                # Average by 4-hour block across all days
                st.subheader("Average price by 4-hour block (across all days)")
                block_avg = cap_df.groupby('period_hour')[['priceupmwh', 'pricedownmwh']].mean().reset_index()
                block_avg['block'] = block_avg['period_hour'].astype(str) + ' - ' + (block_avg['period_hour'] + 4).astype(str)
                st.dataframe(block_avg[['block', 'priceupmwh', 'pricedownmwh']])

                # Monthly averages
                st.subheader("Monthly average capacity prices")
                monthly_cap = cap_df.groupby(['year', 'month_name'])[['priceupmwh', 'pricedownmwh']].mean().reset_index()
                monthly_cap['year_month'] = monthly_cap['year'].astype(str) + '-' + monthly_cap['month_name']
                st.dataframe(monthly_cap[['year_month', 'priceupmwh', 'pricedownmwh']])
                fig_month = px.bar(monthly_cap, x='year_month', y=['priceupmwh', 'pricedownmwh'], barmode='group', title='Monthly average capacity prices')
                st.plotly_chart(fig_month, use_container_width=True)

                # Heatmaps weekday x 4-hour block
                st.subheader("Heatmap: average capacity price by weekday and 4-hour block")
                ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                pivot_up = cap_df.pivot_table(index='weekday', columns='period_hour', values='priceupmwh', aggfunc='mean').reindex(ordered_days)
                pivot_down = cap_df.pivot_table(index='weekday', columns='period_hour', values='pricedownmwh', aggfunc='mean').reindex(ordered_days)
                fig_hu = px.imshow(pivot_up, aspect='auto', color_continuous_scale='RdBu', origin='lower', title='Avg capacity UP by weekday/4h block (EUR/MWh)')
                fig_hd = px.imshow(pivot_down, aspect='auto', color_continuous_scale='RdBu', origin='lower', title='Avg capacity DOWN by weekday/4h block (EUR/MWh)')
                st.plotly_chart(fig_hu, use_container_width=True)
                st.plotly_chart(fig_hd, use_container_width=True)

                # Per-period min/mean/max and step plot
                st.subheader("Time series: per-period min/mean/max (4h step)")
                agg = cap_df.groupby(level=0).agg({
                    'priceupmwh': ['min', 'mean', 'max'],
                    'pricedownmwh': ['min', 'mean', 'max']
                })
                agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
                agg = agg.sort_index()
                st.dataframe(agg.head(200))

                fig = go.Figure()
                for col, name, color in [
                    ('priceupmwh_mean', 'Up avg', '#d62728'),
                    ('priceupmwh_min', 'Up min', '#ff9896'),
                    ('priceupmwh_max', 'Up max', '#a50f15'),
                    ('pricedownmwh_mean', 'Down avg', '#1f77b4'),
                    ('pricedownmwh_min', 'Down min', '#9ecae1'),
                    ('pricedownmwh_max', 'Down max', '#08519c'),
                ]:
                    if col in agg.columns:
                        fig.add_trace(go.Scatter(x=agg.index, y=agg[col], mode='lines', name=name, line_shape='hv', line=dict(color=color)))
                fig.update_layout(title="aFRR capacity prices (accepted bids) - 4h step", yaxis_title="EUR/MWh")
                st.plotly_chart(fig, use_container_width=True)

                # Download
                st.download_button(
                    label="Download capacity stats (CSV)",
                    data=agg.to_csv(index=True).encode('utf-8'),
                    file_name=f"afrr_capacity_stats_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )

                # Empirical acceptance probability model
                st.subheader("Empirical P(accept | bid price, 4h block, direction)")
                st.caption("Uses historical clearing-threshold proxy per block (max accepted pay-as-bid in that block). We plot P(accept) = P(threshold ≥ bid), not the CDF.")

                def compute_clearing_thresholds(df_block: pd.DataFrame, price_col: str) -> pd.Series:
                    # Clearing threshold proxy: maximum accepted price within each 4h block
                    if df_block.empty or price_col not in df_block.columns:
                        return pd.Series(dtype="float64")
                    df_block = df_block.copy()
                    df_block['block_start'] = df_block.index.floor('4H')
                    return df_block.groupby('block_start')[price_col].max().dropna().sort_index()

                # Inputs
                col_in1, col_in2, col_in3 = st.columns(3)
                with col_in1:
                    direction = st.selectbox("Direction", ["Up", "Down"], index=0)
                with col_in2:
                    block_choice = st.selectbox("4h block start hour", ["All blocks", 0, 4, 8, 12, 16, 20], index=0)
                # Determine reasonable price range from data
                sample_prices = (cap_df['priceupmwh'] if direction == 'Up' else cap_df['pricedownmwh']).dropna()
                pmin = float(sample_prices.quantile(0.01)) if not sample_prices.empty else 0.0
                pmax = float(sample_prices.quantile(0.99)) if not sample_prices.empty else 100.0
                default_p = float(sample_prices.median()) if not sample_prices.empty else (pmin + pmax) / 2.0
                with col_in3:
                    bid_price = st.number_input("Your bid price (€/MW/h)", value=round(default_p, 2))

                # Helper to compute probability curve (price -> P(accept)) from thresholds
                def prob_curve_from_thresholds(thresholds_sorted):
                    n = len(thresholds_sorted)
                    if n == 0:
                        return [], []
                    # ECDF y(i) = i/n at x=thresholds[i-1]; P(accept)=1-ECDF
                    x_vals = list(thresholds_sorted)
                    y_vals = [1.0 - (i / n) for i in range(1, n + 1)]
                    return x_vals, y_vals

                price_col = 'priceupmwh' if direction == 'Up' else 'pricedownmwh'
                blocks_to_analyze = [0, 4, 8, 12, 16, 20] if block_choice == "All blocks" else [block_choice]

                # Table of probabilities at the chosen bid
                rows = []
                fig_prob = go.Figure()
                import bisect
                for blk in blocks_to_analyze:
                    cap_block_dir = cap_df[cap_df['period_hour'] == blk]
                    clearing_series = compute_clearing_thresholds(cap_block_dir, price_col)
                    thresholds_sorted = clearing_series.sort_values().values
                    n = len(thresholds_sorted)
                    if n == 0:
                        continue
                    num_leq = bisect.bisect_right(thresholds_sorted, bid_price)
                    prob = max(0.0, min(1.0, 1.0 - (num_leq / n)))
                    rows.append({"block": f"{blk:02d}-{(blk+4)%24:02d}", "P_accept": prob, "n_blocks": n})
                    # Curve
                    x_vals, y_vals = prob_curve_from_thresholds(thresholds_sorted)
                    fig_prob.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"{blk:02d}-{(blk+4)%24:02d}", line_shape='hv'))

                if not rows:
                    st.warning("Not enough historical data to estimate acceptance probability for the selection.")
                else:
                    res_df = pd.DataFrame(rows).sort_values("block")
                    st.dataframe(res_df)
                    # Highlight bid on plot
                    fig_prob.add_vline(x=bid_price, line_dash='dash', line_color='orange', annotation_text='Your bid', annotation_position='top')
                    fig_prob.update_layout(title=f"P(accept) vs Bid price ({direction})", xaxis_title="Bid price (€/MW·h)", yaxis_title="P(accept)")
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # Plot bid price × probability curve to visualize expected acceptance value vs bid
                    fig_val = go.Figure()
                    best_rows = []
                    for blk in blocks_to_analyze:
                        cap_block_dir = cap_df[cap_df['period_hour'] == blk]
                        clearing_series = compute_clearing_thresholds(cap_block_dir, price_col)
                        thresholds_sorted = clearing_series.sort_values().values
                        n = len(thresholds_sorted)
                        if n == 0:
                            continue
                        x_vals, y_vals = prob_curve_from_thresholds(thresholds_sorted)
                        xy_vals = [float(x)*float(y) for x, y in zip(x_vals, y_vals)]
                        fig_val.add_trace(go.Scatter(x=x_vals, y=xy_vals, mode='lines', name=f"{blk:02d}-{(blk+4)%24:02d}", line_shape='hv'))
                        # Best point on curve
                        if xy_vals:
                            idx, best_val = max(enumerate(xy_vals), key=lambda t: t[1])
                            best_price = float(x_vals[idx])
                            best_prob = float(y_vals[idx])
                            best_rows.append({
                                "block": f"{blk:02d}-{(blk+4)%24:02d}",
                                "best_bid_price": best_price,
                                "P_accept_at_best": best_prob,
                                "best_bid_x_prob": best_val,
                                "n_blocks": n
                            })
                    fig_val.update_layout(title=f"Bid × P(accept) vs Bid price ({direction})", xaxis_title="Bid price (€/MW·h)", yaxis_title="Bid × P(accept) (€/MW·h)")
                    st.plotly_chart(fig_val, use_container_width=True)
                    if best_rows:
                        st.subheader("Recommended bid per block (maximizing bid × P(accept))")
                        best_df = pd.DataFrame(best_rows).sort_values("block")
                        st.dataframe(best_df)

    with tab3:
        if cap_df_raw is None or cap_df_raw.empty:
            st.warning("No capacity bids found for the selected range.")
        else:
            st.caption("Data source: aFRR capacity awarded volumes from ELIA Open Data — ods125.")
            cap_df = build_capacity_periods(cap_df_raw)
            # Sum awarded volumes per 4h block timestamp
            vol = cap_df.groupby(level=0).agg({
                'afrrawardedvolumeupmw': 'sum',
                'afrrawardedvolumedownmw': 'sum'
            }).rename(columns={'afrrawardedvolumeupmw': 'awarded_up_mw', 'afrrawardedvolumedownmw': 'awarded_down_mw'})
            st.subheader("Awarded capacity volume per 4h block")
            st.dataframe(vol.head(200))

            # Descriptive statistics
            st.subheader("Descriptive statistics (MW)")
            st.dataframe(vol.describe().T)

            # Monthly averages/sums
            feat_idx = vol.copy()
            feat_idx['year'] = feat_idx.index.year
            feat_idx['month_name'] = feat_idx.index.month_name()
            monthly_vol = feat_idx.groupby(['year', 'month_name'])[['awarded_up_mw', 'awarded_down_mw']].sum().reset_index()
            monthly_vol['year_month'] = monthly_vol['year'].astype(str) + '-' + monthly_vol['month_name']
            st.subheader("Monthly total awarded volumes (MW)")
            st.dataframe(monthly_vol[['year_month', 'awarded_up_mw', 'awarded_down_mw']])
            fig_mv = px.bar(monthly_vol, x='year_month', y=['awarded_up_mw', 'awarded_down_mw'], barmode='group', title='Monthly total awarded volumes (MW)')
            st.plotly_chart(fig_mv, use_container_width=True)

            # Add period_hour and weekday for block-level analysis
            cap_df['period_hour'] = cap_df.index.hour
            cap_df['weekday'] = cap_df.index.day_name()
            st.subheader("Average awarded volume by weekday and 4h block (MW)")
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot_vup = cap_df.pivot_table(index='weekday', columns='period_hour', values='afrrawardedvolumeupmw', aggfunc='sum').reindex(ordered_days)
            pivot_vdown = cap_df.pivot_table(index='weekday', columns='period_hour', values='afrrawardedvolumedownmw', aggfunc='sum').reindex(ordered_days)
            fig_vu = px.imshow(pivot_vup, aspect='auto', origin='lower', title='Awarded UP volume by weekday/4h block (MW)')
            fig_vd = px.imshow(pivot_vdown, aspect='auto', origin='lower', title='Awarded DOWN volume by weekday/4h block (MW)')
            st.plotly_chart(fig_vu, use_container_width=True)
            st.plotly_chart(fig_vd, use_container_width=True)

            # Time series step plot of volumes
            st.subheader("Time series of awarded volumes (4h step)")
            fig_vs = go.Figure()
            fig_vs.add_trace(go.Scatter(x=vol.index, y=vol['awarded_up_mw'], mode='lines', name='Up volume', line_shape='hv'))
            fig_vs.add_trace(go.Scatter(x=vol.index, y=vol['awarded_down_mw'], mode='lines', name='Down volume', line_shape='hv'))
            fig_vs.update_layout(title='Awarded capacity volumes over time', yaxis_title='MW')
            st.plotly_chart(fig_vs, use_container_width=True)

            # Download
            st.download_button(
                label="Download awarded volumes (CSV)",
                data=vol.to_csv(index=True).encode('utf-8'),
                file_name=f"afrr_capacity_awarded_volumes_{start_date}_{end_date}.csv",
                mime="text/csv",
            )
if __name__ == "__main__":
    main()


