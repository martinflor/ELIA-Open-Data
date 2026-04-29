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
    # Ensure index is DatetimeIndex
    if not isinstance(tmp.index, pd.DatetimeIndex):
        tmp.index = pd.to_datetime(tmp.index, errors='coerce')
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
    """Fetch energy data using the chunked fetcher which handles month-by-month internally."""
    progress = st.progress(0, text="Fetching aFRR energy data...")
    status = st.empty()
    
    status.write(f"⏳ Fetching energy from {start_date} to {end_date}...")
    progress.progress(0.5, text="Fetching aFRR energy data...")
    
    try:
        # Use chunked fetcher which handles ods064/ods166 split and month-by-month internally
        df = elia.fetch_afrr_energy_price_range_chunked(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if df is None or df.empty:
            status.empty()
            progress.empty()
            st.warning("No energy data returned.")
            return pd.DataFrame(columns=["afrrpriceup", "afrrpricedown"]).astype({"afrrpriceup": "float64", "afrrpricedown": "float64"})
        
        # Ensure required columns
        for col in ["afrrpriceup", "afrrpricedown"]:
            if col not in df.columns:
                df[col] = pd.NA
        
        status.empty()
        progress.empty()
        
        st.success(f"✅ Fetched {len(df)} rows | {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        status.empty()
        progress.empty()
        st.error(f"❌ Energy fetch failed: {str(e)}")
        return pd.DataFrame(columns=["afrrpriceup", "afrrpricedown"]).astype({"afrrpriceup": "float64", "afrrpricedown": "float64"})


def fetch_capacity_with_progress(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch capacity data month-by-month with a visible progress bar."""
    month_ranges = _iter_month_ranges(start_date, end_date)
    progress = st.progress(0, text="Fetching aFRR capacity data...")
    status = st.empty()
    warning_container = st.container()
    frames: List[pd.DataFrame] = []
    total = len(month_ranges)
    for i, (m_start, m_end) in enumerate(month_ranges, start=1):
        status.write(f"Capacity: fetching {m_start} to {m_end} ({i}/{total})")
        try:
            part = elia.fetch_afrr_capacity_range(m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))
            if part is not None and not part.empty:
                frames.append(part)
                status.write(f"✓ Capacity: {m_start} to {m_end} → {len(part)} rows")
            else:
                with warning_container:
                    st.warning(f"No capacity data returned for {m_start}–{m_end}")
        except Exception as e:
            with warning_container:
                st.error(f"Capacity fetch failed for {m_start}–{m_end}: {e}")
        progress.progress(i / total, text=f"Fetching aFRR capacity data... ({i}/{total})")
    status.empty()
    progress.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).reset_index(drop=True)


def fetch_pv_with_progress(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch photovoltaic production data using the chunked fetcher."""
    progress = st.progress(0, text="Fetching photovoltaic production data...")
    status = st.empty()
    
    status.write(f"⏳ Fetching PV data from {start_date} to {end_date}...")
    progress.progress(0.5, text="Fetching photovoltaic production data...")
    
    try:
        # Ensure we have latest version if file changed during dev session
        try:
            importlib.reload(elia)
        except Exception:
            pass
        
        # Use chunked fetcher to handle month-by-month internally
        df = elia.fetch_photovoltaic_production_range_chunked(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if df is None or df.empty:
            status.empty()
            progress.empty()
            st.warning("No photovoltaic production data returned.")
            return pd.DataFrame()
        
        status.empty()
        progress.empty()
        
        st.success(f"✅ Fetched {len(df)} rows | {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        status.empty()
        progress.empty()
        st.error(f"❌ PV fetch failed: {str(e)}")
        return pd.DataFrame()


def fetch_imbalance_with_progress(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch balancing energy prices (imbalance data) using the chunked fetcher."""
    progress = st.progress(0, text="Fetching imbalance data...")
    status = st.empty()
    
    status.write(f"⏳ Fetching imbalance data from {start_date} to {end_date}...")
    progress.progress(0.5, text="Fetching imbalance data...")
    
    try:
        # Ensure we have latest version if file changed during dev session
        try:
            importlib.reload(elia)
        except Exception:
            pass
        
        # Use chunked fetcher to handle month-by-month internally
        df = elia.fetch_balancing_energy_prices_range_chunked(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if df is None or df.empty:
            status.empty()
            progress.empty()
            st.warning("No imbalance data returned.")
            return pd.DataFrame()
        
        status.empty()
        progress.empty()
        
        st.success(f"✅ Fetched {len(df)} rows | {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        status.empty()
        progress.empty()
        st.error(f"❌ Imbalance fetch failed: {str(e)}")
        return pd.DataFrame()


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
    st.set_page_config(page_title="ELIA Open Data Explorer", layout="wide")
    st.title("ELIA Open Data Explorer")
    st.caption("Interactive exploration of aFRR energy and capacity prices, and photovoltaic production from ELIA Open Data.")

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
        date_range = st.sidebar.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=date(2020, 1, 1),
            max_value=default_end,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            # User selected only one date or cleared selection
            start_date = default_start
            end_date = default_end

    # Outlier settings (hidden defaults)
    iqr_k = 1.5
    outlier_view = "Together"
    exclude_outliers = False

    # Display selected date range
    st.sidebar.info(f"📅 Selected range:\n{start_date} to {end_date}")
    
    st.sidebar.write("\n")
    st.sidebar.subheader("Fetch Data")
    fetch_button = st.sidebar.button("Fetch aFRR data", type="primary")
    fetch_pv_button = st.sidebar.button("Fetch PV data", type="secondary")
    fetch_imbalance_button = st.sidebar.button("Fetch Imbalance data", type="secondary")

    # Check if we're viewing cached data from a different range BEFORE fetching
    cached_range = st.session_state.get("energy_range")
    if not fetch_button and cached_range is not None:
        if cached_range != (start_date, end_date):
            st.sidebar.warning(f"⚠️ Showing cached data from:\n{cached_range[0]} to {cached_range[1]}\n\nClick 'Fetch data' to update.")

    if fetch_button:
        # Fetch energy first with progress
        df_energy = fetch_energy_with_progress(start_date, end_date)
        st.session_state["df_energy"] = df_energy
        st.session_state["energy_range"] = (start_date, end_date)

        # Fetch capacity data with progress
        cap_df_raw = fetch_capacity_with_progress(start_date, end_date)
        st.session_state["cap_df_raw"] = cap_df_raw
        st.session_state["capacity_range"] = (start_date, end_date)

    if fetch_pv_button:
        # Fetch PV data with progress
        df_pv = fetch_pv_with_progress(start_date, end_date)
        st.session_state["df_pv"] = df_pv
        st.session_state["pv_range"] = (start_date, end_date)

    if fetch_imbalance_button:
        # Fetch imbalance data with progress
        df_imbalance = fetch_imbalance_with_progress(start_date, end_date)
        st.session_state["df_imbalance"] = df_imbalance
        st.session_state["imbalance_range"] = (start_date, end_date)

    # Load from session AFTER potential fetch
    df_energy = st.session_state.get("df_energy")
    cap_df_raw = st.session_state.get("cap_df_raw")
    df_pv = st.session_state.get("df_pv")
    df_imbalance = st.session_state.get("df_imbalance")

    # Build tabs always, using session data when present
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["aFRR Energy Prices", "aFRR Capacity Prices", "aFRR Capacity Volume", "Photovoltaic Production", "System Imbalance"])

    with tab1:
            if df_energy is None or df_energy.empty:
                st.warning("No energy data returned for the selected range.")
            else:
                st.caption("Data source: aFRR energy prices from ELIA Open Data — ods064 (historical) and ods166 (from 2024-05-01).")
                # Debug: show date range of data
                st.info(f"📅 Data covers: {df_energy.index.min()} to {df_energy.index.max()} (total: {len(df_energy)} rows)")
                
                # Confirmation of fetch
                feat = add_time_features(df_energy)
                counts_by_month = feat.groupby(["year", "month_name"]).size().reset_index(name="rows").sort_values(["year", "month_name"])
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

                df_normal = df_energy[mask_normal].copy()
                df_outliers = df_energy[~mask_normal].copy()
                df_used = df_normal if exclude_outliers else df_energy
                
                # Ensure DatetimeIndex is preserved (handle timezone-aware indices)
                if not isinstance(df_normal.index, pd.DatetimeIndex):
                    df_normal.index = pd.to_datetime(df_normal.index, utc=True)
                if not df_outliers.empty and not isinstance(df_outliers.index, pd.DatetimeIndex):
                    df_outliers.index = pd.to_datetime(df_outliers.index, utc=True)
                if not isinstance(df_used.index, pd.DatetimeIndex):
                    df_used.index = pd.to_datetime(df_used.index, utc=True)

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

                # ── Negative Price Analysis ──────────────────────────────────────────
                st.subheader("Negative Price Analysis")
                st.caption(
                    "Periods where the aFRR energy price (UP or DOWN) is strictly negative (< 0 €/MWh)."
                )

                neg_analysis = elia.analyze_afrr_negative_prices(df_energy)
                up_r   = neg_analysis.get("up")
                down_r = neg_analysis.get("down")
                summary = neg_analysis.get("summary", {})

                # -- Top-level KPI metrics -------------------------------------------
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                with kpi1:
                    st.metric(
                        "Negative UP periods",
                        up_r["count"] if up_r else "N/A",
                        f"{up_r['pct']} %" if up_r else "",
                    )
                with kpi2:
                    st.metric(
                        "Negative DOWN periods",
                        down_r["count"] if down_r else "N/A",
                        f"{down_r['pct']} %" if down_r else "",
                    )
                with kpi3:
                    st.metric(
                        "Both negative simultaneously",
                        summary.get("both_negative_count", "N/A"),
                    )
                with kpi4:
                    st.metric(
                        "Total periods (UP)",
                        up_r["total"] if up_r else "N/A",
                    )
                with kpi5:
                    st.metric(
                        "Total periods (DOWN)",
                        down_r["total"] if down_r else "N/A",
                    )

                # -- Statistics comparison table -------------------------------------
                st.write("**Negative-price statistics by direction**")
                neg_stats_rows = []
                for label, r in [("UP", up_r), ("DOWN", down_r)]:
                    if r:
                        neg_stats_rows.append({
                            "Direction":  label,
                            "Count":      r["count"],
                            "Total":      r["total"],
                            "Share (%)":  r["pct"],
                            "Min (€/MWh)":  r["min_price"]  if r["count"] > 0 else None,
                            "Max (€/MWh)":  r["max_price"]  if r["count"] > 0 else None,
                            "Mean (€/MWh)": r["mean_price"] if r["count"] > 0 else None,
                        })
                if neg_stats_rows:
                    neg_stats_df = pd.DataFrame(neg_stats_rows).set_index("Direction")
                    st.dataframe(
                        neg_stats_df.style.format(
                            {
                                "Share (%)":    "{:.2f}",
                                "Min (€/MWh)":  "{:.2f}",
                                "Max (€/MWh)":  "{:.2f}",
                                "Mean (€/MWh)": "{:.2f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True,
                    )

                # -- Time-series scatter of negative prices --------------------------
                st.write("**Negative price occurrences over time**")
                fig_neg = go.Figure()
                if up_r and up_r["count"] > 0:
                    neg_up_df = df_energy.loc[up_r["timestamps"], ["afrrpriceup"]]
                    fig_neg.add_trace(
                        go.Scatter(
                            x=neg_up_df.index,
                            y=neg_up_df["afrrpriceup"],
                            mode="markers",
                            name="Negative UP price",
                            marker=dict(color="#d62728", size=5, symbol="circle"),
                        )
                    )
                if down_r and down_r["count"] > 0:
                    neg_dn_df = df_energy.loc[down_r["timestamps"], ["afrrpricedown"]]
                    fig_neg.add_trace(
                        go.Scatter(
                            x=neg_dn_df.index,
                            y=neg_dn_df["afrrpricedown"],
                            mode="markers",
                            name="Negative DOWN price",
                            marker=dict(color="#1f77b4", size=5, symbol="diamond"),
                        )
                    )
                fig_neg.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black",
                    line_width=1,
                    annotation_text="0 €/MWh",
                    annotation_position="top left",
                )
                fig_neg.update_layout(
                    title="aFRR energy prices — negative occurrences",
                    xaxis_title="Datetime",
                    yaxis_title="Price (€/MWh)",
                    hovermode="x unified",
                )
                if (up_r and up_r["count"] > 0) or (down_r and down_r["count"] > 0):
                    st.plotly_chart(fig_neg, use_container_width=True)
                else:
                    st.info("No negative price periods found in the selected range.")

                # -- Monthly breakdown of negative-price occurrences -----------------
                st.write("**Monthly negative-price count — UP vs DOWN**")
                monthly_neg_rows = []
                feat_neg = add_time_features(df_energy)

                for label, col_name, r in [
                    ("UP",   "afrrpriceup",   up_r),
                    ("DOWN", "afrrpricedown", down_r),
                ]:
                    if r and r["count"] > 0:
                        neg_mask_col = feat_neg[col_name] < 0
                        monthly_cnt = (
                            feat_neg[neg_mask_col]
                            .groupby(["year", "month", "month_name"])
                            .size()
                            .reset_index(name="neg_count")
                        )
                        monthly_cnt["direction"] = label
                        monthly_cnt["year_month"] = (
                            monthly_cnt["year"].astype(str)
                            + "-"
                            + monthly_cnt["month_name"]
                        )
                        monthly_neg_rows.append(monthly_cnt)

                if monthly_neg_rows:
                    monthly_neg_df = pd.concat(monthly_neg_rows, ignore_index=True)
                    monthly_neg_df = monthly_neg_df.sort_values(["year", "month"])
                    fig_mneg = px.bar(
                        monthly_neg_df,
                        x="year_month",
                        y="neg_count",
                        color="direction",
                        barmode="group",
                        color_discrete_map={"UP": "#d62728", "DOWN": "#1f77b4"},
                        labels={"neg_count": "Negative periods", "year_month": "Month"},
                        title="Monthly count of negative aFRR energy price periods",
                    )
                    fig_mneg.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_mneg, use_container_width=True)

                    # Downloadable monthly breakdown
                    st.download_button(
                        label="Download monthly negative-price breakdown (CSV)",
                        data=monthly_neg_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"afrr_negative_prices_monthly_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="dl_neg_monthly",
                    )
                else:
                    st.info("No negative price periods to break down by month.")

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

                # Volume-weighted capacity price time series
                st.subheader("Time series: volume-weighted capacity prices (4h step)")
                st.caption("Volume-weighted average per 4h period: Σ(price × awarded_volume) / Σ(awarded_volume)")
                
                # Calculate volume-weighted prices per period (grouping by period_start index)
                def volume_weighted_price(group, price_col, volume_col):
                    """Calculate volume-weighted price for a group."""
                    valid_mask = (group[volume_col].notna()) & (group[volume_col] > 0) & (group[price_col].notna())
                    if valid_mask.sum() == 0:
                        return pd.NA
                    total_volume = group.loc[valid_mask, volume_col].sum()
                    if total_volume == 0:
                        return pd.NA
                    weighted_sum = (group.loc[valid_mask, price_col] * group.loc[valid_mask, volume_col]).sum()
                    return weighted_sum / total_volume
                
                vw_agg = cap_df.groupby(level=0).apply(
                    lambda g: pd.Series({
                        'vw_price_up': volume_weighted_price(g, 'priceupmwh', 'afrrawardedvolumeupmw'),
                        'vw_price_down': volume_weighted_price(g, 'pricedownmwh', 'afrrawardedvolumedownmw')
                    })
                ).sort_index()
                
                st.dataframe(vw_agg.head(200))
                if len(vw_agg) > 200:
                    st.caption(f"Showing first 200 rows of {len(vw_agg)} total rows. Use download button below to get complete dataset.")
                
                # Plot volume-weighted prices as step function
                fig_vw = go.Figure()
                fig_vw.add_trace(go.Scatter(
                    x=vw_agg.index, 
                    y=vw_agg['vw_price_up'], 
                    mode='lines',
                    name='Volume-weighted UP',
                    line_shape='hv',
                    line=dict(color='#d62728')
                ))
                fig_vw.add_trace(go.Scatter(
                    x=vw_agg.index, 
                    y=vw_agg['vw_price_down'], 
                    mode='lines',
                    name='Volume-weighted DOWN',
                    line_shape='hv',
                    line=dict(color='#1f77b4')
                ))
                fig_vw.update_layout(
                    title="Volume-weighted capacity prices (4h step)",
                    yaxis_title="EUR/MW/h (volume-weighted)"
                )
                st.plotly_chart(fig_vw, use_container_width=True)
                
                # Download volume-weighted prices
                st.download_button(
                    label="Download volume-weighted capacity prices (CSV)",
                    data=vw_agg.to_csv(index=True).encode('utf-8'),
                    file_name=f"afrr_volume_weighted_prices_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="dl_vw_prices"
                )

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
                # Store data for download tables
                p_accept_data = []  # For "P(accept) vs Bid Price" table
                bid_x_p_data = []  # For "Bid × P(accept) vs Bid price" table
                
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
                    
                    # Store data for P(accept) vs Bid Price table
                    block_label = f"{blk:02d}-{(blk+4)%24:02d}"
                    for x, y in zip(x_vals, y_vals):
                        p_accept_data.append({
                            "block": block_label,
                            "bid_price_eur_mwh": float(x),
                            "P_accept": float(y)
                        })

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
                        
                        # Store data for Bid × P(accept) vs Bid price table
                        block_label = f"{blk:02d}-{(blk+4)%24:02d}"
                        for x, xy in zip(x_vals, xy_vals):
                            bid_x_p_data.append({
                                "block": block_label,
                                "bid_price_eur_mwh": float(x),
                                "bid_x_P_accept_eur_mwh": float(xy)
                            })
                        
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
                    
                    # Download tables for plot data
                    st.subheader("Download Plot Data")
                    
                    if p_accept_data:
                        p_accept_df = pd.DataFrame(p_accept_data).sort_values(["block", "bid_price_eur_mwh"])
                        st.write("**P(accept) vs Bid Price Data**")
                        st.dataframe(p_accept_df.head(100), use_container_width=True)
                        if len(p_accept_df) > 100:
                            st.caption(f"Showing first 100 rows of {len(p_accept_df)} total rows")
                        st.download_button(
                            label="Download P(accept) vs Bid Price data (CSV)",
                            data=p_accept_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"p_accept_vs_bid_price_{direction}_{block_choice}_{start_date}_{end_date}.csv",
                            mime="text/csv",
                            key="dl_p_accept"
                        )
                    
                    if bid_x_p_data:
                        bid_x_p_df = pd.DataFrame(bid_x_p_data).sort_values(["block", "bid_price_eur_mwh"])
                        st.write("**Bid × P(accept) vs Bid price Data**")
                        st.dataframe(bid_x_p_df.head(100), use_container_width=True)
                        if len(bid_x_p_df) > 100:
                            st.caption(f"Showing first 100 rows of {len(bid_x_p_df)} total rows")
                        st.download_button(
                            label="Download Bid × P(accept) vs Bid price data (CSV)",
                            data=bid_x_p_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"bid_x_p_accept_vs_bid_price_{direction}_{block_choice}_{start_date}_{end_date}.csv",
                            mime="text/csv",
                            key="dl_bid_x_p"
                        )

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

    with tab4:
        if df_pv is None or df_pv.empty:
            st.warning("No photovoltaic production data found. Click 'Fetch PV data' to load data.")
        else:
            st.caption("Data source: Photovoltaic power production estimation and forecast on Belgian grid — ods032.")
            st.info(f"📅 Data covers: {df_pv.index.min()} to {df_pv.index.max()} (total: {len(df_pv)} rows)")
            
            # Region selector
            st.subheader("Region Selection")
            available_regions = sorted(df_pv['region'].unique()) if 'region' in df_pv.columns else []
            if not available_regions:
                st.warning("No region data available.")
            else:
                selected_region = st.selectbox("Select Region", available_regions)
                
                # Filter data by selected region
                df_region = df_pv[df_pv['region'] == selected_region].copy()
                
                if df_region.empty:
                    st.warning(f"No data for region {selected_region}.")
                else:
                    # Ensure we have the required columns
                    required_cols = ['measured', 'monitoredcapacity', 'loadfactor']
                    missing_cols = [col for col in required_cols if col not in df_region.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                    else:
                        st.success(f"Displaying {len(df_region)} rows for {selected_region}")
                        
                        # Ensure index is DatetimeIndex (handle tz-aware values)
                        if not isinstance(df_region.index, pd.DatetimeIndex):
                            df_region.index = pd.to_datetime(df_region.index, utc=True)
                        
                        # Add time features for aggregations
                        df_region['year'] = df_region.index.year
                        df_region['month'] = df_region.index.month
                        df_region['month_name'] = df_region.index.month_name()
                        df_region['date'] = df_region.index.date
                        
                        # Plot 1: Measured values, monitored capacity, and load factor
                        st.subheader("Production Metrics Over Time")
                        
                        # Create separate subplots for better visualization
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=df_region.index, 
                            y=df_region['measured'], 
                            mode='lines', 
                            name='Measured Production (MW)',
                            line=dict(color='#1f77b4')
                        ))
                        fig1.add_trace(go.Scatter(
                            x=df_region.index, 
                            y=df_region['monitoredcapacity'], 
                            mode='lines', 
                            name='Monitored Capacity (MW)',
                            line=dict(color='#ff7f0e', dash='dash')
                        ))
                        fig1.update_layout(
                            title=f"PV Production and Capacity - {selected_region}",
                            xaxis_title="Time",
                            yaxis_title="Power (MW)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Load factor plot
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=df_region.index, 
                            y=df_region['loadfactor'], 
                            mode='lines', 
                            name='Load Factor (%)',
                            line=dict(color='#2ca02c'),
                            fill='tozeroy'
                        ))
                        fig2.update_layout(
                            title=f"Load Factor - {selected_region}",
                            xaxis_title="Time",
                            yaxis_title="Load Factor (%)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Plot 2: Energy (trapezoidal approximation)
                        st.subheader("Cumulative Energy Production")
                        st.caption("Energy calculated using trapezoidal integration (assumes 15-minute intervals)")
                        
                        # Calculate energy using trapezoidal rule
                        # Energy = ∫ Power dt, with Power in MW and time in hours
                        # For 15-minute intervals, dt = 0.25 hours
                        df_region_sorted = df_region.sort_index()
                        
                        # Calculate time differences in hours
                        time_diffs = df_region_sorted.index.to_series().diff().dt.total_seconds() / 3600
                        # For the first point, assume same interval as the second
                        time_diffs.iloc[0] = time_diffs.iloc[1] if len(time_diffs) > 1 else 0.25
                        
                        # Trapezoidal rule: E_i = (P_i + P_{i-1})/2 * dt_i
                        power_avg = (df_region_sorted['measured'] + df_region_sorted['measured'].shift(1)) / 2
                        power_avg.iloc[0] = df_region_sorted['measured'].iloc[0]  # First point
                        
                        energy_increments = power_avg * time_diffs
                        cumulative_energy = energy_increments.cumsum()
                        
                        df_region_sorted['cumulative_energy_mwh'] = cumulative_energy
                        df_region_sorted['energy_increment_mwh'] = energy_increments
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            x=df_region_sorted.index, 
                            y=df_region_sorted['cumulative_energy_mwh'], 
                            mode='lines', 
                            name='Cumulative Energy (MWh)',
                            line=dict(color='#d62728'),
                            fill='tozeroy'
                        ))
                        fig3.update_layout(
                            title=f"Cumulative Energy Production - {selected_region}",
                            xaxis_title="Time",
                            yaxis_title="Cumulative Energy (MWh)",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Plot 3: kWh per month per kWp
                        st.subheader("Monthly Production per kWp")
                        st.caption("Energy produced per month normalized by monitored capacity (kWh/kWp)")
                        
                        # Group by year and month, sum energy increments
                        monthly_energy = df_region_sorted.groupby(['year', 'month', 'month_name']).agg({
                            'energy_increment_mwh': 'sum',
                            'monitoredcapacity': 'mean'  # average capacity for the month
                        }).reset_index()
                        
                        # Convert MWh to kWh and MW to kW
                        monthly_energy['energy_kwh'] = monthly_energy['energy_increment_mwh'] * 1000
                        monthly_energy['capacity_kw'] = monthly_energy['monitoredcapacity'] * 1000
                        
                        # Calculate kWh per kWp (kWp = kW peak)
                        monthly_energy['kwh_per_kwp'] = monthly_energy['energy_kwh'] / monthly_energy['capacity_kw']
                        
                        # Create year-month label for plotting
                        monthly_energy['year_month'] = monthly_energy['year'].astype(str) + '-' + monthly_energy['month_name']
                        
                        # Sort by year and month
                        monthly_energy = monthly_energy.sort_values(['year', 'month'])
                        
                        fig4 = px.bar(
                            monthly_energy, 
                            x='year_month', 
                            y='kwh_per_kwp',
                            title=f"Monthly Energy Production per kWp - {selected_region}",
                            labels={'year_month': 'Month', 'kwh_per_kwp': 'kWh/kWp'},
                            color='kwh_per_kwp',
                            color_continuous_scale='Viridis'
                        )
                        fig4.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        # Display monthly statistics table
                        st.subheader("Monthly Statistics")
                        display_cols = ['year_month', 'energy_kwh', 'capacity_kw', 'kwh_per_kwp']
                        st.dataframe(monthly_energy[display_cols].style.format({
                            'energy_kwh': '{:.2f}',
                            'capacity_kw': '{:.2f}',
                            'kwh_per_kwp': '{:.3f}'
                        }))
                        
                        # Downloads
                        st.subheader("Downloads")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download raw PV data (CSV)",
                                data=df_region.to_csv(index=True).encode("utf-8"),
                                file_name=f"pv_production_{selected_region}_{start_date}_{end_date}.csv",
                                mime="text/csv",
                            )
                        with col2:
                            st.download_button(
                                label="Download monthly statistics (CSV)",
                                data=monthly_energy.to_csv(index=False).encode("utf-8"),
                                file_name=f"pv_monthly_stats_{selected_region}_{start_date}_{end_date}.csv",
                                mime="text/csv",
                            )

    with tab5:
        if df_imbalance is None or df_imbalance.empty:
            st.warning("No imbalance data found. Click 'Fetch Imbalance data' to load data.")
        else:
            st.caption("Data source: Balancing energy prices from ELIA Open Data — ods134.")
            st.info(f"📅 Data covers: {df_imbalance.index.min()} to {df_imbalance.index.max()} (total: {len(df_imbalance)} rows)")
            
            # Ensure required columns exist
            required_cols = ['systemimbalance', 'imbalanceprice']
            missing_cols = [col for col in required_cols if col not in df_imbalance.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Filter out NaN values for plotting
                df_plot = df_imbalance[required_cols].dropna()
                
                if df_plot.empty:
                    st.warning("No valid data points for plotting (all values are NaN).")
                else:
                    st.success(f"Displaying {len(df_plot)} data points")
                    
                    # Create scatter plot with quadrant coloring
                    fig = go.Figure()
                    
                    # Add background rectangles for quadrants
                    # Upper half (positive imbalance price) - light green
                    fig.add_shape(
                        type="rect",
                        x0=-1000, y0=0, x1=1000, y1=2500,
                        fillcolor="rgba(144, 238, 144, 0.2)",  # light green
                        layer="below",
                        line_width=0,
                    )
                    # Lower half (negative imbalance price) - light red
                    fig.add_shape(
                        type="rect",
                        x0=-1000, y0=-1000, x1=1000, y1=0,
                        fillcolor="rgba(255, 182, 193, 0.2)",  # light red
                        layer="below",
                        line_width=0,
                    )
                    
                    # Add reference lines
                    # Vertical line at System Imbalance = 0
                    fig.add_vline(
                        x=0,
                        line_dash="solid",
                        line_color="darkgrey",
                        line_width=2,
                        annotation_text="System Imbalance = 0",
                        annotation_position="top"
                    )
                    # Vertical line at System Imbalance = -100
                    fig.add_vline(
                        x=-100,
                        line_dash="dash",
                        line_color="darkgrey",
                        line_width=1,
                    )
                    # Vertical line at System Imbalance = 100
                    fig.add_vline(
                        x=100,
                        line_dash="dash",
                        line_color="darkgrey",
                        line_width=1,
                    )
                    # Horizontal line at Imbalance Price = 0 (x-axis)
                    fig.add_hline(
                        y=0,
                        line_dash="solid",
                        line_color="darkgrey",
                        line_width=2,
                    )
                    
                    # Add scatter plot
                    fig.add_trace(go.Scatter(
                        x=df_plot['systemimbalance'],
                        y=df_plot['imbalanceprice'],
                        mode='markers',
                        marker=dict(
                            color='darkgrey',
                            size=3,
                            line=dict(color='lightgrey', width=0.5),
                            opacity=0.6
                        ),
                        name='Data points',
                        hovertemplate='System Imbalance: %{x:.2f} MW<br>Imbalance Price: %{y:.2f} €/MWh<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Imbalance Price (€/MWh) vs. System Imbalance (MW)",
                        xaxis_title="System Imbalance (MW)",
                        yaxis_title="Imbalance Price (€/MWh)",
                        xaxis=dict(range=[-1000, 1000]),
                        yaxis=dict(range=[-1000, 2500]),
                        hovermode='closest',
                        width=None,
                        height=600,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Descriptive statistics
                    st.subheader("Descriptive Statistics")
                    st.dataframe(df_plot.describe().T)
                    
                    # Additional statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Data Points", len(df_plot))
                    with col2:
                        st.metric("Mean System Imbalance", f"{df_plot['systemimbalance'].mean():.2f} MW")
                    with col3:
                        st.metric("Mean Imbalance Price", f"{df_plot['imbalanceprice'].mean():.2f} €/MWh")
                    
                    # Quadrant analysis
                    st.subheader("Quadrant Analysis")
                    df_plot_copy = df_plot.copy()
                    df_plot_copy['quadrant'] = 'Other'
                    df_plot_copy.loc[(df_plot_copy['systemimbalance'] < 0) & (df_plot_copy['imbalanceprice'] > 0), 'quadrant'] = 'Upper-Left (Shortage)'
                    df_plot_copy.loc[(df_plot_copy['systemimbalance'] > 0) & (df_plot_copy['imbalanceprice'] < 0), 'quadrant'] = 'Lower-Right (Surplus)'
                    df_plot_copy.loc[(df_plot_copy['systemimbalance'] > 0) & (df_plot_copy['imbalanceprice'] > 0), 'quadrant'] = 'Upper-Right'
                    df_plot_copy.loc[(df_plot_copy['systemimbalance'] < 0) & (df_plot_copy['imbalanceprice'] < 0), 'quadrant'] = 'Lower-Left'
                    
                    quadrant_counts = df_plot_copy['quadrant'].value_counts()
                    total_count = len(df_plot_copy)
                    quadrant_df = pd.DataFrame({
                        'Count': quadrant_counts,
                        'Percentage': (quadrant_counts / total_count * 100).round(2)
                    })
                    st.dataframe(quadrant_df)
                    
                    # Alpha statistics for positive imbalance
                    st.subheader("Alpha Statistics for Positive Imbalance")
                    if 'alpha' in df_imbalance.columns and 'systemimbalance' in df_imbalance.columns:
                        # Filter for positive imbalance
                        positive_imbalance = df_imbalance[df_imbalance['systemimbalance'] > 0].copy()
                        if 'alpha' in positive_imbalance.columns:
                            alpha_positive = positive_imbalance['alpha'].dropna()
                            
                            if len(alpha_positive) > 0:
                                st.info(f"Found {len(alpha_positive)} data points with positive imbalance (out of {len(df_imbalance)} total)")
                                
                                # Statistics table
                                alpha_stats = alpha_positive.describe()
                                st.dataframe(alpha_stats.to_frame(name='Alpha (Positive Imbalance)').T)
                                
                                # Additional metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Count", len(alpha_positive))
                                with col2:
                                    st.metric("Mean", f"{alpha_positive.mean():.4f}")
                                with col3:
                                    st.metric("Median", f"{alpha_positive.median():.4f}")
                                with col4:
                                    st.metric("Std Dev", f"{alpha_positive.std():.4f}")
                                
                                # Distribution plot
                                fig_alpha = go.Figure()
                                fig_alpha.add_trace(go.Histogram(
                                    x=alpha_positive,
                                    nbinsx=50,
                                    name='Alpha (Positive Imbalance)',
                                    marker_color='#1f77b4'
                                ))
                                fig_alpha.update_layout(
                                    title="Distribution of Alpha when System Imbalance > 0",
                                    xaxis_title="Alpha",
                                    yaxis_title="Frequency",
                                    showlegend=False
                                )
                                st.plotly_chart(fig_alpha, use_container_width=True)
                            else:
                                st.warning("No alpha values available for positive imbalance periods.")
                        else:
                            st.warning("Alpha column not found in the data.")
                    else:
                        st.warning("Required columns (alpha, systemimbalance) not found in the data.")
                    
                    # General Alpha and Alpha Prime statistics
                    st.subheader("Alpha and Alpha Prime Statistics (from ods134)")
                    if 'alpha' in df_imbalance.columns or 'alpha_prime' in df_imbalance.columns:
                        alpha_stats_list = []
                        
                        if 'alpha' in df_imbalance.columns:
                            alpha_all = df_imbalance['alpha'].dropna()
                            if len(alpha_all) > 0:
                                alpha_stats_list.append({
                                    'Variable': 'Alpha',
                                    'Count': len(alpha_all),
                                    'Mean': alpha_all.mean(),
                                    'Median': alpha_all.median(),
                                    'Std Dev': alpha_all.std(),
                                    'Min': alpha_all.min(),
                                    'Max': alpha_all.max(),
                                    'Q25': alpha_all.quantile(0.25),
                                    'Q75': alpha_all.quantile(0.75)
                                })
                        
                        if 'alpha_prime' in df_imbalance.columns:
                            alpha_prime_all = df_imbalance['alpha_prime'].dropna()
                            if len(alpha_prime_all) > 0:
                                alpha_stats_list.append({
                                    'Variable': 'Alpha Prime',
                                    'Count': len(alpha_prime_all),
                                    'Mean': alpha_prime_all.mean(),
                                    'Median': alpha_prime_all.median(),
                                    'Std Dev': alpha_prime_all.std(),
                                    'Min': alpha_prime_all.min(),
                                    'Max': alpha_prime_all.max(),
                                    'Q25': alpha_prime_all.quantile(0.25),
                                    'Q75': alpha_prime_all.quantile(0.75)
                                })
                        
                        if alpha_stats_list:
                            stats_df = pd.DataFrame(alpha_stats_list)
                            st.dataframe(stats_df.style.format({
                                'Count': '{:.0f}',
                                'Mean': '{:.6f}',
                                'Median': '{:.6f}',
                                'Std Dev': '{:.6f}',
                                'Min': '{:.6f}',
                                'Max': '{:.6f}',
                                'Q25': '{:.6f}',
                                'Q75': '{:.6f}'
                            }))
                            
                            # Detailed descriptive statistics
                            st.write("**Detailed Descriptive Statistics**")
                            alpha_cols = []
                            if 'alpha' in df_imbalance.columns:
                                alpha_cols.append('alpha')
                            if 'alpha_prime' in df_imbalance.columns:
                                alpha_cols.append('alpha_prime')
                            
                            if alpha_cols:
                                detailed_stats = df_imbalance[alpha_cols].describe().T
                                st.dataframe(detailed_stats)
                            
                            # Distribution plots
                            if 'alpha' in df_imbalance.columns and 'alpha_prime' in df_imbalance.columns:
                                st.write("**Distribution Comparison**")
                                fig_alpha_comp = go.Figure()
                                
                                alpha_all = df_imbalance['alpha'].dropna()
                                alpha_prime_all = df_imbalance['alpha_prime'].dropna()
                                
                                if len(alpha_all) > 0:
                                    fig_alpha_comp.add_trace(go.Histogram(
                                        x=alpha_all,
                                        nbinsx=50,
                                        name='Alpha',
                                        marker_color='#1f77b4',
                                        opacity=0.7
                                    ))
                                
                                if len(alpha_prime_all) > 0:
                                    fig_alpha_comp.add_trace(go.Histogram(
                                        x=alpha_prime_all,
                                        nbinsx=50,
                                        name='Alpha Prime',
                                        marker_color='#ff7f0e',
                                        opacity=0.7
                                    ))
                                
                                fig_alpha_comp.update_layout(
                                    title="Distribution of Alpha and Alpha Prime",
                                    xaxis_title="Value",
                                    yaxis_title="Frequency",
                                    barmode='overlay'
                                )
                                st.plotly_chart(fig_alpha_comp, use_container_width=True)
                            
                            # Time series plot if both are available
                            if 'alpha' in df_imbalance.columns and 'alpha_prime' in df_imbalance.columns:
                                st.write("**Time Series**")
                                alpha_ts = df_imbalance[['alpha', 'alpha_prime']].dropna()
                                if not alpha_ts.empty:
                                    fig_alpha_ts = go.Figure()
                                    fig_alpha_ts.add_trace(go.Scatter(
                                        x=alpha_ts.index,
                                        y=alpha_ts['alpha'],
                                        mode='lines',
                                        name='Alpha',
                                        line=dict(color='#1f77b4')
                                    ))
                                    fig_alpha_ts.add_trace(go.Scatter(
                                        x=alpha_ts.index,
                                        y=alpha_ts['alpha_prime'],
                                        mode='lines',
                                        name='Alpha Prime',
                                        line=dict(color='#ff7f0e')
                                    ))
                                    fig_alpha_ts.update_layout(
                                        title="Alpha and Alpha Prime Over Time",
                                        xaxis_title="Time",
                                        yaxis_title="Value",
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_alpha_ts, use_container_width=True)
                        else:
                            st.warning("No alpha or alpha_prime data available.")
                    else:
                        st.warning("Alpha and alpha_prime columns not found in the data.")
                    
                    # Marginal Decremental Price Statistics
                    st.subheader("Marginal Decremental Price Statistics")
                    if 'marginaldecrementalprice' in df_imbalance.columns:
                        mdp_all = df_imbalance['marginaldecrementalprice'].dropna()
                        
                        if len(mdp_all) > 0:
                            st.info(f"Found {len(mdp_all)} data points with marginal decremental price (out of {len(df_imbalance)} total)")
                            
                            # Statistics table
                            mdp_stats = mdp_all.describe()
                            st.dataframe(mdp_stats.to_frame(name='Marginal Decremental Price').T)
                            
                            # Additional metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Count", len(mdp_all))
                            with col2:
                                st.metric("Mean", f"{mdp_all.mean():.2f} €/MWh")
                            with col3:
                                st.metric("Median", f"{mdp_all.median():.2f} €/MWh")
                            with col4:
                                st.metric("Std Dev", f"{mdp_all.std():.2f} €/MWh")
                            
                            # Count positive vs negative
                            positive_mdp = (mdp_all > 0).sum()
                            negative_mdp = (mdp_all < 0).sum()
                            zero_mdp = (mdp_all == 0).sum()
                            
                            st.write("**Sign Analysis**")
                            sign_col1, sign_col2, sign_col3 = st.columns(3)
                            with sign_col1:
                                st.metric("Positive Values", positive_mdp, f"({positive_mdp/len(mdp_all)*100:.1f}%)")
                            with sign_col2:
                                st.metric("Negative Values", negative_mdp, f"({negative_mdp/len(mdp_all)*100:.1f}%)")
                            with sign_col3:
                                st.metric("Zero Values", zero_mdp, f"({zero_mdp/len(mdp_all)*100:.1f}%)")
                            
                            # Distribution plot
                            fig_mdp = go.Figure()
                            fig_mdp.add_trace(go.Histogram(
                                x=mdp_all,
                                nbinsx=50,
                                name='Marginal Decremental Price',
                                marker_color='#2ca02c'
                            ))
                            fig_mdp.update_layout(
                                title="Distribution of Marginal Decremental Price",
                                xaxis_title="Marginal Decremental Price (€/MWh)",
                                yaxis_title="Frequency",
                                showlegend=False
                            )
                            st.plotly_chart(fig_mdp, use_container_width=True)
                            
                            # Analysis of marginaldecrementalprice - alpha
                            st.subheader("Analysis: Marginal Decremental Price - Alpha (Positive System Imbalance)")
                            if 'alpha' in df_imbalance.columns:
                                # Create a dataframe with required columns, dropping rows where any is NaN
                                # Include systemimbalance and imbalanceprice to ensure we only analyze valid imbalance periods
                                required_cols = ['marginaldecrementalprice', 'alpha', 'systemimbalance', 'imbalanceprice']
                                available_cols = [col for col in required_cols if col in df_imbalance.columns]
                                analysis_df = df_imbalance[available_cols].dropna()

                                # Restrict to positive system imbalance only
                                if 'systemimbalance' in analysis_df.columns:
                                    analysis_df = analysis_df[analysis_df['systemimbalance'] > 0]
                                
                                if len(analysis_df) > 0:
                                    # Calculate the difference
                                    analysis_df['mdp_minus_alpha'] = analysis_df['marginaldecrementalprice'] - analysis_df['alpha']
                                    
                                    # Count cases where mdp - alpha < 0
                                    negative_diff = (analysis_df['mdp_minus_alpha'] < 0).sum()
                                    positive_diff = (analysis_df['mdp_minus_alpha'] > 0).sum()
                                    zero_diff = (analysis_df['mdp_minus_alpha'] == 0).sum()
                                    
                                    st.info(f"Analyzing {len(analysis_df)} data points with positive system imbalance where marginaldecrementalprice, alpha, systemimbalance, and imbalanceprice are all available")
                                    
                                    # Summary statistics
                                    st.write("**Difference Statistics (Marginal Decremental Price - Alpha)**")
                                    diff_stats = analysis_df['mdp_minus_alpha'].describe()
                                    st.dataframe(diff_stats.to_frame(name='MDP - Alpha').T)
                                    
                                    # Count cases where difference is negative
                                    st.write("**Cases where Marginal Decremental Price - Alpha < 0**")
                                    diff_col1, diff_col2, diff_col3 = st.columns(3)
                                    with diff_col1:
                                        st.metric("Negative Difference", negative_diff, f"({negative_diff/len(analysis_df)*100:.1f}%)")
                                    with diff_col2:
                                        st.metric("Positive Difference", positive_diff, f"({positive_diff/len(analysis_df)*100:.1f}%)")
                                    with diff_col3:
                                        st.metric("Zero Difference", zero_diff, f"({zero_diff/len(analysis_df)*100:.1f}%)")
                                    
                                    # Filter cases where mdp - alpha < 0
                                    negative_cases = analysis_df[analysis_df['mdp_minus_alpha'] < 0].copy()
                                    
                                    if len(negative_cases) > 0:
                                        st.write(f"**Analysis of {len(negative_cases)} cases where MDP - Alpha < 0**")
                                        
                                        # Breakdown by MDP sign and resulting imbalance price
                                        total_neg_cases = len(negative_cases)
                                        # Cases where MDP is positive or zero and final imbalance price is negative
                                        pos_or_zero_mdp_neg_ip = (
                                            (negative_cases['marginaldecrementalprice'] >= 0) &
                                            (negative_cases['imbalanceprice'] < 0)
                                        ).sum()
                                        # Cases with negative MDP (any final price)
                                        neg_mdp_cases = (negative_cases['marginaldecrementalprice'] < 0).sum()
                                        # Cases where final imbalance price is exactly zero
                                        zero_imbalance_price_cases = (negative_cases['imbalanceprice'] == 0).sum()
                                        
                                        st.write(f"**In these {total_neg_cases} cases:**")
                                        case_col1, case_col2, case_col3 = st.columns(3)
                                        with case_col1:
                                            st.metric(
                                                "Positive or Zero MDP → Negative Imbalance Price",
                                                pos_or_zero_mdp_neg_ip,
                                                f"({pos_or_zero_mdp_neg_ip/total_neg_cases*100:.1f}%)"
                                            )
                                        with case_col2:
                                            st.metric(
                                                "Cases with Negative MDP (any price)",
                                                neg_mdp_cases,
                                                f"({neg_mdp_cases/total_neg_cases*100:.1f}%)"
                                            )
                                        with case_col3:
                                            st.metric(
                                                "Imbalance Price = 0",
                                                zero_imbalance_price_cases,
                                                f"({zero_imbalance_price_cases/total_neg_cases*100:.1f}%)"
                                            )
                                        
                                        # Statistics for these cases
                                        st.write("**Statistics for cases where MDP - Alpha < 0:**")
                                        negative_cases_stats = negative_cases[['marginaldecrementalprice', 'alpha', 'mdp_minus_alpha']].describe().T
                                        st.dataframe(negative_cases_stats)
                                        
                                        # Scatter plot: MDP vs Alpha for negative difference cases
                                        fig_neg_diff = go.Figure()
                                        fig_neg_diff.add_trace(go.Scatter(
                                            x=negative_cases['alpha'],
                                            y=negative_cases['marginaldecrementalprice'],
                                            mode='markers',
                                            marker=dict(
                                                color=negative_cases['mdp_minus_alpha'],
                                                colorscale='RdBu',
                                                size=5,
                                                showscale=True,
                                                colorbar=dict(title="MDP - Alpha")
                                            ),
                                            name='MDP - Alpha < 0',
                                            hovertemplate='Alpha: %{x:.2f}<br>MDP: %{y:.2f} €/MWh<br>Difference: %{marker.color:.2f}<extra></extra>'
                                        ))
                                        # Add line where MDP = Alpha (diagonal)
                                        if len(negative_cases) > 0:
                                            min_val = min(negative_cases['alpha'].min(), negative_cases['marginaldecrementalprice'].min())
                                            max_val = max(negative_cases['alpha'].max(), negative_cases['marginaldecrementalprice'].max())
                                            fig_neg_diff.add_trace(go.Scatter(
                                                x=[min_val, max_val],
                                                y=[min_val, max_val],
                                                mode='lines',
                                                name='MDP = Alpha (diagonal)',
                                                line=dict(color='red', dash='dash', width=2)
                                            ))
                                        fig_neg_diff.update_layout(
                                            title="Marginal Decremental Price vs Alpha (cases where MDP - Alpha < 0)",
                                            xaxis_title="Alpha",
                                            yaxis_title="Marginal Decremental Price (€/MWh)",
                                            hovermode='closest'
                                        )
                                        st.plotly_chart(fig_neg_diff, use_container_width=True)
                                        
                                        # Distribution of the difference
                                        fig_diff_dist = go.Figure()
                                        fig_diff_dist.add_trace(go.Histogram(
                                            x=analysis_df['mdp_minus_alpha'],
                                            nbinsx=50,
                                            name='MDP - Alpha',
                                            marker_color='#d62728'
                                        ))
                                        fig_diff_dist.add_vline(
                                            x=0,
                                            line_dash="dash",
                                            line_color="black",
                                            line_width=2,
                                            annotation_text="Zero line"
                                        )
                                        fig_diff_dist.update_layout(
                                            title="Distribution of (Marginal Decremental Price - Alpha)",
                                            xaxis_title="MDP - Alpha (€/MWh)",
                                            yaxis_title="Frequency",
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig_diff_dist, use_container_width=True)
                                    else:
                                        st.info("No cases found where Marginal Decremental Price - Alpha < 0")
                                else:
                                    st.warning("No data points available where both marginaldecrementalprice and alpha are present.")
                            else:
                                st.warning("Alpha column not found. Cannot perform MDP - Alpha analysis.")
                        else:
                            st.warning("No marginal decremental price data available.")
                    else:
                        st.warning("Marginal decremental price column not found in the data.")
                    
                    # Download
                    st.subheader("Downloads")
                    st.download_button(
                        label="Download imbalance data (CSV)",
                        data=df_imbalance.to_csv(index=True).encode("utf-8"),
                        file_name=f"imbalance_data_{start_date}_{end_date}.csv",
                        mime="text/csv",
                    )

if __name__ == "__main__":
    main()


