import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import pytz

logger = logging.getLogger(__name__)

BASE_URL = 'https://opendata.elia.be/api/explore/v2.1/catalog/datasets/'

def fetch_day_data(dataset, date, select_fields, time_field, refine_filters=None, limit=100):
    """Fetch all records for a day, applying optional refine filters and pagination."""
    url = f"{BASE_URL}{dataset}/records"

    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_time = date_obj.isoformat()
    end_time = (date_obj + timedelta(days=1)).isoformat()

    all_results = []
    start = 0

    while True:
        # Build the where clause
        where_clause = f'{time_field} >= "{start_time}" AND {time_field} < "{end_time}"'
        
        if refine_filters:
            for key, value in refine_filters.items():
                if key == time_field:  # If it's a time field filter, just append to where clause
                    where_clause += f" AND {value}"
                else:
                    where_clause += f' AND {key} = "{value}"'

        params = {
            'select': select_fields,
            'where': where_clause,
            'limit': limit,
            'start': start,
            'timezone': 'Europe/Brussels'
        }

        logger.info(f"Fetching from ELIA API: {url}")
        logger.info(f"With params: {params}")
        
        # Add a timeout to avoid hanging requests
        response = requests.get(url, params=params, timeout=30)
        logger.info(f"API Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"API Error: {error_text}")
            raise Exception(f"Failed to retrieve {dataset} data. Status code: {response.status_code}")

        data = response.json()
        
        batch = data.get("results", [])
        all_results.extend(batch)

        if len(batch) < limit:
            break  # No more pages
        start += limit

    return {'results': all_results}

def fetch_range_data(dataset, start_date, end_date, select_fields, time_field, refine_filters=None, limit=100):
    """Fetch all records for a date range [start_date, end_date] inclusive.

    Parameters mirror fetch_day_data but cover multiple days using one paginated query.
    - start_date/end_date: strings 'YYYY-MM-DD'
    - limit: page size (API max without group_by is typically 100)
    """
    url_exports = f"{BASE_URL}{dataset}/exports/json"
    url_records = f"{BASE_URL}{dataset}/records"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    start_time = start_dt.isoformat()
    end_time = end_dt.isoformat()

    where_clause = f'{time_field} >= "{start_time}" AND {time_field} < "{end_time}"'

    if refine_filters:
        for key, value in refine_filters.items():
            if key == time_field:
                where_clause += f" AND {value}"
            else:
                where_clause += f' AND {key} = "{value}"'

    params_exports = {
        'select': select_fields,
        'where': where_clause,
        'timezone': 'Europe/Brussels'
    }

    logger.info(f"Fetching (range) from ELIA API (exports): {url_exports}")
    logger.info(f"With params: {params_exports}")

    response = requests.get(url_exports, params=params_exports, timeout=60)
    logger.info(f"Exports API Response status: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return {'results': data}
            if isinstance(data, dict) and data.get('results'):
                return {'results': data.get('results', [])}
            logger.warning("Exports endpoint returned 200 but no rows. Falling back to records pagination.")
        except Exception as e:
            logger.warning(f"Failed to parse exports JSON ({e}). Falling back to records.")
    else:
        logger.error(f"Exports error body (truncated): {response.text[:300]}")

    # Fallback: paginate via /records
    all_results = []
    start = 0
    while True:
        params_records = {
            'select': select_fields,
            'where': where_clause,
            'limit': min(max(int(limit), 1), 100),
            'start': start,
            'timezone': 'Europe/Brussels'
        }
        logger.info(f"Fetching (range fallback) from ELIA API: {url_records}")
        logger.info(f"With params: {params_records}")
        resp = requests.get(url_records, params=params_records, timeout=60)
        logger.info(f"Records API Response status: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"Records error body (truncated): {resp.text[:300]}")
            raise Exception(f"Failed to retrieve {dataset} range data via records. Status code: {resp.status_code}")
        data = resp.json()
        batch = data.get('results', [])
        all_results.extend(batch)
        if len(batch) < params_records['limit']:
            break
        start += params_records['limit']

    return {'results': all_results}

def filter_data(data, index_column=None):
    """Convert API result JSON to a pandas DataFrame, optionally setting the index."""
    filtered_data = data['results'] if 'results' in data else []
    df = pd.DataFrame(filtered_data)
    
    # Handle invalid float values (inf, -inf, NaN)
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        # Replace inf/-inf with None (which will be serialized as null in JSON)
        df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA)
        # Replace NaN with None using a direct assignment
        df[col] = df[col].where(df[col].notna(), None)
    
    if index_column and index_column in df.columns:
        df[index_column] = pd.to_datetime(df[index_column])
        df.set_index(index_column, inplace=True)
    return df

def fetch_balancing_energy_volume(date):
    """Fetch balancing energy volume data from ELIA, with date validation."""
    try:
        # Check if date is in the future or too recent
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        brussels_tz = pytz.timezone('Europe/Brussels')
        now = datetime.now(brussels_tz)
        today = now.date()
        
        if date_obj.date() > today:
            logger.warning(f"Future date requested for balancing energy volume: {date}")
            return pd.DataFrame()
        
        if date_obj.date() > (today - timedelta(days=2)):
            logger.warning(f"Date {date} is too recent for ELIA balancing energy data (less than 2 days old)")
            return pd.DataFrame()
        
        data = fetch_day_data(
            dataset='ods127',
            date=date,
            select_fields='datetime,igccvolumeup,afrrvolumeup,mfrrsaup,mfrrdaup,reserve_sharing_import,igccvolumedown,afrrvolumedown,mfrrsadown,mfrrdadown,reserve_sharing_export,qualitystatus',
            time_field='datetime'
        )
        return filter_data(data, index_column='datetime')
    except Exception as e:
        logger.error(f"Error fetching balancing energy volume for {date}: {str(e)}")
        return pd.DataFrame()

def fetch_balancing_energy_prices(date):
    data = fetch_day_data(
        dataset='ods134',
        date=date,
        select_fields='datetime,ace,systemimbalance,alpha,alpha_prime,marginalincrementalprice,marginaldecrementalprice,imbalanceprice',
        time_field='datetime'
    )
    return filter_data(data, index_column='datetime')

def fetch_afrr_energy_price(date):
    logger.info(f"Starting fetch_afrr_energy_price for {date}")
    try:
        # Check if date is before May 2024
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        may_2024 = datetime(2024, 5, 1)
        
        if date_obj < may_2024:
            logger.info(f"Using historical dataset (ods134) for {date}")
            data = fetch_day_data(
                dataset='ods134',
                date=date,
                select_fields='datetime,marginalincrementalprice,marginaldecrementalprice,imbalanceprice,qualitystatus',
                time_field='datetime'
            )
            # Rename columns to match the expected format
            df = filter_data(data, index_column='datetime')
            if not df.empty:
                df = df.rename(columns={
                    'marginalincrementalprice': 'afrrpriceup',
                    'marginaldecrementalprice': 'afrrpricedown'
                })
        else:
            logger.info(f"Using future dataset (ods166) for {date}")
            data = fetch_day_data(
                dataset='ods166',
                date=date,
                select_fields='datetime,afrrpriceup,marginalincrementalprice,floorprice,afrrpricedown,marginaldecrementalprice,cap,qualitystatus',
                time_field='datetime'
            )
            df = filter_data(data, index_column='datetime')
        
        logger.info(f"Raw energy data response: {data}")
        
        if not data.get('results'):
            logger.warning(f"No results returned for energy data on {date}")
            return pd.DataFrame()
            
        logger.info(f"Energy price data shape: {df.shape}")
        logger.info(f"Energy price data columns: {df.columns.tolist()}")
        
        if df.empty:
            logger.warning(f"Empty DataFrame after filtering for {date}")
            return df
            
        logger.info(f"First row of energy price data: {df.iloc[0].to_dict()}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch energy data for {date}: {str(e)}")
        raise

def fetch_afrr_energy_price_range(start_date, end_date):
    """Fetch aFRR energy prices for a date range [start_date, end_date] inclusive.

    Handles split between historical dataset ods134 (before 2024-05-01) and ods166 (from 2024-05-01).
    Returns a pandas DataFrame indexed by datetime with columns including 'afrrpriceup' and 'afrrpricedown'.
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        may_2024 = datetime(2024, 5, 1)

        frames = []

        # Historical part (before 2024-05-01)
        if start_dt < may_2024:
            hist_start = start_dt
            hist_end = min(end_dt, may_2024 - timedelta(days=1))
            data_hist = fetch_range_data(
                dataset='ods134',
                start_date=hist_start.strftime('%Y-%m-%d'),
                end_date=hist_end.strftime('%Y-%m-%d'),
                select_fields='datetime,marginalincrementalprice,marginaldecrementalprice,imbalanceprice,qualitystatus',
                time_field='datetime'
            )
            df_hist = filter_data(data_hist, index_column='datetime')
            if not df_hist.empty:
                df_hist = df_hist.rename(columns={
                    'marginalincrementalprice': 'afrrpriceup',
                    'marginaldecrementalprice': 'afrrpricedown'
                })
                frames.append(df_hist)

        # New dataset (from 2024-05-01)
        if end_dt >= may_2024:
            fut_start = max(start_dt, may_2024)
            fut_end = end_dt
            data_new = fetch_range_data(
                dataset='ods166',
                start_date=fut_start.strftime('%Y-%m-%d'),
                end_date=fut_end.strftime('%Y-%m-%d'),
                select_fields='datetime,afrrpriceup,marginalincrementalprice,floorprice,afrrpricedown,marginaldecrementalprice,cap,qualitystatus',
                time_field='datetime'
            )
            df_new = filter_data(data_new, index_column='datetime')
            if not df_new.empty:
                frames.append(df_new)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=0)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch energy data for range {start_date} to {end_date}: {str(e)}")
        raise

def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)

def _next_month(dt: datetime) -> datetime:
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    return datetime(year, month, 1)

def fetch_afrr_energy_price_range_chunked(start_date, end_date):
    """Fetch aFRR prices by iterating month-sized chunks to avoid API result size/offset limits.

    Splits [start_date, end_date] into calendar months and merges results.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    frames = []
    cursor = _month_start(start_dt)
    final_end = end_dt

    while cursor <= final_end:
        chunk_start = max(cursor, start_dt)
        chunk_end = min(_next_month(cursor) - timedelta(days=1), final_end)
        try:
            df_chunk = fetch_afrr_energy_price_range(chunk_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'))
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            logger.error(f"Chunk fetch failed for {chunk_start.date()} to {chunk_end.date()}: {e}")
        cursor = _next_month(cursor)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0)
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    return df

def fetch_afrr_capacity(date):
    data = fetch_day_data(
        dataset='ods125',
        date=date,
        select_fields='deliverydate,capacitybiddeliveryperiod,selectedbyoptimizer,afrrofferedvolumeupmw,priceupmwh,afrrofferedvolumedownmw,pricedownmwh',
        time_field='deliverydate',
        refine_filters={'selectedbyoptimizer': 'True'}
    )
    return filter_data(data, index_column='deliverydate')

def fetch_afrr_capacity_range(start_date, end_date):
    """Fetch aFRR capacity bids accepted by optimizer for a date range.

    Returns raw rows with deliverydate and 4-hour period bucket information in columns.
    """
    data = fetch_range_data(
        dataset='ods125',
        start_date=start_date,
        end_date=end_date,
        select_fields='deliverydate,capacitybiddeliveryperiod,selectedbyoptimizer,afrrofferedvolumeupmw,afrrawardedvolumeupmw,priceupmwh,afrrofferedvolumedownmw,afrrawardedvolumedownmw,pricedownmwh,selectedvolumeupafterstep2,selectedvolumedownafterstep2',
        time_field='deliverydate',
        refine_filters={'selectedbyoptimizer': 'True'}
    )
    return filter_data(data, index_column=None)

def fetch_afrr_capacity_range_chunked(start_date, end_date):
    """Month-chunked fetch for aFRR capacity range to avoid API limits."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    frames = []
    cursor = _month_start(start_dt)
    final_end = end_dt
    while cursor <= final_end:
        chunk_start = max(cursor, start_dt)
        chunk_end = min(_next_month(cursor) - timedelta(days=1), final_end)
        try:
            df_chunk = fetch_afrr_capacity_range(chunk_start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'))
            if df_chunk is not None and not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as e:
            logger.error(f"Capacity chunk fetch failed for {chunk_start.date()} to {chunk_end.date()}: {e}")
        cursor = _next_month(cursor)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=0)
    return df.reset_index(drop=True)

