import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
import concurrent.futures
import requests
import json
import os
from datetime import datetime, timezone, timedelta
import time
from millify import millify
import numpy as np


st.set_page_config(layout='wide')

class NoBlobsFoundError(Exception):
    pass

def get_custom_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends the metrics: Hook_rate, Hold_rate and Attraction_index in a df of facebook ads
    Hook_rate: views greater then 3s / impressions
    Hold_rate: views greater then 15s / impressions
    Attraction_index: views greater then 15s / views greater then 3s
    """
    mock_df = df.copy()
    needed_cols = {'spend', 'cost_per_thruplay', 'n_video_view', 'impressions', 'date'}
    if not needed_cols.issubset(set(mock_df.columns)):
        raise Exception('spend, cost_per_thruplay, n_video_view, impressions or date not found in columns')
        return
    else:
        mock_df.loc[mock_df['cost_per_thruplay'] == 0, 'cost_per_thruplay'] = np.nan
        mock_df['thruplay'] = mock_df['spend'] / mock_df['cost_per_thruplay']
        mock_df.loc[mock_df['n_landing_page_view'] == 0, 'n_landing_page_view'] = np.nan
        mock_df['thruview'] = (mock_df['n_landing_page_view']/mock_df['n_link_click']) * 100
        mock_df['CPTV'] = (mock_df['spend'] / mock_df['n_landing_page_view']) * 1.0
        mock_df['Hook_rate'] = (mock_df['n_video_view'] / mock_df['impressions']) * 100
        mock_df['Hold_rate'] = (mock_df['thruplay'] / mock_df['impressions']) * 100
        mock_df['Attraction_index'] = (mock_df['thruplay'] / mock_df['n_video_view']) * 100
        mock_df.loc[:,['Hook_rate', 'Hold_rate', 'Attraction_index']] = mock_df[['Hook_rate', 'Hold_rate', 'Attraction_index']].fillna(value=0)
        mock_df['date'] = pd.to_datetime(mock_df['date'])
        mock_df['date'] = mock_df['date'].dt.date
        mock_df.sort_values(by='date', inplace=True)
        return mock_df
    
def get_data_from_bucket(
        bucket_name: str,
        prefix: str | None = None,
        destination_path: str = './Temp_files') -> pd.DataFrame:
    """
    Download all files from Google Cloud Storage Bucket (bucket_name) and save them in destination_path (local)

    Parameters
    ----------
    path_to_key : str | Path
        Path to .json file with the service key
    bucket_name : str
    prefix: str|None
    To limit blobs with a defined prefix
    max_workers : int
    number of parallelism workers
    destination_path : str | Path
    Returns
    -------
    bool
    """
    destination_path = Path(destination_path)
    
    if destination_path.is_dir():
        destination_path.mkdir(parents=True, exist_ok=True)

    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_STORAGE"])
    client = storage.Client(credentials=credentials)
    source_bucket_name = bucket_name
    bucket = client.bucket(source_bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    blobs = [blob for blob in blobs if blob.name != prefix]
    
    if not blobs:
        raise NoBlobsFoundError('No blobs found with the specific conditions...')
    print('Inicio do download')

    def download_blob(blob, destination_path):
        # Split the blob's name into directory parts and the file name
        parts = blob.name.split('/')
        file_name = parts[-1]
        dirs = parts[:-1]
        dirs = dirs[1:]
    
        # Create subdirectories based on the blob's name
        destination_dir = destination_path / '/'.join(dirs)
        destination_dir.mkdir(parents=True, exist_ok=True)
    
        # Use the original file name without prefixes
        destination_file_path = destination_dir / file_name  # Use 'file_name' directly
    
        if not destination_file_path.is_file():
            with open(destination_file_path, 'wb') as f:
                blob.download_to_file(f)
            return f"Downloaded: {blob.name}"
        else:
            f"Skipped: {blob.name} (already exists)"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda blob: download_blob(blob, destination_path), blobs))
        print("All files downloaded successfully.")
        return True

def datetime_to_milliseconds(dt, end=False):
    """
    Retorna a diferença em milisegundos da data definida por dt a partir de 01/01/1970. Se end = False retorna a diferença
    a partir do inicio do dia 00:00:00:0001 se end = True 23:59:59:9999
    """
    brasilia_tz = timezone(timedelta(hours=-3))
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc)
    dt = pd.to_datetime(dt).tz_localize(brasilia_tz)
    
    if end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    dt_utc = dt.astimezone(timezone.utc)
    delta = dt_utc - epoch
    milliseconds = int(delta.total_seconds() * 1000)
    return milliseconds

def get_hotmart_token() -> None:
    """
    Busca um token de acesso na HotMart via API da HotMart
    """
    client_id = st.secrets['HOTMART_CRED']["Client ID"].strip()
    client_secret = st.secrets['HOTMART_CRED']["Client Secret"].strip()
    basic_auth = st.secrets['HOTMART_CRED']["Basic"].strip()

    url = 'https://api-sec-vlc.hotmart.com/security/oauth/token'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': basic_auth
    }

    params = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    response = requests.post(url, headers=headers, params=params)

    if response.status_code == 200:
        hotmart_json = response.json()
        os.environ['HOTMART_TOKEN'] = json.dumps(hotmart_json)

    else:
        response.raise_for_status()

@st.cache_data
def get_hotmart_sales(
        start_date: datetime,
        end_date: datetime) -> pd.DataFrame:
    """
    Busca na hotmart API os dados referentes a histórico de compras no periodo
    defino por start_date e end_date e retorna um DataFrame
    """
    def query_api(next_page_token=None):

        start = datetime_to_milliseconds(start_date)
        end = datetime_to_milliseconds(end_date, end=True)

        base_url = 'https://developers.hotmart.com/payments/api/v1/sales/summary'

        try:
            token = json.load(os.environ['HOTMART_TOKEN'])
        except:
            get_hotmart_token()
            token = json.loads(os.environ['HOTMART_TOKEN'])

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token.get("access_token")}'
        }
        if next_page_token:
            params = {
                'start_date': start,
                'end_date': end,
                'transaction_status': ['APPROVED','COMPLETE'],
                'max_results': 500,
                'page_token': next_page_token
            }
        else:
            params = {
                'start_date': start,
                'end_date': end,
                'transaction_status': ['APPROVED','COMPLETE']
            }

        response = requests.get(base_url, headers=headers, params=params)
        return response
    
    response = query_api()
    
    next_page = True
    hotmart_values = {'total_items': 0, 'total_revenue': 0}
    while next_page:
        if response.status_code == 200:
            hotmart_json = response.json()
            try:
                hotmart_values['total_items'] = hotmart_values.get('total_items', 0) + hotmart_json['items'][0].get('total_items',0)
                hotmart_values['total_revenue'] = hotmart_values.get('total_revenue', 0) + hotmart_json['items'][0]['total_value'].get('value')
            except Exception as e:
                st.write(e) #Add a logger
            if 'next_page_token' in hotmart_json['page_info']:
                response = query_api(next_page_token=hotmart_json['page_info']['next_page_token'])
            else:
                next_page = False         
        elif response.status_code == 401:
            hotmart_token = get_hotmart_token(os.environ['HOTMART_CRED'])
            get_hotmart_sales(hotmart_token, start_date, end_date)
        elif response.status_code == 429:
            st.warning(f'Excesso de requisições: {response.status_code} aguardando 3s')
            time.sleep(3)
            get_hotmart_sales(hotmart_token=hotmart_token, start_date=start_date, end_date=end_date)
        else:
            st.write(response.status_code)
            return None
    return hotmart_values

@st.cache_data
def process_data():
    temp_files = Path('./Temp_files')
    ga4_data = pd.read_parquet(temp_files/'Processed_concat.parquet', engine='pyarrow')
    fb_data = pd.read_csv(temp_files/'processed_adsets.csv')
    fb_data = get_custom_metrics(fb_data)
    fb = fb_data.loc[(fb_data['campaign_name'] == '[CONVERSAO] [DIP] Broad')].copy()
    return ga4_data, fb

@st.cache_data
def group_data(df):
    grouped_fb = df[['name', 'spend', 'n_purchase']].groupby(by=['name']).sum()
    grouped_fb['(%)'] = (grouped_fb['spend']/grouped_fb['spend'].sum()) * 100
    grouped_fb['(%)'] = grouped_fb['(%)'].round(1)
    grouped_fb['cpa_purchase'] = round(grouped_fb['spend'] / grouped_fb['n_purchase'],2)
    grouped_fb['Valor gasto (R$)'] = grouped_fb['spend'].apply(lambda x: millify(x, precision=1))
    return grouped_fb

###################### GETTING THE DATA #########################################
# DATA LOAD
try: 
    st.session_state['Download_data']
except KeyError:
    st.session_state['Download_data'] = get_data_from_bucket(bucket_name='dashboard_marketing_processed')

ga4, fb = process_data()


# FILTOS
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)))
fb_data = fb.loc[(fb['date'] >= date_range[0]) &(fb['date'] <= date_range[1])].copy()
lista = ['adset_' + name[0] for name in fb_data['name'].str.split(' ')]
fb_data['name'] = lista

selected_adsets = st.sidebar.multiselect(label="Adsets", options=fb_data['name'].unique(), default=fb_data['name'].unique()[0])

##################### GETTING SOME NUMBERS ######################################
n_adsets = fb_data['name'].unique().shape[0]
Total_vendas_fb = fb_data['n_purchase'].sum().astype(int)
Total_gasto_fb = fb_data['spend'].sum()
grouped_fb = group_data(fb_data)
valor_por_adset = round(fb_data['spend'].sum()/n_adsets, 2)
media_vendas = round(Total_vendas_fb/n_adsets,1)
cpa_medio = round(grouped_fb['cpa_purchase'].replace(np.inf, np.nan).dropna().mean(),2)

######################### INICIO DO DASH #########################################
st.title('Analise de criativos')
col_1, col_2 = st.columns(2)
with col_1:
    st.metric(label='Investimento Facebook', value=millify(fb_data['spend'].sum(), precision=1))
    st.metric(label='Vendas pelo Facebook', value=Total_vendas_fb)
with col_2:
    st.metric(label='Media de vendas por adset', value=media_vendas.round(2))
    st.metric(label='Gasto médio por adset', value=valor_por_adset.round(2))
    st.metric(label='CPA Médio', value=cpa_medio)


option_2 = st.radio(label="Selecione a métrica", options=['Valor gasto', 'Vendas totais', 'CPA'], horizontal=True)

if option_2 == 'Valor gasto':
    grouped_fb.sort_values(by='(%)', inplace=True, ascending=True)
    valor_gasto_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb['(%)'], title='Distribuição do valor gasto por adset', color=grouped_fb['(%)'], labels={'(%)':'Porcentagem gasta', 'name':'Criativo'}, hover_name='Valor gasto (R$)')
    st.plotly_chart(valor_gasto_fig, use_container_width=True)
elif option_2 == 'Vendas totais':
    grouped_fb.sort_values(by='n_purchase', inplace=True, ascending=True)
    valor_gasto_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb['n_purchase'], title=f'Distribuição das vendas por adset', color=grouped_fb['n_purchase'], labels={'n_purchase':'Vendas', 'name':'Criativo'})
    st.plotly_chart(valor_gasto_fig, use_container_width=True)
else:
    grouped_fb.sort_values(by='cpa_purchase', inplace=True, ascending=False)
    valor_gasto_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb['cpa_purchase'], title=f'Distribuição das vendas por adset', color=grouped_fb['cpa_purchase'], labels={'cpa_purchase':'CPA', 'name':'Criativo'})
    st.plotly_chart(valor_gasto_fig, use_container_width=True)

col_1, col2 = st.columns(2)
best_tmp = grouped_fb.tail(5)
worst_tmp = grouped_fb.head(5)

with col_1:  
    if option_2 == 'Valor gasto':
        top5_fig = px.bar(best_tmp, x=best_tmp.index, y=best_tmp['(%)'], title='Cinco melhores', text_auto=True, labels={'(%)':'Porcentagem gasta', 'name':'Criativo'}, hover_name='Valor gasto (R$)')
        st.plotly_chart(top5_fig, use_container_width=True)
    elif option_2 == 'Vendas totais':
        top5_fig = px.bar(best_tmp, x=best_tmp.index, y=best_tmp['n_purchase'], title='Cinco melhores', text_auto=True, labels={'n_purchase':'Vendas', 'name':'Criativo'})
        st.plotly_chart(top5_fig, use_container_width=True)
    else:
        top5_fig = px.bar(best_tmp, x=best_tmp.index, y=best_tmp['cpa_purchase'], title='Cinco melhores', text_auto=True, labels={'cpa_purchase':'CPA', 'name':'Criativo'})
        st.plotly_chart(top5_fig, use_container_width=True)

with col2:
    if option_2 == 'Valor gasto':
        worst5_fig = px.bar(worst_tmp, x=worst_tmp.index, y=worst_tmp['(%)'], title='Cinco piores', text_auto=True, labels={'(%)':'Porcentagem gasta', 'name':'Criativo'})
        worst5_fig.update_xaxes(tickangle=45)
        st.plotly_chart(worst5_fig, use_container_width=True)
    
    elif option_2 == 'Vendas totais':
        worst5_fig = px.bar(worst_tmp, x=worst_tmp.index, y=worst_tmp['n_purchase'], title='Cinco piores', text_auto=True, labels={'n_purchase':'Vendas', 'name':'Criativo'})
        st.plotly_chart(worst5_fig, use_container_width=True)
    
    else:
        worst5_fig = px.bar(worst_tmp, x=worst_tmp.index, y=worst_tmp['cpa_purchase'], title='Cinco piores', text_auto=True, labels={'cpa_purchase':'CPA', 'name':'Criativo'})
        st.plotly_chart(worst5_fig, use_container_width=True)


inner_1, inner_2, inner_3, inner_4, inner_5 = st.columns(5)
with inner_1:
    if option_2 == 'Valor gasto':
        st.metric(label=str(best_tmp.index[0]), value=best_tmp.loc[best_tmp.index[0], 'Valor gasto (R$)'], delta=round(best_tmp.loc[best_tmp.index[0], 'spend'] - valor_por_adset,1))
        st.metric(label=str(worst_tmp.index[0]), value=worst_tmp.loc[worst_tmp.index[0], 'Valor gasto (R$)'], delta=round(worst_tmp.loc[worst_tmp.index[0], 'spend'] - valor_por_adset,1))
    elif option_2 == 'Vendas totais':
        st.metric(label=str(best_tmp.index[0]), value=best_tmp.loc[best_tmp.index[0], 'n_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[0], 'n_purchase'] - media_vendas,1))
        st.metric(label=str(worst_tmp.index[0]), value=worst_tmp.loc[worst_tmp.index[0], 'n_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[0], 'n_purchase'] - media_vendas,1))
    else:
        st.metric(label=str(best_tmp.index[0]), value=best_tmp.loc[best_tmp.index[0], 'cpa_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[0], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[0]), value=worst_tmp.loc[worst_tmp.index[0], 'cpa_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[0], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')

with inner_2:
    if option_2 == 'Valor gasto':
        st.metric(label=str(best_tmp.index[1]), value=best_tmp.loc[best_tmp.index[1], 'Valor gasto (R$)'], delta=round(best_tmp.loc[best_tmp.index[1], 'spend'] - valor_por_adset,1))
        st.metric(label=str(worst_tmp.index[1]), value=worst_tmp.loc[worst_tmp.index[1], 'Valor gasto (R$)'], delta=round(worst_tmp.loc[worst_tmp.index[1], 'spend'] - valor_por_adset,1))
    elif option_2 == 'Vendas totais':
        st.metric(label=str(best_tmp.index[1]), value=best_tmp.loc[best_tmp.index[1], 'n_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[1], 'n_purchase'] - media_vendas,1))
        st.metric(label=str(worst_tmp.index[1]), value=worst_tmp.loc[worst_tmp.index[1], 'n_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[1], 'n_purchase'] - media_vendas,1))
    else:
        st.metric(label=str(best_tmp.index[1]), value=best_tmp.loc[best_tmp.index[1], 'cpa_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[1], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[1]), value=worst_tmp.loc[worst_tmp.index[1], 'cpa_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[1], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')

with inner_3:
    if option_2 == 'Valor gasto':
        st.metric(label=str(best_tmp.index[2]), value=best_tmp.loc[best_tmp.index[2], 'Valor gasto (R$)'], delta=round(best_tmp.loc[best_tmp.index[2], 'spend'] - valor_por_adset,0))
        st.metric(label=str(worst_tmp.index[2]), value=worst_tmp.loc[worst_tmp.index[2], 'Valor gasto (R$)'], delta=round(worst_tmp.loc[worst_tmp.index[2], 'spend'] - valor_por_adset,1))
    elif option_2 == 'Vendas totais':
        st.metric(label=str(best_tmp.index[2]), value=best_tmp.loc[best_tmp.index[2], 'n_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[2], 'n_purchase'] - media_vendas,1))
        st.metric(label=str(worst_tmp.index[2]), value=worst_tmp.loc[worst_tmp.index[2], 'n_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[2], 'n_purchase'] - media_vendas,1))
    else:
        st.metric(label=str(best_tmp.index[2]), value=best_tmp.loc[best_tmp.index[2], 'cpa_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[2], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[2]), value=worst_tmp.loc[worst_tmp.index[2], 'cpa_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[2], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')

with inner_4:
    if option_2 == 'Valor gasto':
        st.metric(label=str(best_tmp.index[3]), value=best_tmp.loc[best_tmp.index[3], 'Valor gasto (R$)'], delta=round(best_tmp.loc[best_tmp.index[3], 'spend'] - valor_por_adset,1))
        st.metric(label=str(worst_tmp.index[3]), value=worst_tmp.loc[worst_tmp.index[3], 'Valor gasto (R$)'], delta=round(worst_tmp.loc[worst_tmp.index[3], 'spend'] - valor_por_adset,1))
    elif option_2 == 'Vendas totais':
        st.metric(label=str(best_tmp.index[3]), value=best_tmp.loc[best_tmp.index[3], 'n_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[3], 'n_purchase'] - media_vendas,1))
        st.metric(label=str(worst_tmp.index[3]), value=worst_tmp.loc[worst_tmp.index[3], 'n_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[3], 'n_purchase'] - media_vendas,1))
    else:
        st.metric(label=str(best_tmp.index[3]), value=best_tmp.loc[best_tmp.index[3], 'cpa_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[3], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[3]), value=worst_tmp.loc[worst_tmp.index[3], 'cpa_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[3], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')

with inner_5:
    if option_2 == 'Valor gasto':
        st.metric(label=str(best_tmp.index[4]), value=best_tmp.loc[best_tmp.index[4], 'Valor gasto (R$)'], delta=round(best_tmp.loc[best_tmp.index[4], 'spend'] - valor_por_adset,1))
        st.metric(label=str(worst_tmp.index[4]), value=worst_tmp.loc[worst_tmp.index[4], 'Valor gasto (R$)'], delta=round(worst_tmp.loc[worst_tmp.index[4], 'spend'] - valor_por_adset,1))
    elif option_2 == 'Vendas totais':
        st.metric(label=str(best_tmp.index[4]), value=best_tmp.loc[best_tmp.index[4], 'n_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[4], 'n_purchase'] - media_vendas,1))
        st.metric(label=str(worst_tmp.index[4]), value=worst_tmp.loc[worst_tmp.index[4], 'n_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[4], 'n_purchase'] - media_vendas,1))
    else:
        st.metric(label=str(best_tmp.index[4]), value=best_tmp.loc[best_tmp.index[4], 'cpa_purchase'].round(2), delta=round(best_tmp.loc[best_tmp.index[4], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[4]), value=worst_tmp.loc[worst_tmp.index[4], 'cpa_purchase'].round(2), delta=round(worst_tmp.loc[worst_tmp.index[4], 'cpa_purchase'] - cpa_medio,1), delta_color='inverse')

map_option = {'Valor gasto':'spend', 'CPA':'cpa_purchase', 'Vendas totais':'n_purchase'}

tmp = fb_data[['date', 'name', 'spend', 'n_purchase']].groupby(by=['date', 'name']).sum()
tmp['CPA'] = tmp['spend'] / tmp['n_purchase']
tmp = tmp.loc[tmp.index.get_level_values('name').isin(selected_adsets)]

hist_fig = px.line(data_frame=tmp, x=tmp.index.get_level_values('date'), y=map_option.get(option_2),
                    color=tmp.index.get_level_values('name'), title=f'Evolução no tempo do {selected_adsets} em relação a métrica {option_2}')
st.plotly_chart(hist_fig, use_container_width=True)
