import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime, timedelta
from millify import millify
import numpy as np
import plotly.graph_objects as go
from io import StringIO
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
    
@st.cache_data
def get_data_from_bucket(bucket_name: str) -> bool:

    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_STORAGE"])
    client = storage.Client(credentials=credentials)
    source_bucket_name = bucket_name
    bucket = client.bucket(source_bucket_name)
    blob = bucket.blob('processed_adsets.csv')
    blob_content = blob.download_as_text()
    return blob_content

@st.cache_data
def process_data():
    tmp_file = get_data_from_bucket(bucket_name='dashboard_marketing_processed')
    fb_data = pd.read_csv(StringIO(tmp_file))
    fb_data = get_custom_metrics(fb_data)
    fb = fb_data.loc[(fb_data['campaign_name'] == '[CONVERSAO] [DIP] Broad')].copy()
    return fb

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
#Process
fb = process_data()

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
tmp['cpa_purchase'] = tmp['spend'] / tmp['n_purchase']
tmp = tmp.loc[tmp.index.get_level_values('name').isin(selected_adsets)]#.replace(np.inf, -1)


hist_fig = go.Figure()
for name in tmp.index.get_level_values('name').unique():
    aux = tmp.loc[tmp.index.get_level_values('name') == name]
    hist_fig.add_trace(go.Scatter(x=aux.index.get_level_values('date'), y=aux[map_option.get(option_2)], mode='lines+markers', name=name))

st.plotly_chart(hist_fig, use_container_width=True)
