import pandas as pd
import streamlit as st
import plotly.express as px
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime, timedelta
from millify import millify
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
from facebook_business.adobjects.adcreative import AdCreative
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.ad import Ad
import streamlit.components.v1 as components
from facebook_business.adobjects.adaccount import AdAccount
import requests

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
        mock_df['date'] = pd.to_datetime(mock_df['date'])
        mock_df['date'] = mock_df['date'].dt.date
        mock_df.sort_values(by='date', inplace=True)
        return mock_df
    
@st.cache_data(ttl='12h')
def get_data_from_bucket(bucket_name: str, file_name: str) -> bool:

    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_STORAGE"])
    client = storage.Client(credentials=credentials)
    source_bucket_name = bucket_name
    bucket = client.bucket(source_bucket_name)
    blob = bucket.blob(file_name)
    blob_content = blob.download_as_text()
    return blob_content

@st.cache_data(ttl='12h')
def process_data(file_name):
    tmp_file = get_data_from_bucket(bucket_name='dashboard_marketing_processed', file_name=file_name)
    fb_data = pd.read_csv(StringIO(tmp_file))
    fb_data = get_custom_metrics(fb_data)
    fb_data['action_value_purchase'].fillna(value=0, inplace=True)
    fb_data['lucro'] = fb_data['action_value_purchase'] - fb_data['spend']
    fb_data['lucro'] = fb_data['lucro'].round(2)
    fb = fb_data.loc[(fb_data['campaign_name'] == '[CONVERSAO] [DIP] Broad')].copy()
    return fb

def get_advideos(hash, access_token):
    url = f"https://graph.facebook.com/v18.0/{hash}"

    headers = {
        "Host": "graph.facebook.com",
        "Authorization": f"Bearer {access_token}",
    }
    
    params = {
    "fields": "embed_html",
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    html = data.get("embed_html")
    return html

def show_video(hash, access_token, height, width):
    
    link = get_advideos(hash=hash, access_token=access_token)
    link = link.replace('height="1920"',  f'height="{height}"')
    link = link.replace('width="1080"',  f'width="{width}"')
    components.html(link, height=height, width=width)
    return

def get_adimage(ad_account, img_hash):
    account = AdAccount(ad_account)
    params = {
        'hashes': [img_hash],
    }
    images = account.get_ad_images(params=params, fields=['url'])
    return images[0].get('url')

@st.cache_data
def group_data(df):
    grouped_fb = df[['name', 'spend', 'n_purchase', 'lucro', 'n_post_engagement']].groupby(by=['name']).sum()
    grouped_fb['lucro'] = grouped_fb['lucro'].round(2)
    grouped_fb['Valor gasto (%)'] = (grouped_fb['spend']/grouped_fb['spend'].sum()) * 100
    grouped_fb['Valor gasto (%)'] = grouped_fb['Valor gasto (%)'].round(1)
    grouped_fb['cpa_purchase'] = round(grouped_fb['spend'] / grouped_fb['n_purchase'],2)
    grouped_fb['Valor gasto (R$)'] = grouped_fb['spend'].apply(lambda x: millify(x, precision=1))
    return grouped_fb

def get_preview(ad_id):
    creativeID = Ad(ad_id).get_ad_creatives()[0]["id"]
    fields = [
    ]
    params = {
      'ad_format': 'INSTAGRAM_STANDARD',
    }
    tmp = AdCreative(creativeID).get_previews(fields=fields, params=params)
    tmp = tmp[0]['body']
    return tmp.replace(';t', '&t')

def get_adsets_ativos(date_range, fb_data):
    if date_range[0] < date_range[1]:
        g_data = fb_data[['name', 'date']].groupby(by='name').count()
        adsets_ativos = g_data.loc[g_data['date'] > 1].index.get_level_values('name')
        return adsets_ativos
    else:
        return None
def get_global_metrics(df):
    metricas = {}
    metricas['alcance'] = df['reach'].sum()
    metricas["frequencia"] = df['impressions'].sum()/df['reach'].sum()
    metricas['cpc'] = df['spend'].sum() / df['inline_link_clicks'].sum()
    metricas['true_visits'] = df['n_landing_page_view'].sum() / df['inline_link_clicks'].sum()
    metricas['cptv'] = metricas['cpc'] / metricas['true_visits']
    metricas['cpm'] = df['spend'].sum() / (df['impressions'].sum()/1000)
    metricas['lp_views'] = df['n_landing_page_view'].sum()
    metricas['custo_reaçao'] = df['spend'].sum() / df['n_post_reaction'].sum()
    metricas['custo_comentario'] = df['spend'].sum() / df['n_comments'].sum()
    metricas['custo_compartilhamento'] = df['spend'].sum() / df['n_shares'].sum()
    return metricas



###################### GETTING THE DATA #########################################
# DATA LOAD
access_token = st.secrets['FACEBOOK']['access_token']
act_id = st.secrets['FACEBOOK']['act_id']
#Process
FacebookAdsApi.init(access_token=access_token)
fb = process_data('processed_adsets.csv')
dct_ads = process_data('processed_ads_by_media.csv')
dct_ads['ad_id'] = dct_ads['ad_id'].astype(str)
ads = process_data('processed_ads.csv')
ads['ad_id'] = ads['ad_id'].astype(str)
#saving data in session_state
st.session_state['fb'] = fb
st.session_state['ads'] = ads
st.session_state['dct'] = dct_ads 
# FILTOS
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1))
fb_data = fb.loc[(fb['date'] >= date_range[0]) &(fb['date'] <= date_range[1])].copy()
adsets_ativos = get_adsets_ativos(fb_data=fb_data, date_range=date_range)
more_than_one_day = st.sidebar.radio(label='Somente adsets ativos há mais de um dia?', options=['Sim', 'Não'], horizontal=True)
if (more_than_one_day == 'Sim')&(date_range[0] != date_range[1]):
    fb_data = fb_data.loc[fb_data['name'].isin(adsets_ativos)].copy()

selected_adsets = st.sidebar.multiselect(label="Adsets", options=fb_data['name'].unique(), default=fb_data['name'].unique()[0])
limited_dct = dct_ads.loc[dct_ads['adset_name'].isin(selected_adsets) & (dct_ads['date'] >= date_range[0]) & (dct_ads['date'] <= date_range[1])]
limited_ads = ads.loc[ads['adset_name'].isin(selected_adsets) & (ads['date'] >= date_range[0]) & (ads['date'] <= date_range[1])]

##################### GETTING SOME NUMBERS ######################################
n_adsets = fb_data['name'].unique().shape[0]
Total_vendas_fb = fb_data['n_purchase'].sum().astype(int)
Total_gasto_fb = fb_data['spend'].sum()
grouped_fb = group_data(fb_data)
medias = {'Valor gasto': round(fb_data['spend'].sum()/n_adsets, 1),
          'Vendas totais': round(Total_vendas_fb/n_adsets,1),
          'CPA': round(grouped_fb['cpa_purchase'].replace(np.inf, np.nan).dropna().mean(),1),
          'Lucro': round(grouped_fb['lucro'].sum()/n_adsets, 1),
          'Engajamento': round(grouped_fb['n_post_engagement'].sum() / n_adsets,1)    
           }
nota_de_corte = Total_gasto_fb/n_adsets * 0.2
faturamento = fb_data['action_value_purchase'].sum()
ROAS = faturamento / Total_gasto_fb
lucro = faturamento - Total_gasto_fb
metricas_globais = get_global_metrics(fb_data)

######################### Start #########################################
st.title('Analise Semanal do desempenho no Facebook')
col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.metric(label='Investimento Facebook', value=millify(fb_data['spend'].sum(), precision=1))
    st.metric(label='Faturamento - (Lucro)', value=f'{millify(faturamento, precision=1)} - ({millify(lucro, precision=1)})')
    st.metric(label='ROAS', value=ROAS.round(2))
    st.metric(label='Vendas pelo Facebook', value=Total_vendas_fb)
with col_2:
    st.metric(label='CPC - (CPTV)', value=f'{round(metricas_globais["cpc"],2)} - ({round(metricas_globais["cptv"],2)})')
    st.metric(label='CPM', value=round(metricas_globais['cpm'],2))
    st.metric(label='Visualizações da página de destino', value=round(metricas_globais['lp_views'],2))
with col_3:
    st.metric(label='Custo por reação', value=round(metricas_globais['custo_reaçao'],2))
    st.metric(label='Custo por comentário', value=round(metricas_globais['custo_comentario'],2))
    st.metric(label='Custo por compartilhamento', value=round(metricas_globais['custo_compartilhamento'],2))

option_2 = st.radio(label="Selecione a métrica", options=['Valor gasto', 'CPA', 'Lucro', 'Engajamento'], horizontal=True)
map_option = {'Valor gasto':'spend', 'CPA':'cpa_purchase', 'Lucro':'lucro', 'Engajamento':'n_post_engagement'}

if option_2 == 'CPA':
    grouped_fb.sort_values(by='cpa_purchase', inplace=True, ascending=False)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
    metrica_fig.add_vline(x=medias[option_2], line_dash= 'dash', line_color='grey')

elif option_2 == 'Valor gasto':
    grouped_fb.sort_values(by=map_option.get(option_2), inplace=True, ascending=True)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
    metrica_fig.add_vline(x=nota_de_corte, line_dash='dash', line_color='red')
    metrica_fig.add_vline(x=medias[option_2], line_dash= 'dash', line_color='grey')

elif option_2 == 'Lucro':
    grouped_fb.sort_values(by=map_option.get(option_2), inplace=True, ascending=True)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='lucro')
    metrica_fig.add_vline(x=medias.get(option_2), line_dash= 'dash', line_color='grey')    
else:
    grouped_fb.sort_values(by=map_option.get(option_2), inplace=True, ascending=True)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
    metrica_fig.add_vline(x=medias.get(option_2), line_dash= 'dash', line_color='grey')

st.plotly_chart(metrica_fig, use_container_width=True)
best_tmp = grouped_fb.tail(5)
worst_tmp = grouped_fb.head(5)

#Ajustando o valor gasto para números amigáveis
pretty_values_best = best_tmp['spend'].apply(lambda x: millify(x, precision=1))
pretty_values_best = pretty_values_best.to_numpy().reshape((1, 5))
pretty_values_worst = worst_tmp['spend'].apply(lambda x: millify(x, precision=1))
pretty_values_worst = pretty_values_worst.to_numpy().reshape((1, 5))
fig = make_subplots(rows=1, cols=2, column_titles=[f'5 melhores segundo a métrica {option_2}', f'5 piores segundo a métrica {option_2}'], shared_yaxes=True)

hover_template = 'Valor Gasto: %{customdata}<br> Métrica: %{y}'
fig.add_trace(
    go.Bar(x=best_tmp.index, y=best_tmp[map_option.get(option_2)],
           customdata=pretty_values_best.ravel(), hovertemplate=hover_template),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=worst_tmp.index, y=worst_tmp[map_option.get(option_2)],
           customdata=pretty_values_worst.ravel(), hovertemplate=hover_template), row=1, col=2)

st.plotly_chart(fig, use_container_width=True)
st.divider()

tmp = fb_data[['date', 'name', 'spend', 'n_purchase', 'lucro', 'n_post_engagement']].groupby(by=['date', 'name']).sum()
tmp['cpa_purchase'] = tmp['spend'] / tmp['n_purchase']
tmp = tmp.loc[tmp.index.get_level_values('name').isin(selected_adsets)]

hist_fig = go.Figure()
for name in tmp.index.get_level_values('name').unique():
    aux = tmp.loc[tmp.index.get_level_values('name') == name]
    hist_fig.add_trace(go.Scatter(x=aux.index.get_level_values('date'), y=aux[map_option.get(option_2)], mode='lines+markers', name=name))

hist_fig.update_layout(title= f'Evolução da metrica {option_2} para {selected_adsets} no periodo', yaxis_title=option_2)
st.plotly_chart(hist_fig, use_container_width=True)

