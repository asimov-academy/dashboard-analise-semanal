import pandas as pd
import streamlit as st
import plotly.express as px
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime, timedelta
from millify import millify
import numpy as np
import plotly.graph_objects as go
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
    grouped_fb = df[['name', 'spend', 'n_purchase']].groupby(by=['name']).sum()
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
# FILTOS
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1))
fb_data = fb.loc[(fb['date'] >= date_range[0]) &(fb['date'] <= date_range[1])].copy()
adsets_ativos = get_adsets_ativos(fb_data=fb_data, date_range=date_range)
more_than_one_day = st.sidebar.radio(label='Somente adsets ativos há mais de um dia?', options=['Sim', 'Não'], horizontal=True)
if more_than_one_day == 'Sim':
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
           'CPA': round(grouped_fb['cpa_purchase'].replace(np.inf, np.nan).dropna().mean(),2)}
nota_de_corte = Total_gasto_fb/n_adsets * 0.2

######################### Start #########################################
st.title('Analise Semanal do desempenho no Facebook')
col_1, col_2 = st.columns(2)
with col_1:
    st.metric(label='Investimento Facebook', value=millify(fb_data['spend'].sum(), precision=1))
    st.metric(label='Vendas pelo Facebook', value=Total_vendas_fb)
with col_2:
    st.metric(label='Media de vendas por adset', value=medias.get('Vendas totais').round(2))
    st.metric(label='Gasto médio por adset', value=medias.get('Valor gasto').round(2))
    st.metric(label='CPA Médio', value=medias.get('CPA'))


option_2 = st.radio(label="Selecione a métrica", options=['Valor gasto', 'Vendas totais', 'CPA'], horizontal=True)
map_option = {'Valor gasto':'spend', 'CPA':'cpa_purchase', 'Vendas totais':'n_purchase'}

if option_2 == 'CPA':
    grouped_fb.sort_values(by='cpa_purchase', inplace=True, ascending=False)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'])

elif option_2 == 'Valor gasto':
    grouped_fb.sort_values(by=map_option.get(option_2), inplace=True, ascending=True)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'])
    metrica_fig.add_vline(x=nota_de_corte, line_dash='dash', line_color='red')

else:
    grouped_fb.sort_values(by=map_option.get(option_2), inplace=True, ascending=True)
    metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(option_2)], title=f'Distribuição da métrica {option_2} adset', color=grouped_fb[map_option.get(option_2)], hover_data=['Valor gasto (%)','Valor gasto (R$)'])

st.plotly_chart(metrica_fig, use_container_width=True)


col_1, col2 = st.columns(2)
best_tmp = grouped_fb.tail(5)
worst_tmp = grouped_fb.head(5)

with col_1:  
    top5_fig = px.bar(best_tmp, x=best_tmp.index, y=best_tmp[map_option.get(option_2)], title='Cinco melhores', hover_data=['Valor gasto (%)','Valor gasto (R$)'])
    st.plotly_chart(top5_fig, use_container_width=True)

with col2:
    worst5_fig = px.bar(worst_tmp, x=worst_tmp.index, y=worst_tmp[map_option.get(option_2)], title='Cinco piores',hover_data=['Valor gasto (%)','Valor gasto (R$)'] )
    worst5_fig.update_xaxes(tickangle=45)
    st.plotly_chart(worst5_fig, use_container_width=True)

st.divider()
st.subheader(f'Pódio com relação a métrica {option_2}')
inner_1, inner_2, inner_3, inner_4, inner_5 = st.columns(5)
with inner_1:
    if option_2 == 'CPA':
        st.metric(label=str(best_tmp.index[0]), value=best_tmp.loc[best_tmp.index[0], map_option.get(option_2)], delta=round(best_tmp.loc[best_tmp.index[0], map_option.get(option_2)] - medias.get(option_2),1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[0]), value=worst_tmp.loc[worst_tmp.index[0], map_option.get(option_2)], delta=round(worst_tmp.loc[worst_tmp.index[0],  map_option.get(option_2)] -  medias.get(option_2),1), delta_color='inverse')
    else:
        st.metric(label=str(best_tmp.index[0]), value=millify(best_tmp.loc[best_tmp.index[0], map_option.get(option_2)], precision=1), delta=round(best_tmp.loc[best_tmp.index[0], map_option.get(option_2)] - medias.get(option_2),1))
        st.metric(label=str(worst_tmp.index[0]), value=millify(worst_tmp.loc[worst_tmp.index[0], map_option.get(option_2)], precision=1), delta=round(worst_tmp.loc[worst_tmp.index[0],  map_option.get(option_2)] -  medias.get(option_2),1))

with inner_2:
    if option_2 == 'CPA':
        st.metric(label=str(best_tmp.index[1]), value=best_tmp.loc[best_tmp.index[1], map_option.get(option_2)],  delta=round(best_tmp.loc[best_tmp.index[1], map_option.get(option_2)] - medias.get(option_2),1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[1]), value=worst_tmp.loc[worst_tmp.index[1], map_option.get(option_2)], delta=round(worst_tmp.loc[worst_tmp.index[1],  map_option.get(option_2)] -  medias.get(option_2),1), delta_color='inverse')
    else:
        st.metric(label=str(best_tmp.index[1]), value=millify(best_tmp.loc[best_tmp.index[1], map_option.get(option_2)], precision=1), delta=round(best_tmp.loc[best_tmp.index[1], map_option.get(option_2)] - medias.get(option_2),1))
        st.metric(label=str(worst_tmp.index[1]), value=millify(worst_tmp.loc[worst_tmp.index[1], map_option.get(option_2)], precision=1), delta=round(worst_tmp.loc[worst_tmp.index[1],  map_option.get(option_2)] -  medias.get(option_2),1))

with inner_3:
    if option_2 == 'CPA':
        st.metric(label=str(best_tmp.index[2]), value= best_tmp.loc[best_tmp.index[2], map_option.get(option_2)], delta=round(best_tmp.loc[best_tmp.index[2], map_option.get(option_2)] - medias.get(option_2),1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[2]), value= worst_tmp.loc[worst_tmp.index[2], map_option.get(option_2)], delta=round(worst_tmp.loc[worst_tmp.index[2],  map_option.get(option_2)] -  medias.get(option_2),1), delta_color='inverse')
    else:
        st.metric(label=str(best_tmp.index[2]), value=millify(best_tmp.loc[best_tmp.index[2], map_option.get(option_2)], precision=1), delta=round(best_tmp.loc[best_tmp.index[2], map_option.get(option_2)] - medias.get(option_2),1))
        st.metric(label=str(worst_tmp.index[2]), value=millify(worst_tmp.loc[worst_tmp.index[2], map_option.get(option_2)], precision=1), delta=round(worst_tmp.loc[worst_tmp.index[2],  map_option.get(option_2)] -  medias.get(option_2),1))

with inner_4:
    if option_2 == 'CPA':
        st.metric(label=str(best_tmp.index[3]), value= best_tmp.loc[best_tmp.index[3], map_option.get(option_2)], delta=round(best_tmp.loc[best_tmp.index[3], map_option.get(option_2)] - medias.get(option_2),1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[3]), value= worst_tmp.loc[worst_tmp.index[3], map_option.get(option_2)], delta=round(worst_tmp.loc[worst_tmp.index[3],  map_option.get(option_2)] -  medias.get(option_2),1), delta_color='inverse')
    else:
        st.metric(label=str(best_tmp.index[3]), value=millify(best_tmp.loc[best_tmp.index[3], map_option.get(option_2)], precision=1), delta=round(best_tmp.loc[best_tmp.index[3], map_option.get(option_2)] - medias.get(option_2),1))
        st.metric(label=str(worst_tmp.index[3]), value=millify(worst_tmp.loc[worst_tmp.index[3], map_option.get(option_2)],precision=1), delta=round(worst_tmp.loc[worst_tmp.index[3],  map_option.get(option_2)] -  medias.get(option_2),1))

with inner_5:
    if option_2 == 'CPA':
        st.metric(label=str(best_tmp.index[4]), value=best_tmp.loc[best_tmp.index[4], map_option.get(option_2)], delta=round(best_tmp.loc[best_tmp.index[4], map_option.get(option_2)] - medias.get(option_2),1), delta_color='inverse')
        st.metric(label=str(worst_tmp.index[4]), value=worst_tmp.loc[worst_tmp.index[4], map_option.get(option_2)], delta=round(worst_tmp.loc[worst_tmp.index[4],  map_option.get(option_2)] -  medias.get(option_2),1), delta_color='inverse')
    else:
        st.metric(label=str(best_tmp.index[4]), value=millify(best_tmp.loc[best_tmp.index[4], map_option.get(option_2)], precision=1), delta=round(best_tmp.loc[best_tmp.index[4], map_option.get(option_2)] - medias.get(option_2),1))
        st.metric(label=str(worst_tmp.index[4]), value=millify(worst_tmp.loc[worst_tmp.index[4], map_option.get(option_2)], precision=1), delta=round(worst_tmp.loc[worst_tmp.index[4],  map_option.get(option_2)] -  medias.get(option_2),1))
st.divider()

tmp = fb_data[['date', 'name', 'spend', 'n_purchase']].groupby(by=['date', 'name']).sum()
tmp['cpa_purchase'] = tmp['spend'] / tmp['n_purchase']
tmp = tmp.loc[tmp.index.get_level_values('name').isin(selected_adsets)]

hist_fig = go.Figure()
for name in tmp.index.get_level_values('name').unique():
    aux = tmp.loc[tmp.index.get_level_values('name') == name]
    hist_fig.add_trace(go.Scatter(x=aux.index.get_level_values('date'), y=aux[map_option.get(option_2)], mode='lines+markers', name=name))

hist_fig.update_layout(title= f'Evolução da metrica {option_2} para {selected_adsets} no periodo', yaxis_title=option_2)
st.plotly_chart(hist_fig, use_container_width=True)
st.divider()
st.subheader('Análise criativos')

tmp_1 = limited_dct[['adset_name', 'ad_id', 'name', 'video_name','asset_type', 'spend', 'n_purchase', 'cpa_purchase', 'hash', 'source_image_url', 'preview_link']] #Pegando os dados de ads dct
not_dct_ad = set(selected_adsets) - set(tmp_1['adset_name'])
ads_data_concat = pd.concat([tmp_1, limited_ads.loc[limited_ads['adset_name'].isin(not_dct_ad), ['adset_name', 'ad_id', 'name', 'spend', 'n_purchase', 'cpa_purchase', 'preview_link']]])
ads_data_concat.loc[~ads_data_concat['video_name'].isna(), 'name'] = ads_data_concat.loc[~ads_data_concat['video_name'].isna(), 'video_name'].values
ads_data_concat.drop(['video_name'], axis=1, inplace=True)

option_3 =  st.radio(label="Selecione a métrica", options=['Valor gasto', 'Vendas totais', 'CPA'], horizontal=True, key='ads_option')
tmp_plot = ads_data_concat[['adset_name', 'name', 'spend', 'n_purchase']].groupby(by=['adset_name', 'name']).sum()
tmp_plot['cpa_purchase'] = round(tmp_plot['spend'] / tmp_plot['n_purchase'])
tmp_plot.reset_index(inplace=True)
if option_3 == 'CPA':
    tmp_plot.sort_values(by='cpa_purchase', inplace=True, ascending=False)
    ads_fig = px.bar(data_frame=tmp_plot, x='cpa_purchase', y='name', color='adset_name')
    
else:
    tmp_plot.sort_values(by=map_option.get(option_3), inplace=True, ascending=True)
    ads_fig = px.bar(data_frame=tmp_plot, x=map_option.get(option_3), y='name', color='adset_name')

st.plotly_chart(ads_fig, use_container_width=True)
s_adset = st.selectbox(label='Selecione um adset para exibir os ads', options=selected_adsets, index=0)
prev = ads_data_concat.loc[ads_data_concat['adset_name'] == s_adset]

col_0, col_1, col_2 = st.columns(3)

if not prev['asset_type'].isna().all():  
    for i, name in enumerate(prev['name'].unique()):
        creative = prev.loc[prev['name'] == name]

        if(creative['asset_type'] == 'video_asset').all(): #criativo do tipo video
            id_hash = creative[['ad_id', 'hash']].iloc[0].ravel()
            if i == 0:
                with col_0:
                    show_video(hash=id_hash[1], access_token=access_token, height=600, width=300)

            elif i == 1:
                with col_1:
                    show_video(hash=id_hash[1], access_token=access_token, height=600, width=300)
            else:
                with col_2:
                    show_video(hash=id_hash[1], access_token=access_token, height=600, width=300)
        
        elif(creative['asset_type'] == 'image_asset').all():
            id_hash = creative[['ad_id', 'hash']].iloc[0].ravel()
            
            if i == 0:
                with col_0:
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)

            elif i == 1:
                with col_1:
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)
            else:
                with col_2:
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)
else:  
    for i, name in enumerate(prev['name'].unique()):
        creative = prev.loc[prev['name'] == name]
        id_hash = creative[['ad_id']].iloc[0].ravel()
        if i == 0:
            with col_0:
                components.html(get_preview(id_hash[0]), width=300, height=600)

        elif i == 1:
            with col_1:
                components.html(get_preview(id_hash[0]), width=300, height=600)
        else:
            with col_2:
                components.html(get_preview(id_hash[0]), width=300, height=600)
