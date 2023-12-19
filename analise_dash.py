import pandas as pd
import streamlit as st
import plotly.express as px
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime, timedelta
from millify import millify
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
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
    

def get_data_from_bucket(bucket_name: str, file_name: str, file_type: str = 'csv') -> bool:
    """Get file_name from google storage bucket (bucket_name)"""
    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_STORAGE"])
    client = storage.Client(credentials=credentials)
    source_bucket_name = bucket_name
    bucket = client.bucket(source_bucket_name)
    blob = bucket.blob(file_name)
    if file_type == 'csv':
        blob_content = blob.download_as_text()
    else:
        blob_content = blob.download_as_bytes()
    return blob_content

def upload_dataframe_to_gcs(bucket_name, dataframe, destination_blob_name):
    """Uploads a Pandas DataFrame to Google Cloud Storage in Feather format."""
    feather_buffer = BytesIO()
    dataframe.to_feather(feather_buffer)

    feather_buffer.seek(0)
    
    credentials = service_account.Credentials.from_service_account_info(st.secrets["GOOGLE_STORAGE"])
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(feather_buffer, content_type="application/octet-stream")
    return

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
    grouped_fb = df[['name', 'spend', 'n_purchase', 'lucro', 'n_post_engagement', 'action_value_purchase']].groupby(by=['name']).sum()
    grouped_fb['lucro'] = grouped_fb['lucro'].round(2)
    grouped_fb['Valor gasto (%)'] = (grouped_fb['spend']/grouped_fb['spend'].sum()) * 100
    grouped_fb['Valor gasto (%)'] = grouped_fb['Valor gasto (%)'].round(1)
    grouped_fb['cpa_purchase'] = round(grouped_fb['spend'] / grouped_fb['n_purchase'],2)
    grouped_fb['Valor gasto (R$)'] = grouped_fb['spend'].apply(lambda x: millify(x, precision=1))
    grouped_fb['ROAS'] = round(grouped_fb['action_value_purchase']/grouped_fb['spend'],2)
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
    metricas['investimento'] = df['spend'].sum()
    metricas['faturamento'] = df['action_value_purchase'].sum()
    metricas['roas'] = metricas['faturamento'] / metricas['investimento']
    metricas['lucro'] = metricas['faturamento'] - metricas['investimento']
    return metricas

def update_annotations(old_annotations, new_annotations):
    old_annotations.update(new_annotations)
    upload_dataframe_to_gcs(bucket_name='dashboard_marketing_processed', dataframe=old_annotations, destination_blob_name='annotations_df.feather')
    return

###################### GETTING THE DATA #########################################
# DATA LOAD
access_token = st.secrets['FACEBOOK']['access_token']
act_id = st.secrets['FACEBOOK']['act_id']
tmp_annotations = get_data_from_bucket(bucket_name='dashboard_marketing_processed', file_name='annotations_df.feather', file_type='.feather')
annotations_df = pd.read_feather(BytesIO(tmp_annotations))
annotations_df['big_idea'] = annotations_df['big_idea'].astype(str)
annotations_df['awareness_level'] = annotations_df['awareness_level'].astype(str)

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

# FILTRANDO OS DADOS
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1))
fb_data = fb.loc[(fb['date'] >= date_range[0]) &(fb['date'] <= date_range[1])].copy()
metric_options = ['Valor gasto', 'CPA', 'Lucro', 'Engajamento', 'ROAS']
metric = st.sidebar.radio(label="Selecione a métrica", options=metric_options, horizontal=True)
map_option = {'Valor gasto':'spend', 'CPA':'cpa_purchase', 'Lucro':'lucro', 'Engajamento':'n_post_engagement', 'ROAS':'ROAS'}

# Pegando os dados do mes de referência
dates_benchmark = st.date_input(label='Escolha o período de referência', value=[datetime.strptime('2023-10-01', '%Y-%m-%d'), datetime.strptime('2023-10-31', '%Y-%m-%d')])
fb_benchmark = fb.loc[(fb['date'] >= dates_benchmark[0]) & (fb['date'] <= dates_benchmark[1])].copy()
limited_annotations = annotations_df.loc[annotations_df.index.isin(fb_data['adset_name'].unique())]

# Pegando o número de adsets
adsets_ativos = get_adsets_ativos(fb_data=fb_data, date_range=date_range)
adsets_ativos_benchmark = get_adsets_ativos(fb_data=fb_benchmark, date_range=dates_benchmark)
more_than_one_day = st.sidebar.radio(label='Somente adsets ativos há mais de um dia?', options=['Sim', 'Não'], horizontal=True)
if (more_than_one_day == 'Sim')&(date_range[0] != date_range[1]):
    fb_data = fb_data.loc[fb_data['name'].isin(adsets_ativos)].copy()
    fb_benchmark = fb_benchmark.loc[fb_benchmark['name'].isin(adsets_ativos_benchmark)].copy()

##################### GETTING SOME NUMBERS ######################################
n_adsets = fb_data['name'].unique().shape[0]
metricas_globais = get_global_metrics(fb_data)
referência_globais = get_global_metrics(fb_benchmark)
Total_vendas_fb = fb_data['n_purchase'].sum().astype(int)

grouped_fb = group_data(fb_data)
grouped_fb = grouped_fb.merge(annotations_df, left_index=True, right_index=True, how='left')
medias = {'Valor gasto': round(fb_data['spend'].sum()/n_adsets, 1),              #Medidas em relação a todo o periodo selecionado
          'Vendas totais': round(Total_vendas_fb/n_adsets,1),
          'CPA': round(fb_data['spend'].sum()/fb_data['n_purchase'].sum(), 2),
          'Lucro': round(grouped_fb['lucro'].sum()/n_adsets, 1),
          'Engajamento': round(fb_data['n_post_engagement'].sum() / n_adsets,1), 
          'ROAS': round(fb_data['action_value_purchase'].sum()/fb_data['spend'].sum())   
           }
nota_de_corte = metricas_globais['investimento']/n_adsets * 0.2

######################### Start #########################################
st.title('Analise Semanal do desempenho no Facebook')
col_1, col_2, col_3 = st.columns(3)
with col_1:
    st.metric(label='Investimento Facebook', value=millify(metricas_globais['investimento'], precision=1), delta=millify(metricas_globais['investimento'] - referência_globais['investimento'], precision=1), delta_color='off')
    st.metric(label='Faturamento - (Lucro)', value=f'{millify(metricas_globais["faturamento"], precision=1)} - ({millify(metricas_globais["lucro"], precision=1)})')
    st.metric(label='ROAS', value=metricas_globais['roas'].round(2))
    st.metric(label='Vendas pelo Facebook', value=Total_vendas_fb, delta=Total_vendas_fb - fb_benchmark['n_purchase'].sum())
with col_2:
    st.metric(label='CPC - (CPTV)', value=f'{round(metricas_globais["cpc"],2)} - ({round(metricas_globais["cptv"],2)})',
              delta=f'{round(metricas_globais["cpc"] - referência_globais["cpc"],2)} - ({round(metricas_globais["cptv"] - referência_globais["cptv"],2)})',
              delta_color='inverse')
    st.metric(label='CPM', value=round(metricas_globais['cpm'],2), delta=round(metricas_globais['cpm'] - referência_globais["cpm"],2), delta_color='inverse')
    st.metric(label='Visualizações da página de destino', value=round(metricas_globais['lp_views'],2), delta=round(metricas_globais['lp_views'] - referência_globais['lp_views'], 2))
with col_3:
    st.metric(label='Custo por reação', value=round(metricas_globais['custo_reaçao'],2), delta=round(metricas_globais['custo_reaçao'] - referência_globais['custo_reaçao'], 2), delta_color='inverse')
    st.metric(label='Custo por comentário', value=round(metricas_globais['custo_comentario'],2), delta=round(metricas_globais['custo_comentario'] - referência_globais['custo_comentario'],2), delta_color='inverse')
    st.metric(label='Custo por compartilhamento', value=round(metricas_globais['custo_compartilhamento'],2), delta=round(metricas_globais['custo_compartilhamento'] - referência_globais['custo_compartilhamento'], 2), delta_color='inverse')

adset_expander = st.expander('Nível - Adset', True)
annotation_option = None

with adset_expander:
    annotations_indicator = st.checkbox('Usar dados de anotações (Big Idea e Awareness Level)', value=True)
    if annotations_indicator == True:
        annotation_option = st.sidebar.radio(label='opções', label_visibility='collapsed', options=annotations_df.columns)

    if metric == 'CPA':
        grouped_fb.sort_values(by='cpa_purchase', inplace=True, ascending=False)

        if annotation_option is None:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[map_option.get(metric)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
        
        else:
             metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[annotation_option].astype(str), hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')       
        
        metrica_fig.add_vline(x=medias[metric], line_dash= 'dash', line_color='grey')

    elif metric == 'Valor gasto':
        grouped_fb.sort_values(by=map_option.get(metric), inplace=True, ascending=True)
        if annotation_option is None:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[map_option.get(metric)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
            metrica_fig.add_vline(x=nota_de_corte, line_dash='dash', line_color='red')
            metrica_fig.add_vline(x=medias[metric], line_dash= 'dash', line_color='grey')
        else:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[annotation_option].astype(str), hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
            metrica_fig.add_vline(x=nota_de_corte, line_dash='dash', line_color='red')
            metrica_fig.add_vline(x=medias[metric], line_dash= 'dash', line_color='grey')

    elif metric == 'Lucro':
        grouped_fb.sort_values(by=map_option.get(metric), inplace=True, ascending=True)
        if annotation_option is None:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[map_option.get(metric)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='lucro')
            metrica_fig.add_vline(x=medias.get(metric), line_dash= 'dash', line_color='grey')
        else:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[annotation_option].astype(str), hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='lucro')
            metrica_fig.add_vline(x=medias.get(metric), line_dash= 'dash', line_color='grey')            
    else:
        grouped_fb.sort_values(by=map_option.get(metric), inplace=True, ascending=True)
        if annotation_option is None:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[map_option.get(metric)], hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
            metrica_fig.add_vline(x=medias.get(metric), line_dash= 'dash', line_color='grey')
        else:
            metrica_fig = px.bar(grouped_fb, y=grouped_fb.index, x=grouped_fb[map_option.get(metric)], title=f'Distribuição da métrica {metric} adset', color=grouped_fb[annotation_option].astype(str), hover_data=['Valor gasto (%)','Valor gasto (R$)'], height=800, width=300, text='n_purchase')
            metrica_fig.add_vline(x=medias.get(metric), line_dash= 'dash', line_color='grey')            

    st.plotly_chart(metrica_fig, use_container_width=True)
    best_tmp = grouped_fb.tail(5)
    worst_tmp = grouped_fb.head(5)

    #Ajustando o valor gasto para números amigáveis
    pretty_values_best = best_tmp['spend'].apply(lambda x: millify(x, precision=1))
    pretty_values_best = pretty_values_best.to_numpy().reshape((1, 5))
    pretty_values_worst = worst_tmp['spend'].apply(lambda x: millify(x, precision=1))
    pretty_values_worst = pretty_values_worst.to_numpy().reshape((1, 5))
    fig = make_subplots(rows=1, cols=2, column_titles=[f'5 melhores segundo a métrica {metric}', f'5 piores segundo a métrica {metric}'], shared_yaxes=True)

    hover_template = 'Valor Gasto: %{customdata}<br> Métrica: %{y}'
    fig.add_trace(
        go.Bar(x=best_tmp.index, y=best_tmp[map_option.get(metric)],
            customdata=pretty_values_best.ravel(), hovertemplate=hover_template),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=worst_tmp.index, y=worst_tmp[map_option.get(metric)],
            customdata=pretty_values_worst.ravel(), hovertemplate=hover_template), row=1, col=2)
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    scatter_metrics = st.multiselect('Selecione 2 métricas para o gráfico de dispersão', options=metric_options, max_selections=2, default=['Valor gasto', 'ROAS'])
    if len(scatter_metrics) == 2:
        if annotation_option is None:
            scateer_fig = px.scatter(data_frame=grouped_fb, x=map_option.get(scatter_metrics[0]), y=map_option.get(scatter_metrics[1]),
                                 color=grouped_fb.index, color_discrete_sequence=px.colors.qualitative.Light24)
            scateer_fig.update_layout(showlegend=False)

        else:
            scateer_fig = px.scatter(data_frame=grouped_fb, x=map_option.get(scatter_metrics[0]), y=map_option.get(scatter_metrics[1]),
                                 color=grouped_fb[annotation_option].astype(str), color_discrete_sequence=px.colors.qualitative.Light24)
           
        st.plotly_chart(scateer_fig, use_container_width=True)         
        
ads_expander = st.expander('Análise pontual', True)
with ads_expander:
    selected_adsets = st.multiselect(label="Selecione um ou mais Adsets", options=fb_data['name'].unique())
    tmp = fb_data[['date', 'name', 'spend', 'n_purchase', 'lucro', 'n_post_engagement','action_value_purchase']].groupby(by=['date', 'name']).sum()
    tmp['cpa_purchase'] = tmp['spend'] / tmp['n_purchase']
    tmp['ROAS'] = round(tmp['action_value_purchase'] / tmp['spend'],2)
    tmp = tmp.loc[tmp.index.get_level_values('name').isin(selected_adsets)]

    hist_fig = go.Figure()
    for name in tmp.index.get_level_values('name').unique():
        aux = tmp.loc[tmp.index.get_level_values('name') == name]
        hist_fig.add_trace(go.Scatter(x=aux.index.get_level_values('date'), y=aux[map_option.get(metric)], mode='lines+markers', name=name))

    hist_fig.update_layout(title= f'Evolução da metrica {metric} para {selected_adsets} no periodo', yaxis_title=metric)
    st.plotly_chart(hist_fig, use_container_width=True)

    # Adsets para a análise
    limited_dct = dct_ads.loc[dct_ads['adset_name'].isin(selected_adsets) & (dct_ads['date'] >= date_range[0]) & (dct_ads['date'] <= date_range[1])]
    limited_ads = ads.loc[ads['adset_name'].isin(selected_adsets) & (ads['date'] >= date_range[0]) & (ads['date'] <= date_range[1])]
    
    tmp_dct = limited_dct.loc[limited_dct['adset_name'].isin(selected_adsets)] #Pegando os dados de ads dct
    tmp_dct.loc[~tmp_dct['video_name'].isna(), 'name'] = tmp_dct.loc[~tmp_dct['video_name'].isna(), 'video_name'].values
    tmp_dct.drop(['video_name'], axis=1, inplace=True)
    
    not_dct = set(selected_adsets) - set(limited_dct['adset_name'])
    if len(not_dct) > 0:
        tmp_ads = limited_ads.loc[limited_ads['adset_name'].isin(not_dct)]
        tmp_creatives = pd.concat([tmp_dct, tmp_ads], axis=0)
    else:
        tmp_creatives = tmp_dct
    
    tmp_plot = tmp_creatives[['adset_name', 'name', 'spend', 'n_purchase', 'lucro', 'n_post_engagement','action_value_purchase']].groupby(by=['adset_name', 'name']).sum()
    tmp_plot['cpa_purchase'] = round(tmp_plot['spend'] / tmp_plot['n_purchase'])
    tmp_plot['ROAS'] = round(tmp['action_value_purchase']/tmp['spend'], 2)
    tmp_plot.reset_index(inplace=True)

    if metric == 'CPA':
        tmp_plot.sort_values(by='cpa_purchase', inplace=True, ascending=False)
        ads_fig = px.bar(data_frame=tmp_plot, x='cpa_purchase', y='name', color='adset_name')
        
    else:
        tmp_plot.sort_values(by=map_option.get(metric), inplace=True, ascending=True)
        ads_fig = px.bar(data_frame=tmp_plot, x=map_option.get(metric), y='name', color='adset_name')

    st.plotly_chart(ads_fig, use_container_width=True)
    
    selected_adset = st.selectbox(label="Selecione um Adset para explorar os criativos", options=tmp_creatives['adset_name'].unique())
    prev = tmp_creatives.loc[tmp_creatives['adset_name'] == selected_adset]

    col_0, col_1, col_2 = st.columns(3)  
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
            id_hash = creative[['ad_id', 'hash', 'name']].iloc[0].ravel()
            
            if i == 0:
                with col_0:
                    st.write(id_hash[2])
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)

            elif i == 1:
                with col_1:
                    st.write(id_hash[2])
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)
            else:
                with col_2:
                    st.write(id_hash[2])
                    st.image(get_adimage(act_id, id_hash[1]), use_column_width=True)
        else:  
                id_hash = creative[['ad_id']].iloc[0].ravel()
                if i == 0:
                    with col_0:
                        st.write(name)
                        components.html(get_preview(id_hash[0]), width=300, height=600)

                elif i == 1:
                    with col_1:
                        st.write(name)
                        components.html(get_preview(id_hash[0]), width=300, height=600)
                else:
                    with col_2:
                        st.write(name)
                        components.html(get_preview(id_hash[0]), width=300, height=600)
                        annotatios_exp = st.expander('Anotações')
annotatios_exp = st.expander('Anotações')
with annotatios_exp:
    new_annotations = st.data_editor(data=limited_annotations, use_container_width=True, column_config={'Unnamed: 0':st.column_config.TextColumn('Adset name'),
                                                                                      'big_idea':st.column_config.TextColumn('Big Idea'),
                                                                                      'awareness_level': 'Awareness_level'})
    save = st.button(label='Save')
    if save == True:
        update_annotations(old_annotations=annotations_df, new_annotations=new_annotations)