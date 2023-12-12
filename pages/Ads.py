import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from analise_dash import get_adsets_ativos, show_video, get_adimage, get_preview

########## Secrets ############################
access_token = st.secrets['FACEBOOK']['access_token']
act_id = st.secrets['FACEBOOK']['act_id']

########### DATA LOAD #########################
dct_ads = st.session_state['dct']
ads = st.session_state['ads']
fb = st.session_state['fb']

######### FILTERS ############################
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1))
fb_data = fb.loc[(fb['date'] >= date_range[0]) &(fb['date'] <= date_range[1])].copy()
adsets_ativos = get_adsets_ativos(fb_data=fb_data, date_range=date_range)
more_than_one_day = st.sidebar.radio(label='Somente adsets ativos há mais de um dia?', options=['Sim', 'Não'], horizontal=True)
if more_than_one_day == 'Sim':
    fb_data = fb_data.loc[fb_data['name'].isin(adsets_ativos)].copy()

selected_adset = st.sidebar.selectbox(label="Adsets", options=fb_data['name'].unique())
limited_dct = dct_ads.loc[(dct_ads['adset_name'] == selected_adset) & (dct_ads['date'] >= date_range[0]) & (dct_ads['date'] <= date_range[1])]
limited_ads = ads.loc[(ads['adset_name'] == selected_adset) & (ads['date'] >= date_range[0]) & (ads['date'] <= date_range[1])]
metric =  st.radio(label="Selecione a métrica", options=['Valor gasto', 'Vendas totais', 'CPA'], horizontal=True, key='ads_option')
##################### START ####################
map_option = {'Valor gasto':'spend', 'CPA':'cpa_purchase', 'Vendas totais':'n_purchase'}
tmp_dct = limited_dct[['adset_name', 'ad_id', 'name', 'video_name','asset_type', 'spend', 'n_purchase', 'cpa_purchase', 'hash', 'source_image_url', 'preview_link']].copy() #Pegando os dados de ads dct

if selected_adset not in limited_dct['adset_name'].values:
    not_dct_ad = True
else:
    not_dct_ad = False

tmp_dct.loc[~tmp_dct['video_name'].isna(), 'name'] = tmp_dct.loc[~tmp_dct['video_name'].isna(), 'video_name'].values
tmp_dct.drop(['video_name'], axis=1, inplace=True)

if not_dct_ad == True:
    tmp_plot = limited_ads[['adset_name', 'name', 'spend', 'n_purchase']].groupby(by=['adset_name', 'name']).sum()
    tmp_plot['cpa_purchase'] = round(tmp_plot['spend'] / tmp_plot['n_purchase'])
    tmp_plot.reset_index(inplace=True)
    prev = limited_ads.loc[limited_ads['adset_name'] == selected_adset]

elif not_dct_ad == False:
    tmp_plot = tmp_dct[['adset_name', 'name', 'spend', 'n_purchase']].groupby(by=['adset_name', 'name']).sum()
    tmp_plot['cpa_purchase'] = round(tmp_plot['spend'] / tmp_plot['n_purchase'])
    tmp_plot.reset_index(inplace=True)  
    prev = tmp_dct.loc[tmp_dct['adset_name'] == selected_adset]

if metric == 'CPA':
    tmp_plot.sort_values(by='cpa_purchase', inplace=True, ascending=False)
    ads_fig = px.bar(data_frame=tmp_plot, x='cpa_purchase', y='name', color='adset_name')
    
else:
    tmp_plot.sort_values(by=map_option.get(metric), inplace=True, ascending=True)
    ads_fig = px.bar(data_frame=tmp_plot, x=map_option.get(metric), y='name', color='adset_name')

st.plotly_chart(ads_fig, use_container_width=True)

col_0, col_1, col_2 = st.columns(3)
if not_dct_ad == False:  
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
    for i, name in enumerate(prev['name'].unique()):
        creative = prev.loc[prev['name'] == name]
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
