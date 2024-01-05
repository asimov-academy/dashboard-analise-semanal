import streamlit as st
import pandas as pd
from io import BytesIO
from analise_dash import get_data_from_bucket
from datetime import datetime, timedelta
from millify import millify
import plotly.express as px
import numpy as np

@st.cache_data
def get_sales_att(
        df:pd.DataFrame,

)-> dict:
    """
    Gets sales page attribution - each page that a user visit in a session that there is a purchase gets a percentual value,
      1/number of pages visited

    """
    sessions_with_sales = df.loc[df['event_name'] == 'purchase', 'ga_session_id'].unique()
    paths = {}
    points_per_session = {}
    path_points = {}
    for session in sessions_with_sales:
        pages = df.loc[(df['ga_session_id'] == session) & 
                                (df['event_name'] == 'page_view') &
                                ~(df['event_page_location'].str.contains('hotmart')), 'Path']
        if len(pages) > 0:
            paths[session] = pages
        else:
            paths[session] = ['direct']     
        
        points_per_session[session] = len(df.loc[(df['ga_session_id'] == session)&(df['event_name'] == 'purchase'), 'event_name']) / len(paths.get(session, 1))
        for path in paths.get(session):
            try:
                path_points[path] += points_per_session[session]
            except KeyError:
                path_points[path] = points_per_session[session]        
    return pd.DataFrame(list(path_points.items()), columns=['Path', 'Value'])

@st.cache_data
def get_ga4_metrics(df:pd.DataFrame) -> dict:
    metrics = dict()
    metrics['Sessões'] = df.loc[df['event_name'] == 'session_start', 'count'].sum()
    metrics['Visualizações'] = df.loc[df['event_name'] == 'page_view', 'count'].sum()
    metrics['N_vendas'] = df.loc[df['event_name'] == 'purchase', 'count'].sum()
    return metrics
try:
    ga4 = st.session_state['ga4']

except KeyError:
    tmp_ga4 = get_data_from_bucket(bucket_name='dashboard_marketing_processed', file_name='ga4_data_dash.parquet', file_type='.parquet')
    raw_ga4 = pd.read_parquet(BytesIO(tmp_ga4), engine='pyarrow')
    st.session_state['ga4'] = raw_ga4
    ga4 = st.session_state['ga4']

ga4['count'] = 1
########################## FILTERS ###############################################
date_range = st.date_input("Periodo atual", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1), min_value=ga4['event_date'].min(), key='ga4_dates')
dates_range_benchmark = st.sidebar.date_input("Periodo de para comparação", value=(datetime.today()-timedelta(days=14), datetime.today()-timedelta(days=8)), max_value=datetime.today()-timedelta(days=1), min_value=ga4['event_date'].min(), key='ga4_dates_benchmark')
limited_ga4 = ga4.loc[(ga4['event_date'].dt.date >= date_range[0]) & (ga4['event_date'].dt.date <= date_range[1])]
limited_benchmark = ga4.loc[(ga4['event_date'].dt.date >= dates_range_benchmark[0]) & (ga4['event_date'].dt.date <= dates_range_benchmark[1])]

######################## BEGIN #####################################
st.title('Dados GA4')
current_ga4_metrics = get_ga4_metrics(limited_ga4)
benchmark_ga4_metrics = get_ga4_metrics(limited_benchmark)

map_event = {'Sessões': 'session_start',
             'Visualizações': 'page_view'}
col_1, col_2 = st.columns(2)
with col_1:
    st.metric(label='Total de sessões', value=millify(current_ga4_metrics['Sessões'], precision=1), delta=int(current_ga4_metrics['Sessões'] - benchmark_ga4_metrics['Sessões']))
    st.metric(label='Total de Visualizações de página', value=millify(current_ga4_metrics['Visualizações'], precision=1), delta=int(current_ga4_metrics['Visualizações'] - benchmark_ga4_metrics['Visualizações']))
    st.metric(label='Total de vendas registradas no GA4', value=current_ga4_metrics['N_vendas'], delta=int(current_ga4_metrics['N_vendas'] - benchmark_ga4_metrics['N_vendas']))
with col_2:
    sales_att_data = pd.DataFrame(get_sales_att(limited_ga4)).round(2)
    sales_att_chart = px.pie(data_frame=sales_att_data, names='Path', values='Value', title='Contribuição das páginas por venda').update_traces(textinfo='value+percent')
    st.plotly_chart(sales_att_chart, use_container_width=True)

c1, c2 = st.columns(2)
############# Paths data #################################
paths = limited_ga4.loc[limited_ga4['event_name'] == 'session_start', 'Path'].value_counts().to_frame()
paths['%'] = paths['count']/paths['count'].sum()
paths_data = paths.loc[paths['%'] > 0.01]
paths_chart = px.pie(data_frame=paths_data, names=paths_data.index, values=paths_data['count'], title='Distribuição das sessões por página').update_traces(textinfo='value+percent')
with c1:
    st.plotly_chart(paths_chart, use_container_width=True)

########## Default channel ##############################
default_channel = limited_ga4.loc[limited_ga4['event_name'] == 'session_start', 'default_channel'].value_counts().to_frame()
channels_chart = px.pie(data_frame=default_channel, names=default_channel.index, values=default_channel['count'], title='Distribuição de sessões por canal')

with c2:
    st.plotly_chart(channels_chart, use_container_width=True)

path_details = st.expander('Detalhamento por página', True)
with path_details:
    s_path = st.selectbox('Selecione uma página de interesse', options=paths.index)
    details_path_data = limited_ga4.loc[(limited_ga4['event_name'] == 'session_start') & (limited_ga4['Path'] == s_path)]
    inner_col1, inner_col2 = st.columns(2)
    
    with inner_col1:
        tmp = details_path_data[['utm_source_std', 'default_channel','utm_content','count']].groupby(by=['utm_source_std', 'default_channel','utm_content'], observed=True).sum().reset_index()
        tmp[['utm_source_std', 'default_channel','utm_content']] = tmp[['utm_source_std', 'default_channel','utm_content']].astype(str)
        source_chart = px.sunburst(data_frame=tmp, title='Fontes de tráfego', values='count', path=['utm_source_std', 'default_channel', 'utm_content'], branchvalues='total', maxdepth=-1).update_traces(textinfo='label+value+percent entry')
        st.plotly_chart(source_chart, use_container_width=True)

    with inner_col2:
        selected_channel = st.selectbox('Selecione um canal de tráfego', options=limited_ga4['default_channel'].unique())
        current_channels_data = limited_ga4.loc[limited_ga4['default_channel'] == selected_channel, ['default_channel', 'event_date', 'count']].groupby(by=['event_date', 'default_channel'], observed=True).sum()
        current_channels_data['Periodo'] = 'Atual'
        comparison_channels_data = limited_benchmark.loc[limited_benchmark['default_channel'] == selected_channel, ['default_channel', 'event_date', 'count']].groupby(by=['event_date', 'default_channel'], observed=True).sum()
        comparison_channels_data['Periodo'] = 'Referência'
        channels_data = pd.concat([comparison_channels_data, current_channels_data])
        channels_data.sort_values(by='event_date', inplace=True)
        chanel_hist = px.line(data_frame=channels_data, x=channels_data.index.get_level_values('event_date'), y='count', color='Periodo',
                              title=f'Evolução do número de sessões para o canal {selected_channel}').update_layout(xaxis_title='Data', yaxis_title='Nº sessões diárias')
        st.plotly_chart(chanel_hist, use_container_width=True)

###################### TRILHAS ##########################################
trails_expander = st.expander('Trilhas', True)

dsml = limited_ga4.loc[(limited_ga4['event_name'] == 'session_start')&(limited_ga4['Path']=='/trilha-data-science-e-machine-learning/')]
quant = limited_ga4.loc[(limited_ga4['event_name'] == 'session_start')&(limited_ga4['Path']=='/trading-quantitativo/')]
pyof = limited_ga4.loc[(limited_ga4['event_name'] == 'session_start')&(limited_ga4['Path']=='/trilha-python-office/')]
dip = limited_ga4.loc[(limited_ga4['event_name'] == 'session_start')&(limited_ga4['Path'].isin(['/dashboards-interativos-com-python/', '/dashboards-interativos-com-python-2/']))]

with trails_expander:
    tcol_1, tcol_2, tcol_3, tcol_4 = st.columns(4)
    with tcol_1:
        tmp = dip[['utm_source_std', 'default_channel','utm_content','count']].groupby(by=['utm_source_std', 'default_channel','utm_content'], observed=True).sum().reset_index()
        tmp[['utm_source_std', 'default_channel','utm_content']] = tmp[['utm_source_std', 'default_channel','utm_content']].astype(str)
        dip_chart = px.sunburst(data_frame=tmp, title='DIP', values='count', path=['utm_source_std', 'default_channel', 'utm_content'], branchvalues='total', maxdepth=-1).update_traces(textinfo='label+value+percent entry')
        st.plotly_chart(dip_chart, use_container_width=True)
    
    with tcol_2:
        tmp = pyof[['utm_source_std', 'default_channel','utm_content','count']].groupby(by=['utm_source_std', 'default_channel','utm_content'], observed=True).sum().reset_index()
        tmp[['utm_source_std', 'default_channel','utm_content']] = tmp[['utm_source_std', 'default_channel','utm_content']].astype(str)
        pyof_chart = px.sunburst(data_frame=tmp, title='Python Office', values='count', path=['utm_source_std', 'default_channel', 'utm_content'], branchvalues='total', maxdepth=-1).update_traces(textinfo='label+value+percent entry')
        st.plotly_chart(pyof_chart, use_container_width=True)
    
    with tcol_3:
        tmp = dsml[['utm_source_std', 'default_channel','utm_content','count']].groupby(by=['utm_source_std', 'default_channel','utm_content'], observed=True).sum().reset_index()
        tmp[['utm_source_std', 'default_channel','utm_content']] = tmp[['utm_source_std', 'default_channel','utm_content']].astype(str)
        dsml_chart = px.sunburst(data_frame=tmp, title='DSML', values='count', path=['utm_source_std', 'default_channel', 'utm_content'], branchvalues='total', maxdepth=-1).update_traces(textinfo='label+value+percent entry')
        st.plotly_chart(dsml_chart, use_container_width=True)   
    
    with tcol_4:
        tmp = quant[['utm_source_std', 'default_channel','utm_content','count']].groupby(by=['utm_source_std', 'default_channel','utm_content'], observed=True).sum().reset_index()
        tmp[['utm_source_std', 'default_channel','utm_content']] = tmp[['utm_source_std', 'default_channel','utm_content']].astype(str)
        quant_chart = px.sunburst(data_frame=tmp, title='Quant', values='count', path=['utm_source_std', 'default_channel', 'utm_content'], branchvalues='total', maxdepth=-1).update_traces(textinfo='label+value+percent entry')
        st.plotly_chart(quant_chart, use_container_width=True)

