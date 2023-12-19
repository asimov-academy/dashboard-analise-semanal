import streamlit as st
import streamlit_authenticator as stauth
from analise_dash import get_data_from_bucket
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
from millify import millify
import plotly.express as px

def get_metrics(df: pd.DataFrame) -> dict:
    metrics = dict()
    metrics['billing'] = df['price.value'].sum()
    metrics['n_sales'] = df.shape[0]
    metrics['refunds'] = df.loc[df['status'] == 'REFUNDED'].shape[0]
    metrics['avarage_ticket'] = metrics['billing'] / metrics['n_sales']
    return metrics


authenticator = stauth.Authenticate(
    dict(st.secrets['credentials']),
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
    st.secrets['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar')
    st.title('Dados Hotmart')
    try:
        hotmart = st.session_state['hotmart_data']
    except:
        tmp_hotmart = get_data_from_bucket(bucket_name='dashboard_marketing_processed', file_name='processed_hotmart.parquet', file_type='.parquet')
        raw_hotmart = pd.read_parquet(BytesIO(tmp_hotmart))
        raw_hotmart['count'] = 1
        st.session_state['hotmart_data'] = raw_hotmart
        hotmart = st.session_state['hotmart_data'] = raw_hotmart
    
    ############# FILTRANDO OS DADOS ###########################################
    date_range = st.sidebar.date_input("Periodo atual", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1), min_value=hotmart['order_date'].min(), key='hotmart_dates')
    dates_benchmark_hotmart = st.date_input("Periodo de referência", value=(datetime.today()-timedelta(days=14), datetime.today()-timedelta(days=8)), max_value=datetime.today()-timedelta(days=1), min_value=hotmart['order_date'].min(), key='hotmart_dates_benchmark')
    limited_hotmart = hotmart.loc[(hotmart['order_date'] >= date_range[0]) & 
                                  (hotmart['order_date'] <= date_range[1]) & 
                                  (hotmart['status'].isin(['APPROVED','REFUNDED','COMPLETE']))] #desprezando compras canceladas
    
    benchmark = hotmart.loc[(hotmart['order_date'] >= dates_benchmark_hotmart[0]) & 
                            (hotmart['order_date'] <= dates_benchmark_hotmart[1]) & 
                            (hotmart['status'].isin(['APPROVED','REFUNDED','COMPLETE']))] #desprezando compras canceladas
    
    ################ CALCULOS #######################################################
    current_metrics = get_metrics(limited_hotmart)
    benchmark_metrics = get_metrics(benchmark)
    options = {'Faturamento' : 'price.value',
               'Vendas' : 'count'}

    
    ################ INICIO ########################
    col_1, col_2 = st.columns(2)
    
    with col_1:
        st.metric('Faturamento', value = f'R${millify(current_metrics["billing"], precision=1)}', delta = millify(current_metrics['billing'] - benchmark_metrics['billing'], precision=1))
        st.metric('Ticket Médio', value=f'R${millify(current_metrics["avarage_ticket"], precision=1)}', delta=millify(current_metrics['avarage_ticket'] - benchmark_metrics['avarage_ticket'], precision=1))
    
    with col_2:
        st.metric('Vendas', value=current_metrics['n_sales'], delta=current_metrics['n_sales'] - benchmark_metrics['n_sales'])
        st.metric('Reembolsos', value=current_metrics['refunds'], delta= current_metrics['refunds'] - benchmark_metrics['refunds'], delta_color='inverse')


    sck_figure = px.pie(data_frame=limited_hotmart, values='count', names= 'tracking.source_sck', hole=0.5, 
                        title='Distribuição das vendas por sck', height=600).update_traces(textinfo='percent+value').update_layout(showlegend=True)
    st.plotly_chart(sck_figure, use_container_width=True)

    hotmart_metric = st.selectbox(label='Selecione uma métrica para acompanhar a evolução', options=['Faturamento', 'Vendas'], index=1)
    historic_data = hotmart[['approved_date', 'price.value', 'count']].groupby(by='approved_date').sum()
    historic_data.sort_index(ascending=True, inplace=True)
    historic_fig = px.line(data_frame=historic_data, x=historic_data.index, y=options[hotmart_metric], title=f'Histórico da metrica: {hotmart_metric}')
    st.plotly_chart(historic_fig, use_container_width=True)


    

    
    
