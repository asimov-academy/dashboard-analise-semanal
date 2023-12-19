import streamlit as st
import streamlit_authenticator as stauth
from analise_dash import get_data_from_bucket
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
from millify import millify

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
        st.session_state['hotmart_data'] = raw_hotmart
        hotmart = st.session_state['hotmart_data'] = raw_hotmart
    
    date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=7), datetime.today()-timedelta(days=1)), max_value=datetime.today()-timedelta(days=1), min_value=hotmart['order_date'].min(), key='hotmart_dates')
    limited_hotmart = hotmart.loc[(hotmart['order_date'] >= date_range[0])&(hotmart['order_date'] <= date_range[1])]
    st.write(limited_hotmart.head())
    hotmart_viables = limited_hotmart.loc[limited_hotmart['status'].isin(['APPROVED','REFUNDED', 'COMPLETE'])] #desprezando compras canceladas
    hotmart_no_duplicates = hotmart_viables.drop_duplicates(subset=['buyer_ucode']) # Talvez alguém apareça duas vezes pois pode ter realizado a compra em um período (STAtUS APPROVED)
                                                                                             # e depois que o periodo de reebolso acabou (STATUS COMPLETE) e apareceu 2x no banco hora como APPROVED hora como COMPLETE
    billing = hotmart_no_duplicates.loc[hotmart_no_duplicates['status'].isin(['APPROVED', 'COMPLETE']), 'profit'].sum()
    n_sales = hotmart_no_duplicates.shape[0]
    refunds = hotmart_no_duplicates.loc[hotmart_no_duplicates['status'] == 'REFUNDED'].shape[0]
    avarage_ticket = billing/n_sales

    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric('Faturamento', value = f' R${millify(billing, precision=1)}')
        st.metric('Ticket Médio', value=f'R$ {millify(avarage_ticket, precision=1)}')
    with col_2:
        st.metric('Vendas', value=n_sales)
        st.metric('Reembolsos', value=refunds)

    
    
