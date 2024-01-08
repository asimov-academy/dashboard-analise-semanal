import streamlit as st
import re
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from google.oauth2 import service_account
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import typing
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

tables = [
        'users',
        'usermeta',
        'activities',
        'courses_completed',
        'trails_completed',
        'credits',
        'posts',
        'postmeta',
        'terms',
        'term_relationships',
        'term_taxonomy']


# ===== Data Load =====
@st.cache_data(ttl=60000)
def run_query(tables) -> dict[typing.Any, pd.DataFrame]:
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    def read_csv(table) -> pd.DataFrame:
        df = pd.read_gbq(f"SELECT * FROM hub_data.{table}", 'scidata-299417', use_bqstorage_api=True, credentials=credentials)
        return df
    
    def read_dataframes_in_parallel(tables):
        with ThreadPoolExecutor() as executor:
            dataframes = list(executor.map(read_csv, tables))
        return dataframes

    list_dfs = read_dataframes_in_parallel(tables)
    return {i: j for i, j in zip(tables, list_dfs)}

### HOTMART V2 -> Pega o token automatica se o mesmo venceu
def datetime_to_milliseconds(dt, end=False) -> int:
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

    response = requests.post(url=url, headers=headers, params=params)

    if response.status_code == 200:
        hotmart_json = response.json()
        os.environ['HOTMART_TOKEN'] = json.dumps(hotmart_json)

    else:
        response.raise_for_status()

def get_hotmart_sales_hist(
        start_date: datetime,
        end_date: datetime) -> pd.DataFrame:
    """
    Busca na hotmart API os dados referentes a histórico de compras no periodo
    defino por start_date e end_date e retorna um DataFrame
    """
    def query_api(next_page_token=None):

        start = datetime_to_milliseconds(start_date)
        end = datetime_to_milliseconds(end_date, end=True)

        base_url = 'https://developers.hotmart.com/payments/api/v1/sales/history'

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

        response = requests.get(url=base_url, headers=headers, params=params)
        return response
    
    response = query_api()
    
    next_page = True
    df_hotmart = pd.DataFrame()
    while next_page:
        if response.status_code == 200:
            hotmart_json = response.json()
            try:
                tmp = pd.DataFrame(hotmart_json['items'])
                df_hotmart = pd.concat([df_hotmart, tmp])
            except Exception as e:
                st.write(e) #Add a logger
            if 'next_page_token' in hotmart_json['page_info']:
                response = query_api(next_page_token=hotmart_json['page_info']['next_page_token'])
            else:
                next_page = False         
        elif response.status_code == 401:
            hotmart_token = get_hotmart_token(os.environ['HOTMART_CRED'])
            get_hotmart_sales_hist(start_date=hotmart_token, end_date=start_date)
        elif response.status_code == 429:
            st.warning(f'Excesso de requisições: {response.status_code} aguardando 3s')
            time.sleep(3)
            get_hotmart_sales_hist(hotmart_token=hotmart_token, start_date=start_date, end_date=end_date)
        else:
            st.write(response.status_code)
            return None
    df_hotmart["date"] = df_hotmart["purchase"].apply(lambda x: datetime.utcfromtimestamp(x["approved_date"] / 1000))
    df_hotmart["price"] = df_hotmart["purchase"].apply(lambda x: float(x["price"]["value"]))
    df_hotmart["status"] = df_hotmart["purchase"].apply(lambda x: x["status"])
    df_hotmart["transaction"] = df_hotmart["purchase"].apply(lambda x: x["transaction"])

    df_hotmart["email"] = df_hotmart["buyer"].apply(lambda x: x["email"])
    df_hotmart["name"] = df_hotmart["buyer"].apply(lambda x: x["name"])
    df_hotmart["date"] = df_hotmart["date"].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
    df_hotmart["tracking"] = df_hotmart["purchase"].apply(lambda x: x["tracking"]["source_sck"] if ("tracking" in x) and ("source_sck" in x["tracking"]) else "")
    return df_hotmart

# ===== Data Process =====
@st.cache_data(ttl=6000)
def get_hub_users() -> pd.DataFrame:
    usermeta_ids = dfs["usermeta"]["user_id"].value_counts().index
    user_ids = list(set(dfs["users"].index).intersection(usermeta_ids))
    users = []

    for id_ in user_ids:
        users.append({"id": id_})
        for row in dfs["usermeta"][dfs["usermeta"]["user_id"] == id_].values:
            users[-1][row[2]] = row[3]
        users[-1]['email'] = dfs["users"].loc[id_]['user_email']
        users[-1]['user_registered'] = dfs["users"].loc[id_]['user_registered']
    df_users_hub = pd.DataFrame(users)

    padrao = r'"([^"]*)"'
    df_users_hub["role"] = df_users_hub["jaikj_capabilities"].apply(lambda x: re.search(padrao, x).group(1))
    return df_users_hub

def process_activities():
    dict_posts_temp = dfs["posts"].set_index("ID").to_dict()
    dfs["activities"]["post_title"] = dfs["activities"]["post_id"].map(dict_posts_temp["post_title"])
    dfs["activities"]["post_name"] = dfs["activities"]["post_id"].map(dict_posts_temp["post_name"])
    dfs["activities"]["post_type"] = dfs["activities"]["post_id"].map(dict_posts_temp["post_type"])
    dfs["activities"]["created_at"] = pd.to_datetime(dfs["activities"]["created_at"]).dt.tz_localize(None)
    dfs["activities"]["updated_at"] = pd.to_datetime(dfs["activities"]["updated_at"]).dt.tz_localize(None).dt.tz_localize('America/Sao_Paulo')

    # ID casa com object_id do dfs["term_relationships"]
    df_posts = dfs["posts"][["ID", "post_title", "post_type"]]
    dict_temp = dfs["terms"].set_index("term_id").to_dict()
    dfs["term_taxonomy"]["name"] = dfs["term_taxonomy"]["term_taxonomy_id"].map(dict_temp["name"])
    dfs["term_taxonomy"]["slug"] = dfs["term_taxonomy"]["term_taxonomy_id"].map(dict_temp["slug"])

    dfs["term_relationships"]["post_name"] = dfs["term_relationships"]["object_id"].map(df_posts.set_index("ID").to_dict()["post_title"])
    dfs["term_relationships"]["post_type"] = dfs["term_relationships"]["object_id"].map(df_posts.set_index("ID").to_dict()["post_type"])
    dfs["term_relationships"]["taxonomy_name"] = dfs["term_relationships"]["term_taxonomy_id"].map(dfs["term_taxonomy"].set_index("term_taxonomy_id").to_dict()["name"])
    dfs["term_relationships"]["taxonomy"] = dfs["term_relationships"]["term_taxonomy_id"].map(dfs["term_taxonomy"].set_index("term_taxonomy_id").to_dict()["taxonomy"])
    dfs["term_relationships"]["count"] = dfs["term_relationships"]["term_taxonomy_id"].map(dfs["term_taxonomy"].set_index("term_taxonomy_id").to_dict()["count"])

    dfs["activities"]["curso"] = dfs["activities"]["post_id"].map(dfs["term_relationships"][dfs["term_relationships"]["taxonomy"] == "curso"].set_index("object_id").to_dict()["taxonomy_name"])
    # dfs["activities"]["curso"].fillna("", inplace=True)
    dfs["activities"]["curso"] = dfs["activities"]["curso"].apply(lambda x: x.replace('&amp;', '&') if type(x) == type(str) else x)
    
    projects_query = (dfs["activities"]["post_type"] == "atividade") & (dfs["activities"]["curso"].isna())
    dfs["activities"].loc[projects_query, "projeto"] = dfs["activities"].loc[projects_query, "post_id"].map(dfs["term_relationships"][dfs["term_relationships"]["taxonomy"] == "project"].set_index("object_id").to_dict()["taxonomy_name"])
    st.session_state["course_ids"] = dfs["posts"][dfs["posts"]["post_type"] == "curso"].set_index("post_title").to_dict()["ID"]

def process_hub_users(df_users_hub):
    # Users treatment
    df_users_hub = df_users_hub[["id", "first_name", "last_name", "email", "role", "user_registered"]]
    df_users_hub.loc[:, "credits"] = df_users_hub.index.map(dfs["credits"].set_index("user_id").to_dict()["credits"])

    df_users_hub["buyer"] = df_users_hub["email"].isin(df_hotmart["email"])
    df_users_hub["buyer_date"] = df_users_hub["email"].map(df_hotmart.set_index("email").to_dict()["date"])
    df_users_hub["tracking"] = df_users_hub["email"].map(df_hotmart.set_index("email").to_dict()["tracking"])
    df_users_hub["total_activities"] = dfs["activities"].groupby("user_id")[["id"]].count().to_dict()["id"]

    df_users_hub.loc[:, "last_activity"] = pd.to_datetime(df_users_hub["id"].map(dfs["activities"].groupby("user_id")[["updated_at"]].last().to_dict()["updated_at"]))
    df_users_hub["days_to_buy"] = (df_users_hub["buyer_date"] - df_users_hub["user_registered"]).apply(lambda x: x.days).fillna(0)
    df_users_hub["days_registered"] = datetime.now().date() - df_users_hub["user_registered"].apply(lambda x: x.date())

    df_users_hub.loc[:, "total_days"] = (datetime.now(tz=timezone.utc) - df_users_hub["user_registered"]).apply(lambda x: x.days)
    df_users_hub["idle_days"] = datetime.now(tz=timezone.utc) - df_users_hub["last_activity"]#.dt.tz_localize('America/Sao_Paulo')
    df_users_hub["idle_days"] = df_users_hub["idle_days"].apply(lambda x: x.days)
    df_users_hub["used_days"] = (df_users_hub["total_days"] - df_users_hub["idle_days"]).fillna(0)

    df_users_hub["weeks_to_buy"] = df_users_hub["days_to_buy"].apply(lambda x: int(x / 7)).apply(lambda x: x if x < 20 else 20)
    df_users_hub["total_weeks"] = df_users_hub["total_days"].apply(lambda x: int(x / 7)).apply(lambda x: x if x < 20 else 20)
    df_users_hub["buyer"] = df_users_hub["email"].isin(df_hotmart["email"]).apply(lambda x: 1 if x else 0)

    df_users_hub = df_users_hub.sort_values(by="buyer_date")
    df_subs = df_users_hub[df_users_hub["role"] == "subscriber"]
    df_pro = df_users_hub[df_users_hub["role"] == "pro"]
    return df_users_hub, df_subs, df_pro

def get_course_data(df_act) -> dict:
    course_ids = dfs["posts"].query("post_type=='curso'")["ID"].values
    # pdb.set_trace()
    dict_course = {}
    df_act_count = df_act.groupby("post_id").count()["grade"]
    dict_posts_name = dfs["posts"].set_index("ID").to_dict()["post_title"]

    for cid in course_ids:
        dfs["postmeta"].query(f"post_id=={cid}")
        dict_course[cid] = {}
        for row in dfs["postmeta"].query(f"post_id=={cid}").values:
            if "modules" in row[2] and "title" in row[2] and row[2][0] != "_":
                dict_course[cid][int(row[2].split("_")[1])] = {}
                dict_course[cid][int(row[2].split("_")[1])]["name"] = row[3]
            
            if "modules" in row[2] and "activities" in row[2] and row[2][0] != "_":
                result = re.findall(r'"[^"]*"',row[3])
                
                results = [int(s[1:-1]) for s in result]
                dict_course[cid][int(row[2].split("_")[1])]["activities"] = {}
                
                for i, result in enumerate(results):
                    if result in df_act_count.index:
                        dict_act_temp = {
                            "name": dict_posts_name[result], 
                            "count": df_act_count[result], 
                            "id": result}
                        dict_course[cid][int(row[2].split("_")[1])]["activities"][i] = dict_act_temp

    # Obtem a contagem de atividades por modulo
    for cid in dict_course.keys():
        for act in dict_course[cid].keys():
            activities_sum = sum(item['count'] for item in dict_course[cid][act]["activities"].values())
            dict_course[cid][act]["count"] = activities_sum
    return dict_course

@st.cache_data    
def get_courses_by_tracks(dfs: dict)-> dict:
    """
    Dado o dicionário dfs contendo as tabelas do wp como Dataframe, retorna um dicionário contendo
    a qual trilha cada curso pertence.
    """
    tracks_ids = dfs['term_taxonomy'].loc[dfs["term_taxonomy"]['taxonomy'] == 'track', 'term_taxonomy_id']
    tracks_courses = {}
    a = {}

    for track_id in tracks_ids.unique():
        tracks_courses[str(track_id)] = dfs["term_relationships"].loc[dfs['term_relationships']['term_taxonomy_id'] == track_id, 'object_id'].to_frame().merge(dfs['posts'][['ID', 'post_title']], left_on='object_id', right_on='ID', how='inner')['post_title'].values
    
    all_courses = tracks_courses.values()
    all_courses_flatten = [item for sublist in all_courses for item in sublist]
    tmp_counter = pd.Series(all_courses_flatten).value_counts().to_frame()
    common_values = tmp_counter.drop(tmp_counter.loc[tmp_counter['count'] < 2].index).index.to_numpy()
    for key in tracks_courses.keys():
        tracks_courses[key] = tracks_courses[key][~np.isin(tracks_courses[key], common_values)]
    
    tracks_courses['Multiplas'] = common_values
    return tracks_courses

def get_color_codes(df, track_courses_dict):
    trilhas = np.full(shape=df.shape[0], fill_value='Sem trilha')
    
    for key in track_courses_dict.keys():
        trilhas[np.isin(df['curso'], track_courses_dict[key])] = str(key)
    return trilhas

st.title('Dashboard - Ensino')
# Data Load
dfs = run_query(tables)
dfs["users"]["user_registered"] = pd.to_datetime(dfs["users"]["user_registered"]).dt.tz_convert('America/Sao_Paulo')
dfs["users"].set_index("ID", inplace=True)
del dfs["term_relationships"]["term_order"], dfs["term_taxonomy"]["term_id"], dfs["term_taxonomy"]["description"]

end_date = datetime.today()
start_date = '2023-09-01'

try:
    df_hotmart = st.session_state['hotmart']
except KeyError:
    st.session_state['hotmart'] = get_hotmart_sales_hist(start_date=start_date, end_date=end_date)
    df_hotmart = st.session_state['hotmart']

df_users_hub = get_hub_users()
process_activities()
df_users_hub, df_subs, df_pro = process_hub_users(df_users_hub)

#pdb.set_trace()

# ============================
# Controladores
# ============================
all_courses = dfs["activities"]["curso"].unique()
date_range = st.sidebar.date_input("Datas", value=(datetime.today()-timedelta(days=15), datetime.today()))
courses = st.sidebar.multiselect("Cursos a excluir", all_courses, ["Python Starter"])
role = st.sidebar.selectbox("Role", ["subscriber", "pro"])


cids = sorted(list(st.session_state["course_ids"].keys()))
course = st.sidebar.selectbox("Curso", cids, cids.index("Python Starter"))
course_id = st.session_state["course_ids"][course] # Dict nome_curso : id
courses_by_track_dict = get_courses_by_tracks(dfs) 

# Filtro de data e Roles
df_act_ = dfs["activities"][(dfs["activities"]["created_at"].dt.date >= date_range[0]) & (dfs["activities"]["created_at"].dt.date <= date_range[-1])]
df_users_filt = df_users_hub[df_users_hub["role"] == role]
df_users_filt_register = df_users_filt[(df_users_filt["user_registered"].dt.date >= date_range[0]) & (df_users_filt["user_registered"].dt.date <= date_range[-1])]
user_ids = df_users_filt["id"].unique()
df_act_ = df_act_[df_act_["user_id"].isin(values=user_ids)]
df_act_['Trilha'] = get_color_codes(df=df_act_, track_courses_dict=courses_by_track_dict) #Adicionando a coluna das trilhas

dict_course = get_course_data(df_act=df_act_)
df_act_cut = df_act_[~df_act_["curso"].isin(values=courses)]





# pdb.set_trace()
# explorar dict_course

# ============================
# Data Display
# ============================
total_users_all_time = len(df_users_filt)
total_users_date_range = len(df_users_filt_register)

col1, col2, col3 = st.columns(3)
col1.metric(label=f"Total: {role.capitalize()}", value=f"{total_users_all_time}")
col2.metric(label=f"Total registrado no periodo: {role.capitalize()}", value=f"{total_users_date_range}")

# Cursos mais assistidos
with st.expander("Cursos e Trilhas assistidas"):
    col1, col2 = st.columns(2)
    activities_done_df = df_act_cut["curso"].value_counts().to_frame().merge(df_act_cut[['curso', 'Trilha']], on='curso', how='inner')
    activities_done_df.drop_duplicates(inplace=True)
    labels={'144':'Python Office', '145':'Análise e Visualização de Dados', '146':'Visão Computacional com Python',
             '147':'Data Science &amp; Machine Learning','148':'Trading Quantitativo', '149':'Dashboards Interativos com Python',
              'Multiplas': 'Múltiplas', 'sem trilha': 'Sem trilha'}
    activities_done_df['Trilha'] = activities_done_df['Trilha'].map(labels)
    color_map = {label: px.colors.qualitative.Plotly[i] for i, label in enumerate(labels.values())}


    fig = px.bar(activities_done_df, x=activities_done_df['count'], y='curso',color='Trilha',color_discrete_map=color_map)
    fig.update_layout(height=900, showlegend=False).update_yaxes(categoryorder="total ascending")
    col1.plotly_chart(fig, use_container_width=True)

    # Trilhas mais assistidas
    for course_, total in df_act_cut["curso"].value_counts().items():
        course_ = course_
        dfs["term_relationships"].loc[dfs["term_relationships"]["post_name"] == course_, "total_done"] = total
    df_terms_ = dfs["term_relationships"][~dfs["term_relationships"]["post_name"].isin([i for i in courses])]
    df_tracks = (df_terms_[df_terms_["taxonomy"] == 'track'].groupby("taxonomy_name")["total_done"].sum())
    df_tracks = df_tracks.to_frame().reset_index()
    df_tracks.columns = ['Trilhas', 'total_done']
    fig2 = px.bar(df_tracks, x='total_done', y='Trilhas', color='Trilhas', color_discrete_map=color_map).update_yaxes(categoryorder="total ascending")
    fig2.update_layout(height=900, showlegend=False)
    col2.plotly_chart(fig2, use_container_width=True)


# Análise de atividades
list_modules = []
for module in range(max(dict_course[course_id].keys())+1):
    dados = [(j["name"], j["count"], module) for i, j in dict_course[course_id][module]["activities"].items()]
    list_modules += dados
df = pd.DataFrame(list_modules, columns=['name', 'count', 'module'])

# pdb.set_trace()
st.divider()
total_users = len(set(df_act_[df_act_["curso"] == course]["user_id"].unique()).intersection(set(user_ids)))
st.metric(label=f"Usuários fazendo o curso no período", value=f"{total_users}")
# df["count"] =  df["count"] /  total_users


tmp_df = df_act_.loc[(df_act_["curso"] == course) & (df_act_['user_id'].isin(user_ids))]
tmp_df = tmp_df[['user_id', 'post_id']].groupby(by='user_id').count()
tmp_df.columns = ['Total_activities']
tmp_df = tmp_df.merge(df_users_filt_register[['id', 'user_registered']], left_index=True, right_on='id', how='inner') # inner para limitar os usuarios do periodo selecionado
tmp_df['days_since_reg'] = (datetime.now().astimezone() - tmp_df['user_registered']).dt.days
hist_fig = px.histogram(data_frame=tmp_df[['days_since_reg', 'Total_activities']], 
                        y='days_since_reg', x='Total_activities', histfunc='avg', text_auto='d', marginal='box',
                        title='Histograma dos módulos completos por tempo de registro', 
                        labels={'Total_activities':'Nº de atividades completas do módulo'}).update_layout(yaxis_title='Nº médio de dias registrado', bargap=0.15)
hist_fig.update_yaxes(showgrid=False)
st.plotly_chart(hist_fig, use_container_width=True)


df["name"] = df.index.astype(str) + " - " + df["name"]
df.set_index(['name'], inplace=True)
fig4 = px.bar(df, color='module')
fig4.update_layout(height=800, showlegend=False)
st.plotly_chart(fig4, use_container_width=True)
# df.iloc[::-1].plot(kind="barh")


# Análise de módulos
dados = [(f"{key} - " + val['name'], val['count'], len(val["activities"])) for key, val in dict_course[course_id].items()]
# len(dict_course[course_id][0]["activities"])
df = pd.DataFrame(dados, columns=['name', 'count', 'module_size'])
df["potential_size"] = df["module_size"] * total_users
# pdb.set_trace()
df["count"] = df["count"] / df["potential_size"]
df.set_index(['name'], inplace=True)
fig3 = px.bar(df, y="count")
fig3.update_layout(height=600, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)
