import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------------------------
# テキスト辞書（日本語と英語）
# ------------------------------
T = {
    "page_title": {"ja": "モビリティデータ分析デモ", "en": "Mobility Data Analysis Demo"},
    "sidebar_title": {"ja": "株式会社 Data Hiker", "en": "Data Hiker Inc."},
    "official_site": {"ja": "[公式サイトはこちら](https://www.data-hiker.com)", "en": "[Official Website](https://www.data-hiker.com)"},
    "consultation": {"ja": "### 分析のご相談", "en": "### Analysis Consultation"},
    "email": {"ja": "**Email:** datahiker.info@gmail.com", "en": "**Email:** datahiker.info@gmail.com"},
    "settings": {"ja": "### 設定", "en": "### Settings"},
    "dbscan_slider_label": {"ja": "DBSCAN min_samples", "en": "DBSCAN min_samples"},
    "dbscan_slider_help": {"ja": "各点の近くにどれくらい点があるかの目安です", "en": "Indicative number of points near each sample point"},
    "intro_paragraph": {
        "ja": """このデモは、ニューヨークのタクシー（黄色タクシーおよび緑色タクシー）の乗車記録データを用いた分析デモです。  
データは、[ニューヨークタクシー＆リムジン委員会（TLC）のウェブサイト](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)からダウンロードされています。
""",
        "en": """This demo is an analysis demonstration using trip record data of New York taxis (yellow and green taxis).  
The data was downloaded from the [New York Taxi & Limousine Commission (TLC) website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
"""
    },
    "section1_title": {"ja": "1. 乗車・降車地点マップ", "en": "1. Pickup and Dropoff Map"},
    "section1_insight": {
        "ja": "【インサイト】この地図を確認することで、ニューヨークでタクシーに乗る場所と降りる場所の全体的な分布が一目で分かります。",
        "en": "Insight: This map provides an overview of the distribution of taxi pickup and dropoff locations in New York."
    },
    "combined_map_title": {
        "ja": "乗車・降車地点の分布",
        "en": "Distribution of Pickup and Dropoff Locations"
    },
    "section2_title": {"ja": "2. 時間帯別アニメーション", "en": "2. Time-based Animation"},
    "section2_insight": {
        "ja": "【インサイト】このアニメーションでは、時間の経過に伴うタクシーの乗降地点の変化を確認できます。",
        "en": "Insight: This animation shows how taxi pickup and dropoff locations change over time."
    },
    "section3_title": {"ja": "3. DBSCANクラスタリング（乗車地点）", "en": "3. DBSCAN Clustering (Pickup Locations)"},
    "section3_insight": {
        "ja": "【インサイト】クラスタリングにより、類似した乗車地点がグループ化され、主要な乗車エリアが視覚的に把握できます。",
        "en": "Insight: Clustering groups similar pickup locations, visually highlighting major pickup areas."
    },
    "section3_stats_title": {"ja": "#### 乗車地点クラスタ別 距離統計", "en": "#### Pickup Location Cluster Distance Statistics"},
    "pickup_clustering_title": {"ja": "乗車地点のクラスタリング", "en": "Pickup Location Clustering"},
    "pickup_bubble_title": {"ja": "乗車地点のクラスタごとの移動距離の特徴", "en": "Travel Distance Characteristics by Pickup Cluster"},
    "section3_bubble_insight": {
        "ja": "【インサイト】バブルチャートでは、横軸が平均距離、縦軸が中央値距離を示し、バブルの大きさは件数、色は距離のばらつきを表しています。",
        "en": "Insight: In the bubble chart, the horizontal axis represents the average distance, the vertical axis the median distance, bubble size indicates count, and color shows distance variability."
    },
    "section4_title": {"ja": "4. DBSCANクラスタリング（降車地点）", "en": "4. DBSCAN Clustering (Dropoff Locations)"},
    "section4_insight": {
        "ja": "【インサイト】降車地点のクラスタリングにより、タクシーの降車が多い主要エリアが把握できます。",
        "en": "Insight: Clustering dropoff locations highlights major areas where taxis are dropped off."
    },
    "section4_stats_title": {"ja": "#### 降車地点クラスタ別 距離統計", "en": "#### Dropoff Location Cluster Distance Statistics"},
    "dropoff_clustering_title": {"ja": "降車地点のクラスタリング", "en": "Dropoff Location Clustering"},
    "dropoff_bubble_title": {"ja": "降車地点のクラスタごとの移動距離の特徴", "en": "Travel Distance Characteristics by Dropoff Cluster"},
    "section4_bubble_insight": {
        "ja": "【インサイト】バブルチャートでは、横軸が平均距離、縦軸が中央値距離を示し、バブルの大きさは件数、色は距離のばらつきを表しています。",
        "en": "Insight: In the bubble chart, the horizontal axis represents the average distance, the vertical axis the median distance, bubble size indicates count, and color shows distance variability."
    },
    "section5_title": {"ja": "5. ピックアップから降車への移動フロー（Sankey Diagram）", "en": "5. Flow from Pickup to Dropoff (Sankey Diagram)"},
    "section5_insight": {
        "ja": "【インサイト】このフロー図は、どの乗車クラスタからどの降車クラスタへタクシーが移動しているかを直感的に示します。",
        "en": "Insight: This flow diagram intuitively shows which pickup clusters lead to which dropoff clusters."
    },
    "sankey_title": {"ja": "乗車クラスタから降車クラスタへの移動フロー", "en": "Flow from Pickup to Dropoff Clusters"},
    "pickup_cluster_prefix": {"ja": "乗車クラスタ ", "en": "Pickup Cluster "},
    "dropoff_cluster_prefix": {"ja": "降車クラスタ ", "en": "Dropoff Cluster "}
}

# ------------------------------
# UI設定
# ------------------------------
st.set_page_config(
    page_title="モビリティデータ分析デモ / Mobility Data Analysis Demo", 
    layout="wide", 
    page_icon="🚕",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    # 言語選択
    lang = st.radio("言語 / Language", options=["日本語", "English"])
    lang_code = "ja" if lang == "日本語" else "en"
    
    st.markdown("### " + T["sidebar_title"][lang_code])
    st.markdown(T["official_site"][lang_code])
    st.markdown("---")
    st.markdown(T["consultation"][lang_code])
    st.markdown(T["email"][lang_code])
    st.markdown("---")
    st.markdown(T["settings"][lang_code])
    min_samples = st.slider(
        T["dbscan_slider_label"][lang_code],
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help=T["dbscan_slider_help"][lang_code]
    )

st.title(T["page_title"][lang_code])
st.markdown(T["intro_paragraph"][lang_code])

# ------------------------------
# データ読み込みと前処理
# ------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("data/newyork_dataset.csv")
    required_cols = [
        'pickup_latitude', 'pickup_longitude', 
        'dropoff_latitude', 'dropoff_longitude', 
        'tpep_pickup_datetime'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"CSVに必要なカラムがありません: {missing}")
        st.stop()
    df = df[required_cols].dropna()
    df.rename(columns={
        'pickup_latitude': 'pickup_lat',
        'pickup_longitude': 'pickup_lng',
        'dropoff_latitude': 'dropoff_lat',
        'dropoff_longitude': 'dropoff_lng',
        'tpep_pickup_datetime': 'pickup_time'
    }, inplace=True)
    
    # pickup_time をUTCとして読み込み（元データはUTCの5時～20時）
    df['pickup_time'] = pd.to_datetime(df['pickup_time'], utc=True)
    # UTC時刻からhourを取得
    df['hour'] = df['pickup_time'].dt.hour
    return df

df = load_data()

# ------------------------------
# 乗車・降車間の距離計算
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

df['trip_distance_km'] = haversine(df['pickup_lat'], df['pickup_lng'], 
                                   df['dropoff_lat'], df['dropoff_lng'])

# ------------------------------
# 1. 乗車・降車地点マップ
# ------------------------------
st.markdown(T["section1_title"][lang_code])
st.markdown(T["section1_insight"][lang_code])
def render_combined_map(df):
    pickup = df[['pickup_lat', 'pickup_lng']].rename(columns={'pickup_lat': 'lat', 'pickup_lng': 'lng'})
    dropoff = df[['dropoff_lat', 'dropoff_lng']].rename(columns={'dropoff_lat': 'lat', 'dropoff_lng': 'lng'})
    center_lat = pd.concat([pickup['lat'], dropoff['lat']]).mean()
    center_lng = pd.concat([pickup['lng'], dropoff['lng']]).mean()
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=pickup['lat'], lon=pickup['lng'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=6, color='blue', opacity=0.4),
        name='Pickup'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=dropoff['lat'], lon=dropoff['lng'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=6, color='orange', opacity=0.4),
        name='Dropoff'
    ))
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=go.layout.mapbox.Center(lat=center_lat, lon=center_lng),
            zoom=11
        ),
        margin={"r":0, "t":40, "l":0, "b":0},
        title=T["combined_map_title"][lang_code]
    )
    st.plotly_chart(fig, use_container_width=True)

render_combined_map(df)

# ------------------------------
# 2. 時間帯別アニメーション
# ------------------------------
st.markdown(T["section2_title"][lang_code])
st.markdown(T["section2_insight"][lang_code])
df_pick = df[['pickup_lat', 'pickup_lng', 'hour']].copy()
df_pick['type'] = 'Pickup'
df_pick = df_pick.rename(columns={'pickup_lat': 'lat', 'pickup_lng': 'lng'})
df_drop = df[['dropoff_lat', 'dropoff_lng', 'hour']].copy()
df_drop['type'] = 'Dropoff'
df_drop = df_drop.rename(columns={'dropoff_lat': 'lat', 'dropoff_lng': 'lng'})
full_anim = pd.concat([df_pick, df_drop])
fig_anim = px.scatter_mapbox(
    full_anim,
    lat="lat", lon="lng",
    color="type",
    animation_frame="hour",
    category_orders={"hour": list(range(24))},
    center=dict(lat=full_anim['lat'].mean(), lon=full_anim['lng'].mean()),
    zoom=10, height=700,
    mapbox_style="carto-darkmatter",
    opacity=0.5,
    color_discrete_map={"Pickup": "blue", "Dropoff": "orange"}
)
st.plotly_chart(fig_anim, use_container_width=True)

# ------------------------------
# 3. DBSCANクラスタリング（乗車地点）と統計・バブルチャート
# ------------------------------
st.markdown(T["section3_title"][lang_code])
st.markdown(T["section3_insight"][lang_code])
pickup_coords = df[['pickup_lat', 'pickup_lng']].values
pickup_scaled = StandardScaler().fit_transform(pickup_coords)
@st.cache_data(show_spinner=True)
def compute_dbscan(data, min_samples):
    eps_fixed = 0.15
    db = DBSCAN(eps=eps_fixed, min_samples=min_samples).fit(data)
    return db.labels_
df['pickup_cluster'] = compute_dbscan(pickup_scaled, min_samples)
pickup_clusters = df[df['pickup_cluster'] != -1]

fig_pickup = go.Figure()
for cid in sorted(pickup_clusters['pickup_cluster'].unique()):
    cdata = pickup_clusters[pickup_clusters['pickup_cluster'] == cid]
    fig_pickup.add_trace(go.Scattermapbox(
        lat=cdata['pickup_lat'],
        lon=cdata['pickup_lng'],
        mode="markers",
        marker=dict(size=6, opacity=0.5),
        name=f"Cluster {cid}"
    ))
fig_pickup.update_layout(
    mapbox=dict(
        style="carto-darkmatter",
        center=dict(lat=pickup_clusters['pickup_lat'].mean(), lon=pickup_clusters['pickup_lng'].mean()),
        zoom=10
    ),
    margin={"r":0, "t":40, "l":0, "b":0},
    title=T["pickup_clustering_title"][lang_code]
)
st.plotly_chart(fig_pickup, use_container_width=True)

# 集計は内部的には英語表記のカラム名で行い、その後表示用にリネーム
pickup_cluster_stats = df[df['pickup_cluster'] != -1].groupby("pickup_cluster")["trip_distance_km"].agg(
    mean="mean", median="median", std="std", count="count"
).reset_index()
if lang_code == "ja":
    pickup_cluster_stats.rename(columns={"pickup_cluster": "クラスタ", "mean": "平均", "median": "中央値", "std": "標準偏差", "count": "件数"}, inplace=True)
    hover_col_pickup = "クラスタ"
    labels_pickup = {
        "平均": "平均距離 (km)",
        "中央値": "中央値距離 (km)",
        "標準偏差": "距離のばらつき",
        "件数": "件数",
        "クラスタ": "クラスタ"
    }
else:
    pickup_cluster_stats.rename(columns={"pickup_cluster": "Cluster", "mean": "Average", "median": "Median", "std": "Std Dev", "count": "Count"}, inplace=True)
    hover_col_pickup = "Cluster"
    labels_pickup = {
        "Average": "Average Distance (km)",
        "Median": "Median Distance (km)",
        "Std Dev": "Distance Variability",
        "Count": "Count",
        "Cluster": "Cluster"
    }
st.markdown(T["section3_stats_title"][lang_code])
st.dataframe(pickup_cluster_stats.style.set_properties(**{'font-size': '14px'}))

fig_bubble_pickup = px.scatter(
    pickup_cluster_stats,
    x=("平均" if lang_code=="ja" else "Average"), 
    y=("中央値" if lang_code=="ja" else "Median"),
    size=("件数" if lang_code=="ja" else "Count"),
    color=("標準偏差" if lang_code=="ja" else "Std Dev"),
    hover_name=hover_col_pickup,
    title=T["pickup_bubble_title"][lang_code],
    labels=labels_pickup
)
st.plotly_chart(fig_bubble_pickup, use_container_width=True)
st.markdown(T["section3_bubble_insight"][lang_code])

# ------------------------------
# 4. DBSCANクラスタリング（降車地点）と統計・バブルチャート
# ------------------------------
st.markdown(T["section4_title"][lang_code])
st.markdown(T["section4_insight"][lang_code])
dropoff_coords = df[['dropoff_lat', 'dropoff_lng']].values
dropoff_scaled = StandardScaler().fit_transform(dropoff_coords)
df['dropoff_cluster'] = compute_dbscan(dropoff_scaled, min_samples)
dropoff_clusters = df[df['dropoff_cluster'] != -1]

fig_dropoff = go.Figure()
for cid in sorted(dropoff_clusters['dropoff_cluster'].unique()):
    cdata = dropoff_clusters[dropoff_clusters['dropoff_cluster'] == cid]
    fig_dropoff.add_trace(go.Scattermapbox(
        lat=cdata['dropoff_lat'],
        lon=cdata['dropoff_lng'],
        mode="markers",
        marker=dict(size=6, opacity=0.5),
        name=f"Cluster {cid}"
    ))
fig_dropoff.update_layout(
    mapbox=dict(
        style="carto-darkmatter",
        center=dict(lat=dropoff_clusters['dropoff_lat'].mean(), lon=dropoff_clusters['dropoff_lng'].mean()),
        zoom=10
    ),
    margin={"r":0, "t":40, "l":0, "b":0},
    title=T["dropoff_clustering_title"][lang_code]
)
st.plotly_chart(fig_dropoff, use_container_width=True)

dropoff_cluster_stats = df[df['dropoff_cluster'] != -1].groupby("dropoff_cluster")["trip_distance_km"].agg(
    mean="mean", median="median", std="std", count="count"
).reset_index()
if lang_code == "ja":
    dropoff_cluster_stats.rename(columns={"dropoff_cluster": "クラスタ", "mean": "平均", "median": "中央値", "std": "標準偏差", "count": "件数"}, inplace=True)
    hover_col_dropoff = "クラスタ"
    labels_dropoff = {
        "平均": "平均距離 (km)",
        "中央値": "中央値距離 (km)",
        "標準偏差": "距離のばらつき",
        "件数": "件数",
        "クラスタ": "クラスタ"
    }
else:
    dropoff_cluster_stats.rename(columns={"dropoff_cluster": "Cluster", "mean": "Average", "median": "Median", "std": "Std Dev", "count": "Count"}, inplace=True)
    hover_col_dropoff = "Cluster"
    labels_dropoff = {
        "Average": "Average Distance (km)",
        "Median": "Median Distance (km)",
        "Std Dev": "Distance Variability",
        "Count": "Count",
        "Cluster": "Cluster"
    }
st.markdown(T["section4_stats_title"][lang_code])
st.dataframe(dropoff_cluster_stats.style.set_properties(**{'font-size': '14px'}))

fig_bubble_dropoff = px.scatter(
    dropoff_cluster_stats,
    x=("平均" if lang_code=="ja" else "Average"), 
    y=("中央値" if lang_code=="ja" else "Median"),
    size=("件数" if lang_code=="ja" else "Count"),
    color=("標準偏差" if lang_code=="ja" else "Std Dev"),
    hover_name=hover_col_dropoff,
    title=T["dropoff_bubble_title"][lang_code],
    labels=labels_dropoff
)
st.plotly_chart(fig_bubble_dropoff, use_container_width=True)
st.markdown(T["section4_bubble_insight"][lang_code])

# ------------------------------
# 5. ピックアップから降車への移動フロー（Sankey Diagram）
# ------------------------------
st.markdown(T["section5_title"][lang_code])
st.markdown(T["section5_insight"][lang_code])
df_flow = df[(df['pickup_cluster'] != -1) & (df['dropoff_cluster'] != -1)]
flow_data = df_flow.groupby(['pickup_cluster', 'dropoff_cluster']).size().reset_index(name='count')

# ノードラベルは各言語に合わせて設定
pickup_labels = [T["pickup_cluster_prefix"][lang_code] + str(x) for x in sorted(df_flow['pickup_cluster'].unique())]
dropoff_labels = [T["dropoff_cluster_prefix"][lang_code] + str(x) for x in sorted(df_flow['dropoff_cluster'].unique())]
labels = pickup_labels + dropoff_labels

source = flow_data['pickup_cluster'].apply(lambda x: pickup_labels.index(T["pickup_cluster_prefix"][lang_code] + str(x))).tolist()
target = flow_data['dropoff_cluster'].apply(lambda x: dropoff_labels.index(T["dropoff_cluster_prefix"][lang_code] + str(x)) + len(pickup_labels)).tolist()
values = flow_data['count'].tolist()

sankey_data = dict(
    type='sankey',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)
fig_sankey = go.Figure(data=[sankey_data])
fig_sankey.update_layout(title_text=T["sankey_title"][lang_code], font_size=10)
st.plotly_chart(fig_sankey, use_container_width=True)
