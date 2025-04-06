import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------------------------
# UI設定
# ------------------------------
st.set_page_config(
    page_title="モビリティデータ分析デモ", 
    layout="wide", 
    page_icon="🚕",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("### 株式会社 Data Hiker")
    st.markdown("[公式サイトはこちら](https://www.data-hiker.com)")
    st.markdown("---")
    st.markdown("### 分析のご相談")
    st.markdown("**Email:** datahiker.info@gmail.com")
    st.markdown("---")
    st.markdown("### 設定")
    min_samples = st.slider(
        "DBSCAN min_samples", 
        min_value=5, 
        max_value=100, 
        value=30, 
        step=5, 
        help="各点の近くにどれくらい点があるかの目安です"
    )

st.title("モビリティデータ分析デモ")
st.markdown("""
このデモは、ニューヨークのタクシー（黄色タクシーおよび緑色タクシー）の乗車記録データを用いた分析デモです。  
データは、[ニューヨークタクシー＆リムジン委員会（TLC）のウェブサイト](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)からダウンロードされています。
""")

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
st.markdown("### 1. 乗車・降車地点マップ")
st.markdown("【インサイト】この地図を確認することで、ニューヨークでタクシーに乗る場所と降りる場所の全体的な分布が一目で分かります。")
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
        name='乗車'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=dropoff['lat'], lon=dropoff['lng'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=6, color='orange', opacity=0.4),
        name='降車'
    ))
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=go.layout.mapbox.Center(lat=center_lat, lon=center_lng),
            zoom=11
        ),
        margin={"r":0, "t":40, "l":0, "b":0},
        title="乗車・降車地点の分布"
    )
    st.plotly_chart(fig, use_container_width=True)

render_combined_map(df)

# ------------------------------
# 2. 時間帯別アニメーション
# ------------------------------
st.markdown("### 2. 時間帯別アニメーション")
st.markdown("【インサイト】このアニメーションでは、時間の経過に伴うタクシーの乗降地点の変化を確認できます。")
df_pick = df[['pickup_lat', 'pickup_lng', 'hour']].copy()
df_pick['type'] = '乗車'
df_pick = df_pick.rename(columns={'pickup_lat': 'lat', 'pickup_lng': 'lng'})
df_drop = df[['dropoff_lat', 'dropoff_lng', 'hour']].copy()
df_drop['type'] = '降車'
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
    color_discrete_map={"乗車": "blue", "降車": "orange"}
)
st.plotly_chart(fig_anim, use_container_width=True)

# ------------------------------
# 3. DBSCANクラスタリング（乗車地点）とクラスタ別統計・バブルチャート
# ------------------------------
st.markdown("### 3. DBSCANクラスタリング（乗車地点）")
st.markdown("【インサイト】クラスタリングにより、類似した乗車地点がグループ化され、主要な乗車エリアが視覚的に把握できます。")
pickup_coords = df[['pickup_lat', 'pickup_lng']].values
pickup_scaled = StandardScaler().fit_transform(pickup_coords)
@st.cache_data(show_spinner=True)
def compute_dbscan(data, min_samples):
    eps_fixed = 0.15
    db = DBSCAN(eps=eps_fixed, min_samples=min_samples).fit(data)
    return db.labels_
df['pickup_cluster'] = compute_dbscan(pickup_scaled, min_samples)
clusters_pickup = len(set(df['pickup_cluster'])) - (1 if -1 in df['pickup_cluster'] else 0)
#st.markdown(f"乗車地点 クラスタ数: **{clusters_pickup}** (グループに入らなかった点は -1)")
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
    title="乗車地点のクラスタリング"
)
st.plotly_chart(fig_pickup, use_container_width=True)
pickup_cluster_stats = df[df['pickup_cluster'] != -1].groupby("pickup_cluster")["trip_distance_km"].agg(
    平均="mean", 中央値="median", 標準偏差="std", 件数="count"
).reset_index()
st.markdown("#### 乗車地点クラスタ別 距離統計")
st.dataframe(pickup_cluster_stats)
fig_bubble_pickup = px.scatter(
    pickup_cluster_stats,
    x="平均", y="中央値",
    size="件数",
    color="標準偏差",
    hover_name="pickup_cluster",
    title="乗車地点のクラスタごとの移動距離の特徴",
    labels={
        "平均": "平均距離 (km)",
        "中央値": "中央値距離 (km)",
        "標準偏差": "距離のばらつき",
        "件数": "件数",
        "pickup_cluster": "クラスタ"
    }
)
st.plotly_chart(fig_bubble_pickup, use_container_width=True)
st.markdown("【インサイト】この結果から、乗車地点の各クラスタの規模や移動距離のばらつきを比較できます。")

# ------------------------------
# 4. DBSCANクラスタリング（降車地点）とクラスタ別統計・バブルチャート
# ------------------------------
st.markdown("### 4. DBSCANクラスタリング（降車地点）")
st.markdown("【インサイト】降車地点のクラスタリングにより、タクシーの降車が多い主要エリアが把握できます。")
dropoff_coords = df[['dropoff_lat', 'dropoff_lng']].values
dropoff_scaled = StandardScaler().fit_transform(dropoff_coords)
df['dropoff_cluster'] = compute_dbscan(dropoff_scaled, min_samples)
clusters_dropoff = len(set(df['dropoff_cluster'])) - (1 if -1 in df['dropoff_cluster'] else 0)
#st.markdown(f"降車地点 クラスタ数: **{clusters_dropoff}** (グループに入らなかった点は -1)")
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
    title="降車地点のクラスタリング"
)
st.plotly_chart(fig_dropoff, use_container_width=True)
dropoff_cluster_stats = df[df['dropoff_cluster'] != -1].groupby("dropoff_cluster")["trip_distance_km"].agg(
    平均="mean", 中央値="median", 標準偏差="std", 件数="count"
).reset_index()
st.markdown("#### 降車地点クラスタ別 距離統計")
st.dataframe(dropoff_cluster_stats)
fig_bubble_dropoff = px.scatter(
    dropoff_cluster_stats,
    x="平均", y="中央値",
    size="件数",
    color="標準偏差",
    hover_name="dropoff_cluster",
    title="降車地点のクラスタごとの移動距離の特徴",
    labels={
        "平均": "平均距離 (km)",
        "中央値": "中央値距離 (km)",
        "標準偏差": "距離のばらつき",
        "件数": "件数",
        "dropoff_cluster": "クラスタ"
    }
)
st.plotly_chart(fig_bubble_dropoff, use_container_width=True)
st.markdown("【インサイト】この図から、降車地点の各クラスタの利用規模と移動距離の特徴を比較できます。")

# ------------------------------
# 5. ピックアップから降車への移動フロー（Sankey Diagram）
# ------------------------------
st.markdown("### 5. ピックアップから降車への移動フロー（Sankey Diagram）")
st.markdown("【インサイト】このフロー図は、どの乗車クラスタからどの降車クラスタへタクシーが移動しているかを直感的に示します。")

df_flow = df[(df['pickup_cluster'] != -1) & (df['dropoff_cluster'] != -1)]
flow_data = df_flow.groupby(['pickup_cluster', 'dropoff_cluster']).size().reset_index(name='count')

# クラスタの名前は、クラスタリング時の結果をもとに設定
pickup_labels = ["Pickup Cluster " + str(x) for x in sorted(df_flow['pickup_cluster'].unique())]
dropoff_labels = ["Dropoff Cluster " + str(x) for x in sorted(df_flow['dropoff_cluster'].unique())]
labels = pickup_labels + dropoff_labels

# ピックアップノードはpickup_labelsの順、降車ノードはdropoff_labelsの順にインデックスを割り当てる
source = flow_data['pickup_cluster'].apply(lambda x: pickup_labels.index("Pickup Cluster " + str(x))).tolist()
target = flow_data['dropoff_cluster'].apply(lambda x: dropoff_labels.index("Dropoff Cluster " + str(x)) + len(pickup_labels)).tolist()
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
fig_sankey.update_layout(title_text="乗車クラスタから降車クラスタへの移動フロー", font_size=10)
st.plotly_chart(fig_sankey, use_container_width=True)
