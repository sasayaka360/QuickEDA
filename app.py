import streamlit as st

from src.eda import (
    load_csv,
    render_filter,
    render_overview,
    render_profile_and_detail,
    render_viz,
)

st.set_page_config(page_title="Pythonアプリ - EDA", layout="wide")
st.title("📊 Pythonアプリ（EDA拡張版）")

# Sidebar: 読み込み設定
st.sidebar.header("読み込み設定")
sep = st.sidebar.selectbox("区切り文字", [",", "\t", ";", "|"], index=0)
encoding = st.sidebar.selectbox(
    "文字コード", ["utf-8", "utf-8-sig", "cp932", "shift_jis"], index=1
)
skiprows = st.sidebar.number_input(
    "先頭の読み飛ばし行（skiprows）", min_value=0, max_value=1000, value=0, step=1
)
na_values_text = st.sidebar.text_input(
    "欠損として扱う文字（カンマ区切り）", value="NA,N/A,null,NULL,"
)

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is None:
    st.info("まずはCSVをアップロードしてください。")
    st.stop()

# 読み込み
try:
    df = load_csv(
        uploaded_file,
        sep=sep,
        encoding=encoding,
        skiprows=skiprows,
        na_values_text=na_values_text,
    )
except Exception as e:
    st.error(
        "CSVの読み込みに失敗しました。区切り文字・文字コードなどを見直してください。"
    )
    st.exception(e)
    st.stop()

# Tabs
tab_overview, tab_profile, tab_filter, tab_viz = st.tabs(
    ["概要", "列分析", "フィルタ/抽出", "可視化"]
)

with tab_overview:
    render_overview(df)

with tab_profile:
    render_profile_and_detail(df)

with tab_filter:
    render_filter(df)

with tab_viz:
    render_viz(df)
