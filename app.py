import streamlit as st

from src.core.io import load_csv
from src.tabs.overview import render as render_overview
from src.tabs.quality import render as render_quality
from src.tabs.relationships import render as render_relationships
from src.tabs.sample import render as render_sample
from src.tabs.variables import render as render_variables

st.set_page_config(page_title="QuickEDA", layout="wide")
st.title("📊 QuickEDA（思考プロセス支援EDA）")

# --- Sidebar: 読み込み設定 ---
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

# --- Load ---
try:
    df = load_csv(
        uploaded_file=uploaded_file,
        sep=sep,
        encoding=encoding,
        skiprows=skiprows,
        na_values_text=na_values_text,
    )
except Exception as e:
    st.error(
        "CSVの読み込みに失敗しました。設定（区切り文字・文字コード等）を見直してください。"
    )
    st.exception(e)
    st.stop()

st.success(f"読み込み完了: {df.shape[0]:,} 行 × {df.shape[1]:,} 列")

# --- Tabs (EDAの型に沿う) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "① Overview",
        "② Variables",
        "③ Data Quality",
        "④ Relationships",
        "⑤ Sample/Inspect",
    ]
)

with tab1:
    render_overview(df)

with tab2:
    render_variables(df)

with tab3:
    render_quality(df)

with tab4:
    render_relationships(df)

with tab5:
    render_sample(df)
