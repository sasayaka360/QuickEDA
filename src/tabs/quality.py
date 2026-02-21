import pandas as pd
import streamlit as st


def render(df: pd.DataFrame):
    st.caption("目的：データの信頼性を確認し、問題（欠損・重複・怪しい列）を洗い出す")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("重複行", f"{int(df.duplicated().sum()):,}")
    with c2:
        st.metric("欠損がある列", f"{int((df.isnull().sum() > 0).sum()):,}")
    with c3:
        st.metric("欠損セル合計", f"{int(df.isnull().sum().sum()):,}")

    st.subheader("欠損率ランキング（上位20列）")
    miss = (df.isnull().mean().sort_values(ascending=False) * 100).head(20)
    if miss.max() == 0:
        st.success("欠損は見当たりません（少なくとも上位20列では0%）。")
    else:
        st.dataframe(miss.rename("missing_%").to_frame(), use_container_width=True)
        st.bar_chart(miss)

    st.subheader("簡易アラート（v0.1）")
    alerts = []

    # 定数列
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        alerts.append(f"定数列（値が1種類以下）: {len(const_cols)}列")

    # 高カーディナリティ（雑に）
    high_card = (nunique / len(df)).sort_values(ascending=False)
    high_card_cols = high_card[high_card > 0.5].index.tolist()  # 閾値は暫定
    if high_card_cols:
        alerts.append(f"高カーディナリティ（unique率>50%）: {len(high_card_cols)}列")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.info("簡易アラートはありません（v0.1の判定条件内）。")

    with st.expander("詳細（候補列一覧）", expanded=False):
        st.write("### 定数列")
        st.write(const_cols if const_cols else "なし")
        st.write("### 高カーディナリティ列（unique率>50%）")
        st.write(high_card_cols if high_card_cols else "なし")
