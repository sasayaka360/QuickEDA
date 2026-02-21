import pandas as pd
import streamlit as st


def render(df: pd.DataFrame):
    st.caption("目的：変数間の関係（相関・強いペア）から仮説の種を見つける")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.info("数値列が2列以上ないため、相関は表示できません。")
        return

    st.subheader("相関行列（数値列）")
    corr = df[num_cols].corr(numeric_only=True)
    st.dataframe(corr, use_container_width=True)
