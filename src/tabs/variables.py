import streamlit as st

from src.eda import render_profile_and_detail


def render(df):
    st.caption("目的：各変数の性質（分布・偏り・ユニーク・外れ値など）を理解する")
    render_profile_and_detail(df)
