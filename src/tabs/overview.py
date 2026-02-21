import streamlit as st

from src.eda import render_overview


def render(df):
    st.caption("目的：データの全体像（規模・型・欠損・基本統計）を掴む")
    render_overview(df)
