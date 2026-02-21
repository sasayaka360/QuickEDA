import streamlit as st

from src.eda import render_filter


def render(df):
    st.caption("目的：実データを確認し、抽出・ダウンロードで調査を進める")
    render_filter(df)
