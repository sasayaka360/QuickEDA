import pandas as pd
import streamlit as st


def parse_na_values(text: str):
    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x != ""]


@st.cache_data(show_spinner=False)
def load_csv(
    uploaded_file,
    sep=",",
    encoding="utf-8-sig",
    skiprows=0,
    na_values_text="NA,N/A,null,NULL,",
):
    uploaded_file.seek(0)
    df = pd.read_csv(
        uploaded_file,
        sep=sep,
        encoding=encoding,
        skiprows=skiprows,
        na_values=parse_na_values(na_values_text),
    )
    return df
