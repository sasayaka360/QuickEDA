# src/eda.py
import io

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None


# -----------------------
# Altair helpers
# -----------------------
def a_field(name: str) -> str:
    """
    Altairは ':' を shorthand の区切りとして解釈することがあるため、
    列名に ':' が含まれる場合は '\:' にエスケープする。
    """
    return name.replace(":", r"\:")


# -----------------------
# I/O
# -----------------------
def parse_na_values(text: str):
    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x != ""]


def df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


# -----------------------
# Tables
# -----------------------
def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_rate = (missing / len(df) * 100).round(2)
    out = pd.DataFrame({"欠損数": missing, "欠損率(%)": missing_rate}).sort_values(
        "欠損数", ascending=False
    )
    return out


def profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        n_missing = int(s.isnull().sum())
        n_unique = int(s.nunique(dropna=True))

        top, top_freq = None, None
        try:
            vc = s.value_counts(dropna=True)
            if len(vc) > 0:
                top = str(vc.index[0])
                top_freq = int(vc.iloc[0])
        except Exception:
            pass

        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": n_missing,
            "missing_rate(%)": round(n_missing / n * 100, 2) if n else 0.0,
            "unique": n_unique,
            "top": top,
            "top_freq": top_freq,
        }

        if pd.api.types.is_numeric_dtype(s):
            row.update(
                {
                    "min": s.min(skipna=True),
                    "max": s.max(skipna=True),
                    "mean": s.mean(skipna=True),
                    "std": s.std(skipna=True),
                }
            )
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["missing", "unique"], ascending=[False, True])
        .reset_index(drop=True)
    )


# -----------------------
# Type helpers
# -----------------------
def try_parse_datetime(series: pd.Series, threshold: float = 0.6):
    """
    object/string列をdatetimeにできそうなら変換して返す。無理ならNone。
    混在タイムゾーン対策として utc=True でパースし、最後にtzを外して datetime64[ns] にする。
    """
    # すでにdatetime系ならそのまま（tz付きも含む）
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # pandasのStringDtype（"string"）も対象にする
    if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
        return None

    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        # parsed は datetime64[ns, UTC] になるので .dt が使える
        if parsed.notna().mean() >= threshold:
            # 表示・集計を扱いやすくするため tz を外す（datetime64[ns]）
            return parsed.dt.tz_convert(None)
    except Exception:
        return None

    return None


# -----------------------
# Rendering (Streamlit)
# -----------------------
def render_overview(df: pd.DataFrame):
    st.subheader("データプレビュー")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("基本サマリ")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("行数", f"{df.shape[0]:,}")
    with c2:
        st.metric("列数", f"{df.shape[1]:,}")
    with c3:
        st.metric("重複行", f"{int(df.duplicated().sum()):,}")
    with c4:
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("推定メモリ(MB)", f"{mem_mb:,.2f}")

    st.subheader("型情報（df.info）")
    st.code(df_info_text(df), language="text")

    st.subheader("欠損（列ごと）")
    st.dataframe(missing_table(df), use_container_width=True)

    st.subheader("基本統計量（describe）")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


def render_profile_and_detail(df: pd.DataFrame):
    st.subheader("列プロファイル（実務でまず見るやつ）")

    prof = profile_table(df)

    if "selected_col" not in st.session_state:
        st.session_state["selected_col"] = df.columns[0]

    selected_from_table = None
    try:
        event = st.dataframe(
            prof,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="profile_table",
        )
        if event and event.selection and event.selection.rows:
            ridx = event.selection.rows[0]
            selected_from_table = prof.loc[ridx, "column"]
    except Exception:
        st.dataframe(prof, use_container_width=True, hide_index=True)

    if selected_from_table is not None:
        st.session_state["selected_col"] = selected_from_table

    target_col = st.selectbox(
        "列を選択（表選択と連動）",
        options=list(df.columns),
        index=list(df.columns).index(st.session_state["selected_col"]),
        key="selected_col",
    )

    st.divider()
    render_column_detail(df, target_col)


def render_column_detail(df: pd.DataFrame, target_col: str):
    s = df[target_col]
    st.subheader(f"列ごとの詳細：{target_col}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("dtype:", s.dtype)
    with c2:
        st.write("欠損数:", int(s.isnull().sum()))
    with c3:
        st.write("ユニーク数:", int(s.nunique(dropna=True)))
    with c4:
        st.write("欠損率(%):", round(float(s.isnull().mean() * 100), 2))

    s_dt = try_parse_datetime(s)

    if alt is None:
        st.warning(
            "Altairが未導入のため、可視化は簡易表示になります。必要なら `pip install altair` を実行してください。"
        )

    # ---- 数値列
    if pd.api.types.is_numeric_dtype(s):
        st.markdown("### 分布（数値）")
        data_num = pd.DataFrame({target_col: s.dropna()})

        cc1, cc2 = st.columns(2)
        with cc1:
            if alt is not None and len(data_num) > 0:
                bins = st.slider(
                    "ヒストグラム bins", 5, 100, 30, key=f"bins_{target_col}"
                )
                f = a_field(target_col)
                hist = (
                    alt.Chart(data_num)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            f,
                            type="quantitative",
                            bin=alt.Bin(maxbins=bins),
                            title=target_col,
                        ),
                        # count():Q のような「:」付きshorthandを避ける
                        y=alt.Y("count()", type="quantitative", title="count"),
                        tooltip=[alt.Tooltip("count()", type="quantitative")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(hist, use_container_width=True)
            else:
                st.bar_chart(data_num[target_col].value_counts(bins=30).sort_index())

        with cc2:
            if alt is not None and len(data_num) > 0:
                f = a_field(target_col)
                box = (
                    alt.Chart(data_num)
                    .mark_boxplot()
                    .encode(y=alt.Y(f, type="quantitative", title=target_col))
                    .properties(height=260)
                )
                st.altair_chart(box, use_container_width=True)
            else:
                st.dataframe(data_num.describe(), use_container_width=True)

        st.markdown("### 上位/下位サンプル")
        k = st.slider("表示件数", 5, 50, 10, key=f"topk_{target_col}")
        st.write("小さい順")
        st.dataframe(
            df[[target_col]].sort_values(target_col).head(k), use_container_width=True
        )
        st.write("大きい順")
        st.dataframe(
            df[[target_col]].sort_values(target_col, ascending=False).head(k),
            use_container_width=True,
        )
        return

    # ---- datetimeっぽい列
    if s_dt is not None:
        st.markdown("### 日時の分布")
        dt_min, dt_max = s_dt.min(), s_dt.max()
        st.write("期間:", dt_min, "〜", dt_max)

        freq = st.selectbox(
            "集計粒度", ["D(日)", "W(週)", "M(月)"], index=2, key=f"dtfreq_{target_col}"
        )
        if freq.startswith("D"):
            grp = s_dt.dt.to_period("D").astype(str)
        elif freq.startswith("W"):
            grp = s_dt.dt.to_period("W").astype(str)
        else:
            grp = s_dt.dt.to_period("M").astype(str)

        vc = grp.value_counts().sort_index()
        st.line_chart(vc)

        st.markdown("### サンプル")
        st.dataframe(df[[target_col]].dropna().head(50), use_container_width=True)
        return

    # ---- カテゴリ / 文字列列
    st.markdown("### 頻度（カテゴリ/文字列）")
    vc = s.value_counts(dropna=False).head(30)
    st.dataframe(vc.rename("count").to_frame(), use_container_width=True)
    if len(vc) > 0:
        st.bar_chart(vc)

    st.markdown("### サンプル")
    st.dataframe(df[[target_col]].dropna().head(50), use_container_width=True)

    # カテゴリ×数値（オプション）
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) > 0:
        st.markdown("### カテゴリ × 数値（箱ひげ）")
        ycol = st.selectbox(
            "数値列を選択", options=num_cols, index=0, key=f"cat_num_{target_col}"
        )
        tmp = df[[target_col, ycol]].dropna().copy()

        topn = st.slider("カテゴリ上位N", 5, 50, 20, key=f"cat_topn_{target_col}")
        cats = tmp[target_col].value_counts().head(topn).index
        tmp = tmp[tmp[target_col].isin(cats)]

        if alt is not None and len(tmp) > 0:
            fx = a_field(target_col)
            fy = a_field(ycol)
            box_by_cat = (
                alt.Chart(tmp)
                .mark_boxplot()
                .encode(
                    x=alt.X(fx, type="nominal", sort="-y", title=target_col),
                    y=alt.Y(fy, type="quantitative", title=ycol),
                    tooltip=[
                        alt.Tooltip(fx, type="nominal", title=target_col),
                        alt.Tooltip(fy, type="quantitative", title=ycol),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(box_by_cat, use_container_width=True)
        else:
            st.write(tmp.groupby(target_col)[ycol].describe())


def render_filter(df: pd.DataFrame):
    st.subheader("フィルタ＆ダウンロード（実データ確認）")
    st.caption("列を選び、条件を指定して抽出します（簡易版）。")

    cols = list(df.columns)
    if len(cols) == 0:
        st.info("列がありません。")
        return

    target_col = st.selectbox("フィルタ対象列", cols, key="filter_col")
    s = df[target_col]
    filtered = df.copy()

    if pd.api.types.is_numeric_dtype(s):
        vmin = float(pd.to_numeric(s, errors="coerce").min())
        vmax = float(pd.to_numeric(s, errors="coerce").max())
        if pd.isna(vmin) or pd.isna(vmax):
            st.info("数値として扱える値がありません。")
        else:
            lo, hi = st.slider(
                "範囲",
                min_value=vmin,
                max_value=vmax,
                value=(vmin, vmax),
                key=f"num_range_{target_col}",
            )
            filtered = filtered[
                pd.to_numeric(filtered[target_col], errors="coerce").between(lo, hi)
            ]
    else:
        mode = st.radio(
            "条件",
            ["含む（部分一致）", "一致（完全一致）", "上位カテゴリから選択"],
            horizontal=True,
            key=f"str_mode_{target_col}",
        )
        if mode == "上位カテゴリから選択":
            top = s.value_counts(dropna=False).head(50).index.tolist()
            selected = st.multiselect("選択", options=top, default=top[:3])
            if selected:
                filtered = filtered[filtered[target_col].isin(selected)]
        else:
            q = st.text_input("検索文字列")
            if q:
                if mode == "一致（完全一致）":
                    filtered = filtered[filtered[target_col].astype(str) == q]
                else:
                    filtered = filtered[
                        filtered[target_col].astype(str).str.contains(q, na=False)
                    ]

    st.write(f"抽出結果: {filtered.shape[0]:,} 行 × {filtered.shape[1]:,} 列")
    st.dataframe(filtered.head(200), use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "抽出結果をCSVダウンロード",
        data=csv,
        file_name="filtered.csv",
        mime="text/csv",
    )


def render_viz(df: pd.DataFrame):
    st.subheader("可視化（簡易）")

    st.markdown("### 欠損率（上位20列）")
    miss = (df.isnull().mean().sort_values(ascending=False) * 100).head(20)
    if miss.max() == 0:
        st.success("欠損は見当たりません。")
    else:
        st.dataframe(miss.rename("missing_%").to_frame(), use_container_width=True)
        st.bar_chart(miss)

    st.markdown("### 相関（数値列）")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.info("数値列が2列以上ないため、相関は表示できません。")
        return

    corr = df[num_cols].corr(numeric_only=True)
    st.dataframe(corr, use_container_width=True)
