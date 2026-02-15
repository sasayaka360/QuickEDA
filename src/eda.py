# src/eda.py
import io

import pandas as pd
import streamlit as st

# AltairはStreamlitでよく使う可視化。入っていない環境の場合は pip install altair
try:
    import altair as alt
except Exception:
    alt = None


# -----------------------
# I/O
# -----------------------
def parse_na_values(text: str):
    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x != ""]


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


def df_info_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


# -----------------------
# Metrics / Tables
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
# Column type helpers
# -----------------------
def try_parse_datetime(series: pd.Series, threshold: float = 0.6):
    """object列などをdatetimeにできそうなら変換して返す。無理ならNone。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype != "object":
        return None
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= threshold:
            return parsed
    except Exception:
        return None
    return None


# -----------------------
# Rendering functions (Streamlit UI parts)
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

    # 連動用の状態
    if "selected_col" not in st.session_state:
        st.session_state["selected_col"] = df.columns[0]

    st.caption(
        "表で1行選ぶと、その列が下の「列ごとの詳細」に連動します（対応Streamlitのみ）。"
    )

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

    # Altairがない環境でも最低限動くようにフォールバック
    if alt is None:
        st.warning(
            "Altairが未導入のため、可視化は簡易表示になります。必要なら `pip install altair` を実行してください。"
        )

    # 1) 数値列
    if pd.api.types.is_numeric_dtype(s):
        st.markdown("### 分布（数値）")

        data_num = pd.DataFrame({target_col: s.dropna()})

        cc1, cc2 = st.columns(2)
        with cc1:
            if alt is not None and len(data_num) > 0:
                bins = st.slider(
                    "ヒストグラム bins",
                    min_value=5,
                    max_value=100,
                    value=30,
                    key=f"bins_{target_col}",
                )
                hist = (
                    alt.Chart(data_num)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            f"{target_col}:Q",
                            bin=alt.Bin(maxbins=bins),
                            title=target_col,
                        ),
                        y=alt.Y("count():Q", title="count"),
                        tooltip=["count():Q"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(hist, use_container_width=True)
            else:
                st.bar_chart(data_num[target_col].value_counts(bins=30).sort_index())

        with cc2:
            if alt is not None and len(data_num) > 0:
                box = (
                    alt.Chart(data_num)
                    .mark_boxplot()
                    .encode(y=alt.Y(f"{target_col}:Q", title=target_col))
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

    # 2) 日付列
    elif s_dt is not None:
        st.markdown("### 時系列（日時）")
        tmp = pd.DataFrame({"dt": s_dt}).dropna()

        # ★ここが重要：必ずdatetimeに寄せる（ダメならNaTになる）
        tmp["dt"] = pd.to_datetime(tmp["dt"], errors="coerce")
        tmp = tmp.dropna()

        # それでもdatetimeにならない場合は日付扱いを諦める（安全策）
        if not pd.api.types.is_datetime64_any_dtype(tmp["dt"]):
            st.warning(
                "日時として解釈できない値が多いため、日時プロットをスキップしました。"
            )
            st.dataframe(tmp.head(50), use_container_width=True)
            return

        freq = st.selectbox(
            "集計粒度", ["日次", "週次", "月次"], index=0, key=f"freq_{target_col}"
        )

        if freq == "日次":
            g = tmp.groupby(tmp["dt"].dt.date).size().reset_index(name="count")
            g.columns = ["date", "count"]
            x_enc = alt.X("date:T", title="time") if alt is not None else "date"
        elif freq == "週次":
            g = (
                tmp.groupby(tmp["dt"].dt.to_period("W").astype(str))
                .size()
                .reset_index(name="count")
            )
            g.columns = ["week", "count"]
            x_enc = alt.X("week:N", title="time") if alt is not None else "week"
        else:
            g = (
                tmp.groupby(tmp["dt"].dt.to_period("M").astype(str))
                .size()
                .reset_index(name="count")
            )
            g.columns = ["month", "count"]
            x_enc = alt.X("month:N", title="time") if alt is not None else "month"

        if alt is not None:
            chart = (
                alt.Chart(g)
                .mark_line(point=True)
                .encode(
                    x=x_enc,
                    y=alt.Y("count:Q", title="count"),
                    tooltip=list(g.columns),
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(g.set_index(g.columns[0])["count"])

        st.dataframe(g.tail(30), use_container_width=True)

    # 3) カテゴリ/文字列列
    else:
        st.markdown("### 分布（カテゴリ/文字列）")
        topn = st.slider("上位Nカテゴリ", 5, 50, 20, key=f"topn_{target_col}")

        vc = s.value_counts(dropna=True).head(topn).reset_index()
        vc.columns = ["category", "count"]

        if alt is not None:
            bar = (
                alt.Chart(vc)
                .mark_bar()
                .encode(
                    y=alt.Y("category:N", sort="-x", title=target_col),
                    x=alt.X("count:Q", title="count"),
                    tooltip=["category:N", "count:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.dataframe(vc, use_container_width=True)

        st.markdown("### カテゴリ別の箱ひげ図（数値列がある場合）")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) == 0:
            st.info("数値列がないため、カテゴリ別箱ひげ図は表示できません。")
        else:
            ycol = st.selectbox(
                "Y（数値）を選択", options=num_cols, key=f"ycol_{target_col}"
            )
            tmp = df[[target_col, ycol]].dropna()

            limit_cats = st.checkbox(
                "カテゴリ数を上位に絞る（見やすさ優先）",
                value=True,
                key=f"limitcats_{target_col}",
            )
            if limit_cats:
                topcats = df[target_col].value_counts(dropna=True).head(20).index
                tmp = tmp[tmp[target_col].isin(topcats)]

            if alt is not None and len(tmp) > 0:
                box_by_cat = (
                    alt.Chart(tmp)
                    .mark_boxplot()
                    .encode(
                        x=alt.X(f"{target_col}:N", sort="-y", title=target_col),
                        y=alt.Y(f"{ycol}:Q", title=ycol),
                        tooltip=[
                            alt.Tooltip(f"{target_col}:N"),
                            alt.Tooltip(f"{ycol}:Q"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(box_by_cat, use_container_width=True)
            else:
                st.dataframe(tmp.head(200), use_container_width=True)

        st.markdown("### 上位カテゴリ一覧")
        st.dataframe(vc, use_container_width=True)

    st.markdown("### 生値サンプル")
    sample_mode = st.radio(
        "表示内容",
        ["先頭20", "欠損行（この列が欠損）", "非欠損行（この列が埋まっている）"],
        horizontal=True,
        key=f"samplemode_{target_col}",
    )
    if sample_mode == "先頭20":
        st.dataframe(df[[target_col]].head(20), use_container_width=True)
    elif sample_mode == "欠損行（この列が欠損）":
        st.dataframe(df[df[target_col].isnull()].head(50), use_container_width=True)
    else:
        st.dataframe(df[df[target_col].notnull()].head(50), use_container_width=True)


def render_filter(df: pd.DataFrame):
    st.subheader("フィルタ/抽出")

    show_cols = st.multiselect(
        "表示する列", options=list(df.columns), default=list(df.columns)
    )
    view_df = df[show_cols].copy() if show_cols else df.copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        only_missing = st.checkbox("欠損がある行だけ")
    with c2:
        only_duplicates = st.checkbox("重複行だけ")
    with c3:
        limit_rows = st.number_input(
            "表示行数上限", min_value=10, max_value=20000, value=500, step=10
        )

    if only_missing:
        view_df = view_df[view_df.isnull().any(axis=1)]
    if only_duplicates:
        view_df = view_df[view_df.duplicated(keep=False)]

    st.dataframe(view_df.head(int(limit_rows)), use_container_width=True)
    st.caption(f"抽出後: {len(view_df):,} 行")

    st.subheader("CSVとしてダウンロード（抽出結果）")
    csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "抽出結果をCSVでダウンロード",
        data=csv_bytes,
        file_name="filtered.csv",
        mime="text/csv",
    )


def render_viz(df: pd.DataFrame):
    st.subheader("可視化")

    st.markdown("### ヒストグラム（数値列）")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) == 0:
        st.info("数値列がないため、ヒストグラムは表示できません。")
    else:
        x = st.selectbox("数値列を選択", options=num_cols, key="viz_num_col")
        bins = st.slider("bins", min_value=5, max_value=100, value=30, key="viz_bins")
        st.bar_chart(df[x].dropna().value_counts(bins=bins).sort_index())

    st.markdown("### 相関（数値列）")
    if len(num_cols) < 2:
        st.info("数値列が2列以上ないため、相関は表示できません。")
    else:
        corr = df[num_cols].corr(numeric_only=True)
        st.dataframe(corr, use_container_width=True)

    st.markdown("### 欠損率の可視化（上位20列）")
    miss = df.isnull().mean().sort_values(ascending=False).head(20) * 100
    if miss.max() == 0:
        st.success("欠損は見当たりません（少なくとも上位20列では0%）。")
    else:
        st.bar_chart(miss)
