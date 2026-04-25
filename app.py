import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium

from streamlit_folium import st_folium
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


st.set_page_config(
    page_title="Демография РФ",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_css():
    with open("assets/style.css", encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


load_css()


@st.cache_data
def load_data():
    data = pd.read_csv("data/population.csv")
    data["year"] = data["year"].astype(int)

    numeric_columns = [
        "population",
        "birth_rate",
        "death_rate",
        "natural_growth",
        "migration",
        "density",
        "lat",
        "lon",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=["lat", "lon", "population"])


df = load_data()

if "selected_place" not in st.session_state:
    st.session_state.selected_place = df.iloc[0]["municipality"]


def format_number(value):
    return f"{int(value):,}".replace(",", " ")


def get_place_data(place, selected_year):
    place_data = df[df["municipality"] == place].sort_values("year")
    current = place_data[place_data["year"] == selected_year].iloc[0]

    previous = place_data[place_data["year"] < selected_year]

    if len(previous) > 0:
        prev = previous.iloc[-1]
        change = (current["population"] - prev["population"]) / prev["population"] * 100
    else:
        change = 0

    return place_data, current, float(change)


def build_growth_table(selected_year):
    start_year = df["year"].min()
    rows = []

    for municipality in df["municipality"].unique():
        temp = df[
            (df["municipality"] == municipality)
            & (df["year"].isin([start_year, selected_year]))
        ].sort_values("year")

        if len(temp) == 2:
            start_pop = float(temp.iloc[0]["population"])
            end_pop = float(temp.iloc[1]["population"])
            change = (end_pop - start_pop) / start_pop * 100

            rows.append({
                "Регион": temp.iloc[-1]["region"],
                "Муниципалитет": municipality,
                "Население в начале": int(start_pop),
                "Население сейчас": int(end_pop),
                "Рост/снижение, %": round(change, 2),
            })

    return pd.DataFrame(rows)


years = sorted(df["year"].unique())

st.sidebar.markdown("## Настройки")
selected_year = st.sidebar.selectbox("Год анализа", years, index=len(years) - 1)
forecast_horizon = st.sidebar.slider("Горизонт прогноза", 5, 15, 10)
map_mode = st.sidebar.selectbox(
    "Режим карты",
    ["Муниципалитеты", "Регионы"]
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Демографический мониторинг РФ</div>
        <div class="hero-subtitle">
            Интерактивный прототип веб-приложения для анализа, прогноза и ИИ-аналитики по муниципальным образованиям России.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_map, tab_list, tab_analytics = st.tabs(["Карта", "Список", "Аналитика"])

with tab_map:
    left, right = st.columns([4.7, 1.45])

    with left:
        st.markdown("## Интерактивная карта")

        latest = df[df["year"] == selected_year].copy()

        m = folium.Map(
            location=[58.0, 70.0],
            zoom_start=4,
            tiles="CartoDB dark_matter",
            control_scale=True,
            zoom_control=True,
        )

        if map_mode == "Регионы":
            region_points = (
                latest.groupby("region", as_index=False)
                .agg(
                    population=("population", "sum"),
                    density=("density", "mean"),
                    lat=("lat", "mean"),
                    lon=("lon", "mean"),
                )
            )

            for _, row in region_points.iterrows():
                marker_class = "pulse-marker-big" if row["population"] >= 1_500_000 else "pulse-marker"

                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    tooltip=row["region"],
                    popup=f"""
                    <div style="font-family:Arial;font-size:15px;">
                        <b>{row['region']}</b><br>
                        Население: {int(row['population']):,}<br>
                        Средняя плотность: {row['density']:.1f}
                    </div>
                    """,
                    icon=folium.DivIcon(
                        html=f'<div class="{marker_class}"></div>'
                    )
                ).add_to(m)

        else:
            for _, row in latest.iterrows():
                marker_class = "pulse-marker-big" if row["population"] >= 1_000_000 else "pulse-marker"

                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    tooltip=row["municipality"],
                    popup=f"""
                    <div style="font-family:Arial;font-size:15px;">
                        <b>{row['municipality']}</b><br>
                        Регион: {row['region']}<br>
                        Население: {int(row['population']):,}<br>
                        Плотность: {row['density']:.1f}
                    </div>
                    """,
                    icon=folium.DivIcon(
                        html=f'<div class="{marker_class}"></div>'
                    )
                ).add_to(m)

        result = st_folium(
            m,
            height=760,
            use_container_width=True,
            returned_objects=["last_object_clicked_tooltip"]
        )

        if result and result.get("last_object_clicked_tooltip"):
            clicked = result["last_object_clicked_tooltip"]

            if clicked in df["municipality"].unique():
                st.session_state.selected_place = clicked
            else:
                region_places = df[df["region"] == clicked]["municipality"].unique()
                if len(region_places) > 0:
                    st.session_state.selected_place = region_places[0]

    with right:
        place = st.session_state.selected_place
        place_data, current_row, population_change = get_place_data(place, selected_year)
        change_class = "good" if population_change >= 0 else "bad"

        st.markdown(
            f"""
            <div class="blue-card">
                <div class="metric-label">Выбранная территория</div>
                <div class="metric-value">{place}</div>
                <div style="font-size:18px;font-weight:800;margin-top:8px;">
                    {current_row["region"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="white-card">
                <div class="metric-label">Население сейчас</div>
                <div class="metric-value">{format_number(current_row["population"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="white-card">
                <div class="metric-label">Изменение за год</div>
                <div class="metric-value {change_class}">{population_change:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="white-card">
                <div class="metric-label">Плотность населения</div>
                <div class="metric-value">{current_row["density"]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card">
                Наведите курсор на светящуюся точку, чтобы увидеть данные.
                Кликните по точке, чтобы выбрать территорию для аналитики.
            </div>
            """,
            unsafe_allow_html=True
        )

with tab_list:
    st.markdown("## Список территорий")

    latest = df[df["year"] == selected_year].copy()
    search = st.text_input("Поиск региона или муниципалитета")

    if search:
        latest = latest[
            latest["region"].str.contains(search, case=False, na=False)
            | latest["municipality"].str.contains(search, case=False, na=False)
        ]

    table = latest[
        [
            "region",
            "municipality",
            "population",
            "birth_rate",
            "death_rate",
            "natural_growth",
            "migration",
            "density",
        ]
    ].rename(columns={
        "region": "Регион",
        "municipality": "Муниципалитет",
        "population": "Население",
        "birth_rate": "Рождаемость",
        "death_rate": "Смертность",
        "natural_growth": "Естественный прирост",
        "migration": "Миграция",
        "density": "Плотность",
    })

    st.dataframe(
        table.sort_values("Население", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## Рейтинг роста и снижения населения")

    growth_df = build_growth_table(selected_year)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Наибольший рост")
        st.dataframe(
            growth_df.sort_values("Рост/снижение, %", ascending=False).head(10),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("### Наибольшее снижение")
        st.dataframe(
            growth_df.sort_values("Рост/снижение, %", ascending=True).head(10),
            use_container_width=True,
            hide_index=True
        )

with tab_analytics:
    place = st.session_state.selected_place
    place_data, current_row, population_change = get_place_data(place, selected_year)

    st.markdown(f"## Аналитика: {place}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Население", format_number(current_row["population"]))
    c2.metric("Рост/снижение", f"{population_change:.2f}%")
    c3.metric("Рождаемость", current_row["birth_rate"])
    c4.metric("Смертность", current_row["death_rate"])

    fig_population = px.line(
        place_data,
        x="year",
        y="population",
        markers=True,
        title="Фактическая динамика численности населения",
        template="plotly_dark"
    )
    fig_population.update_traces(line=dict(width=4), marker=dict(size=10))
    fig_population.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)")
    st.plotly_chart(fig_population, use_container_width=True)

    demo_long = place_data.melt(
        id_vars=["year"],
        value_vars=["birth_rate", "death_rate", "natural_growth", "migration"],
        var_name="Показатель",
        value_name="Значение"
    )

    fig_demo = px.line(
        demo_long,
        x="year",
        y="Значение",
        color="Показатель",
        markers=True,
        title="Демографические показатели",
        template="plotly_dark"
    )
    fig_demo.update_traces(line=dict(width=4), marker=dict(size=9))
    fig_demo.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)")
    st.plotly_chart(fig_demo, use_container_width=True)

    st.markdown("## Прогноз численности населения")

    forecast_data = place_data[["year", "population"]].copy()
    y = forecast_data["population"].values

    if len(y) >= 5:
        train_size = int(len(y) * 0.8)
        train = y[:train_size]
        test = y[train_size:]

        model = ExponentialSmoothing(train, trend="add").fit()
        test_pred = model.forecast(len(test))

        mae = mean_absolute_error(test, test_pred)
        rmse = np.sqrt(mean_squared_error(test, test_pred))
        mape = np.mean(np.abs((test - test_pred) / test)) * 100

        final_model = ExponentialSmoothing(y, trend="add").fit()
        forecast = final_model.forecast(forecast_horizon)

        last_year = int(forecast_data["year"].max())
        future_years = list(range(last_year + 1, last_year + forecast_horizon + 1))

        fact_df = pd.DataFrame({
            "year": forecast_data["year"],
            "population": forecast_data["population"],
            "type": "Факт"
        })

        forecast_df = pd.DataFrame({
            "year": future_years,
            "population": forecast,
            "type": "Прогноз"
        })

        forecast_df["lower"] = forecast_df["population"] * 0.97
        forecast_df["upper"] = forecast_df["population"] * 1.03

        combined = pd.concat([fact_df, forecast_df], ignore_index=True)

        fig_forecast = px.line(
            combined,
            x="year",
            y="population",
            color="type",
            markers=True,
            title="Фактические данные и прогнозные значения",
            template="plotly_dark"
        )

        fig_forecast.add_scatter(
            x=forecast_df["year"],
            y=forecast_df["upper"],
            mode="lines",
            name="Верхняя граница интервала",
            line=dict(dash="dash", width=2)
        )

        fig_forecast.add_scatter(
            x=forecast_df["year"],
            y=forecast_df["lower"],
            mode="lines",
            name="Нижняя граница интервала",
            line=dict(dash="dash", width=2)
        )

        fig_forecast.update_traces(line=dict(width=4), marker=dict(size=9))
        fig_forecast.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)")
        st.plotly_chart(fig_forecast, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("MAPE", f"{mape:.2f}%")

    if population_change > 0:
        trend_text = "наблюдается положительная динамика численности населения"
        recommendation = "целесообразно планировать расширение социальной, транспортной и коммунальной инфраструктуры"
    elif population_change < 0:
        trend_text = "наблюдается снижение численности населения"
        recommendation = "целесообразно усилить меры поддержки занятости, семейной политики и удержания молодежи"
    else:
        trend_text = "численность населения остается относительно стабильной"
        recommendation = "целесообразно поддерживать текущий уровень инфраструктуры и регулярно отслеживать показатели"

    analytics = f"""
Аналитическая справка

Регион: {current_row["region"]}
Муниципальное образование: {place}
Год анализа: {selected_year}

1. Краткое резюме динамики населения

В выбранном муниципальном образовании {trend_text}.
Текущая численность населения составляет {format_number(current_row["population"])} человек.
Изменение за последний год составило {population_change:.2f}%.

2. Демографические тенденции и возможные факторы влияния

Коэффициент рождаемости: {current_row["birth_rate"]}.
Коэффициент смертности: {current_row["death_rate"]}.
Естественный прирост: {current_row["natural_growth"]}.
Миграция: {int(current_row["migration"])}.
Плотность населения: {current_row["density"]}.

Возможные факторы влияния:
- миграционный приток или отток;
- уровень занятости;
- доступность жилья;
- развитие социальной инфраструктуры;
- транспортная доступность;
- возрастная структура населения.

3. Прогнозная оценка

На горизонте 5–10 лет динамика будет зависеть от текущего тренда,
миграционных процессов, рождаемости, смертности и экономической привлекательности территории.

4. Рекомендации

{recommendation}.
Также рекомендуется использовать прогнозные значения при разработке программ социальной политики
и документов территориального планирования.
"""

    st.markdown("## ИИ-аналитическая справка")
    st.text_area("Сформированная справка", analytics, height=460)

    st.download_button(
        "Скачать справку",
        analytics,
        file_name=f"analytics_{place}.txt"
    )