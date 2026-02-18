# file: dashboard_vendite.py

import numpy as np
import pandas as pd
from datetime import timedelta
import streamlit as st
import plotly.express as px

# =========================
# PARTE 0 – CREAZIONE DATASET DI ESEMPIO
# =========================

def genera_dataset(n=500, random_state=42):
    np.random.seed(random_state)

    # Date ordine e spedizione
    order_dates = pd.date_range('2022-01-01', '2024-12-31', periods=n)
    ship_dates = order_dates + pd.to_timedelta(
        np.random.randint(1, 7, size=n), unit='D'
    )

    # Dimensioni categoriali
    categories = ['Furniture', 'Office Supplies', 'Technology']
    sub_categories = {
        'Furniture': ['Chairs', 'Tables', 'Bookcases'],
        'Office Supplies': ['Paper', 'Binders', 'Storage'],
        'Technology': ['Phones', 'Computers', 'Accessories']
    }
    regions = ['North', 'South', 'East', 'West']
    states = ['Lombardia', 'Lazio', 'Toscana', 'Sicilia', 'Veneto', 'Piemonte']

    rows = []
    for i in range(n):
        cat = np.random.choice(categories)
        rows.append({
            'Order Date': order_dates[i],
            'Ship Date': ship_dates[i],
            'Category': cat,
            'Sub-Category': np.random.choice(sub_categories[cat]),
            'Sales': round(np.random.uniform(20, 2000), 2),
            'Profit': round(np.random.uniform(-200, 600), 2),
            'Region': np.random.choice(regions),
            'State': np.random.choice(states),
            'Quantity': np.random.randint(1, 10)
        })

    df = pd.DataFrame(rows)
    return df


# =========================
# PARTE 1 – PULIZIA DATI
# =========================

def pulizia_dati(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Convertire le colonne data in datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    # 2. Controllare valori nulli e duplicati
    # (qui ci limitiamo a segnalarli, ma potresti dropparli o imputarli)
    nulli = df.isna().sum()
    duplicati = df.duplicated().sum()

    st.sidebar.subheader("Info pulizia dati")
    st.sidebar.write("Valori nulli per colonna:")
    st.sidebar.write(nulli)
    st.sidebar.write(f"Righe duplicate: {duplicati}")

    # Esempio: rimozione duplicati (opzionale)
    df = df.drop_duplicates()

    # 3. Creare colonna Year dall’Order Date
    df['Year'] = df['Order Date'].dtyear

    return df


# =========================
# PARTE 2 – ANALISI ESPLORATIVA
# =========================

def analisi_eda(df: pd.DataFrame):
    st.title("Dashboard Vendite & Profitto")

    # Filtri laterali
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Seleziona anno/i", years, default=years
    )

    categories = sorted(df['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Seleziona categoria/e", categories, default=categories
    )

    # Applico filtri
    mask = df['Year'].isin(selected_years) & df['Category'].isin(selected_categories)
    dff = df[mask]

    st.subheader("Dataset filtrato")
    st.dataframe(dff.head())

    # 4. Totale vendite e profitti per anno
    st.subheader("Totale vendite e profitti per anno")
    yearly = (
        dff
        .groupby('Year', as_index=False)
        .agg({'Sales': 'sum', 'Profit': 'sum'})
    )

    col1, col2 = st.columns(2)
    col1.metric("Vendite totali (selezione)", f"€ {yearly['Sales'].sum():,.2f}")
    col2.metric("Profitto totale (selezione)", f"€ {yearly['Profit'].sum():,.2f}")

    fig_year = px.bar(
        yearly,
        x='Year',
        y=['Sales', 'Profit'],
        barmode='group',
        title='Vendite e Profitti per Anno'
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # 5. Top 5 sottocategorie più vendute (per Sales)
    st.subheader("Top 5 sottocategorie per vendite")
    subcat = (
        dff
        .groupby('Sub-Category', as_index=False)
        .agg({'Sales': 'sum'})
        .sort_values('Sales', ascending=False)
        .head(5)
    )

    fig_subcat = px.bar(
        subcat,
        x='Sub-Category',
        y='Sales',
        title='Top 5 Sottocategorie per Vendite',
        text_auto=True
    )
    st.plotly_chart(fig_subcat, use_container_width=True)

    # 6. Mappa interattiva delle vendite
    # Per semplicità, mappiamo le "State" italiane a coordinate fittizie
    st.subheader("Mappa interattiva delle vendite per Stato")

    state_coords = {
        'Lombardia': (45.4668, 9.1905),
        'Lazio': (41.9028, 12.4964),
        'Toscana': (43.7711, 11.2486),
        'Sicilia': (37.5027, 14.0120),
        'Veneto': (45.4419, 12.3155),
        'Piemonte': (45.0703, 7.6869),
    }

    geo_df = (
        dff
        .groupby('State', as_index=False)
        .agg({'Sales': 'sum'})
    )

    geo_df['lat'] = geo_df['State'].map(lambda s: state_coords[s][0])
    geo_df['lon'] = geo_df['State'].map(lambda s: state_coords[s][1])

    fig_map = px.scatter_mapbox(
        geo_df,
        lat='lat',
        lon='lon',
        size='Sales',
        color='Sales',
        hover_name='State',
        hover_data={'lat': False, 'lon': False},
        zoom=4,
        height=500,
        color_continuous_scale='Blues',
        title='Vendite per Stato (dimensione e colore = Sales)'
    )

    fig_map.update_layout(
        mapbox_style='open-street-map',
        margin=dict(r=0, t=40, l=0, b=0)
    )

    st.plotly_chart(fig_map, use_container_width=True)


# =========================
# MAIN
# =========================

def main():
    # In un progetto reale qui leggeresti da CSV:
    # df = pd.read_csv('sales_data.csv')
    df = genera_dataset(n=500)  # dati inventati per esempio
    df = pulizia_dati(df)
    analisi_eda(df)


if __name__ == "__main__":
    main()
