"""
This script creates a comprehensive, interactive Streamlit dashboard for 
Exploratory Data Analysis (EDA), RFM analysis with K-Means clustering, 
Time-Series Sales Forecasting, and Dynamic Pricing recommendations.

To run this application:
1. Make sure you have the required libraries installed:
   pip install pandas openpyxl plotly streamlit scikit-learn seaborn squarify kaleido torch holidays statsmodels
2. Save this script as a single Python file (e.g., app.py).
3. Run this script from your terminal: streamlit run app.py
"""

# --- 0. LIBRARY IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import squarify
from typing import Optional, Dict, Tuple, List, Any
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import holidays

# --- 1. APP CONFIGURATION & INITIALIZATION ---
st.set_page_config(layout="wide", page_title="All-in-One Retail Analysis Dashboard")

# --- 1a. STYLING & COLOR PALETTE ---
COLOR_PALETTE = {
    "primary": "#1f77b4",    # Muted blue
    "secondary": "#ff7f0e",  # Safety orange
    "success": "#2ca02c",    # Cooked asparagus green
    "danger": "#d62728",     # Brick red
    "info": "#9467bd",       # Muted purple
    "light": "#aec7e8",      # Light blue
    "dark": "#2c3e50"        # Dark slate gray
}

# --- 2. DATA LOADING ---
@st.cache_data
def load_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from a file uploaded via Streamlit.
    Supports CSV and Excel formats.
    """
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 3. DATA CLEANING & PREPROCESSING FUNCTIONS ---

def handle_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with negative Quantity or zero Price."""
    return df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Removes outliers from specified columns using the IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for 'Customer ID' and 'Description'."""
    df_imputed = df.copy()
    df_imputed['Customer ID'].fillna('Unknown', inplace=True)
    stockcode_map = df_imputed.dropna(subset=['Description']).set_index('StockCode')['Description'].to_dict()
    df_imputed['Description'] = df_imputed['Description'].fillna(df_imputed['StockCode'].map(stockcode_map))
    df_imputed['Description'].fillna('Unknown', inplace=True)
    return df_imputed

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features like Revenue and time-based attributes."""
    df_featured = df.copy()
    df_featured['Revenue'] = df_featured['Quantity'] * df_featured['Price']
    df_featured['InvoiceDate'] = pd.to_datetime(df_featured['InvoiceDate'])
    df_featured['InvoiceYearMonth'] = df_featured['InvoiceDate'].dt.strftime('%Y-%m')
    df_featured['InvoiceWeekday'] = df_featured['InvoiceDate'].dt.day_name()
    df_featured['InvoiceHour'] = df_featured['InvoiceDate'].dt.hour
    return df_featured

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the entire data preprocessing workflow."""
    with st.spinner("Getting your data ready for analysis..."):
        st.write("Step 1: Removing returns and invalid entries...")
        df = handle_negative_values(df)
        st.write("Step 2: Removing extreme, unusual values...")
        df = remove_outliers_iqr(df, ['Quantity', 'Price'])
        st.write("Step 3: Filling in missing information...")
        df = impute_missing_values(df)
        st.write("Step 4: Creating new data points for analysis...")
        df = engineer_features(df)
    return df

# --- 4. EDA ANALYSIS & PLOTTING FUNCTIONS ---

def plot_monthly_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for monthly sales."""
    monthly_sales = df.groupby('InvoiceYearMonth')['Revenue'].sum().reset_index()
    fig = px.bar(
        monthly_sales, x='InvoiceYearMonth', y='Revenue',
        title=f'<b>{title_prefix}Monthly Sales Trend</b>',
        labels={'InvoiceYearMonth': 'Month', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_daily_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for daily sales."""
    daily_sales = df.groupby('InvoiceWeekday')['Revenue'].sum().reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['InvoiceWeekday'] = pd.Categorical(daily_sales['InvoiceWeekday'], categories=weekday_order, ordered=True)
    daily_sales = daily_sales.sort_values('InvoiceWeekday')
    fig = px.bar(
        daily_sales, x='InvoiceWeekday', y='Revenue',
        title=f'<b>{title_prefix}Sales by Day of the Week</b>',
        labels={'InvoiceWeekday': 'Day of the Week', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['secondary']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_hourly_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for hourly sales."""
    hourly_sales = df.groupby('InvoiceHour')['Revenue'].sum().reset_index()
    fig = px.bar(
        hourly_sales, x='InvoiceHour', y='Revenue',
        title=f'<b>{title_prefix}Sales by Hour of the Day</b>',
        labels={'InvoiceHour': 'Hour of the Day (24h format)', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['info']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_geographical_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a choropleth map of sales by country."""
    country_sales = df.groupby('Country')['Revenue'].sum().reset_index()
    fig = px.choropleth(
        country_sales,
        locations="Country",
        locationmode='country names',
        color="Revenue",
        hover_name="Country",
        color_continuous_scale=px.colors.sequential.Blues,
        title=f'<b>{title_prefix}Geographical Sales Distribution</b>'
    )
    return fig

def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Creates a horizontal bar chart of top N products by revenue."""
    top_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n).sort_values(ascending=True)
    fig = px.bar(
        top_products,
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        title=f'<b>What are your {top_n} best-selling products?</b>',
        labels={'x': 'Total Sales', 'y': 'Product'},
        color_discrete_sequence=[COLOR_PALETTE['success']]
    )
    return fig

def plot_worst_performers(df: pd.DataFrame, bottom_n: int = 10) -> go.Figure:
    """Creates horizontal bar charts of bottom N products and countries by revenue."""
    worst_products = df[df['Description'] != 'Unknown'].groupby('Description')['Revenue'].sum().nsmallest(bottom_n).sort_values(ascending=False)
    worst_countries = df.groupby('Country')['Revenue'].sum().nsmallest(bottom_n).sort_values(ascending=False)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"<b>Bottom {bottom_n} Products by Sales</b>", f"<b>Bottom {bottom_n} Countries by Sales</b>")
    )

    fig.add_trace(go.Bar(
        x=worst_products.values, y=worst_products.index, orientation='h',
        marker_color=COLOR_PALETTE['danger'], name='Worst Products'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=worst_countries.values, y=worst_countries.index, orientation='h',
        marker_color=COLOR_PALETTE['info'], name='Worst Countries'
    ), row=1, col=2)
    
    fig.update_layout(title_text=f'<b>Which products and countries are underperforming?</b>', showlegend=False, template='plotly_white')
    return fig

def plot_new_vs_returning_customers(df: pd.DataFrame) -> go.Figure:
    """Creates a pie chart showing revenue from new vs. returning customers."""
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    invoice_counts = known_customers_df.groupby('Customer ID')['Invoice'].nunique().reset_index()
    invoice_counts.columns = ['Customer ID', 'InvoiceCount']
    df_with_counts = pd.merge(known_customers_df, invoice_counts, on='Customer ID')
    df_with_counts['Customer_Category'] = np.where(df_with_counts['InvoiceCount'] == 1, 'New Customers', 'Returning Customers')
    revenue_summary = df_with_counts.groupby('Customer_Category')['Revenue'].sum().reset_index()
    fig = px.pie(
        revenue_summary, values='Revenue', names='Customer_Category',
        title='<b>Who brings in more money: New or Returning Customers?</b>',
        color_discrete_map={'Returning Customers': COLOR_PALETTE['primary'], 'New Customers': COLOR_PALETTE['light']}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_average_order_value(df: pd.DataFrame) -> go.Figure:
    """Creates a line chart showing the trend of Average Order Value (AOV)."""
    order_revenue = df.groupby(['InvoiceYearMonth', 'Invoice'])['Revenue'].sum().reset_index()
    aov_trend = order_revenue.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()
    fig = px.line(
        aov_trend, x='InvoiceYearMonth', y='Revenue',
        title='<b>Are customers spending more per order over time?</b>',
        labels={'InvoiceYearMonth': 'Month', 'Revenue': 'Average Order Value ($)'},
        markers=True, color_discrete_sequence=[COLOR_PALETTE['danger']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def analyze_market_basket(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Analyzes and returns a heatmap figure for market basket analysis."""
    top_products = df['Description'].value_counts().nlargest(top_n).index
    df_top = df[df['Description'].isin(top_products)]
    crosstab = pd.crosstab(df_top['Invoice'], df_top['Description'])
    crosstab[crosstab > 0] = 1
    co_occurrence_matrix = crosstab.T.dot(crosstab)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    fig = go.Figure(data=go.Heatmap(
        z=co_occurrence_matrix.values, x=co_occurrence_matrix.columns, y=co_occurrence_matrix.index, colorscale='Blues'))
    fig.update_layout(title=f'<b>Which of Your Top {top_n} Products are Bought Together?</b>',
                      xaxis_title="Products", yaxis_title="Products")
    return fig

def display_eda_insights(df: pd.DataFrame):
    """Generates and displays a summary of business insights from the EDA."""
    st.header("💡 Key Business Takeaways from the Overview")
    
    monthly_sales = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    daily_sales = df.groupby('InvoiceWeekday')['Revenue'].sum()
    hourly_sales = df.groupby('InvoiceHour')['Revenue'].sum()
    top_product = df.groupby('Description')['Revenue'].sum().nlargest(1)
    top_country = df.groupby('Country')['Revenue'].sum().nlargest(1)
    
    st.subheader("Performance Highlights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Busiest Month", monthly_sales.idxmax(), f"${monthly_sales.max():,.0f} in sales")
    col2.metric("Busiest Day", daily_sales.idxmax())
    col3.metric("Busiest Hour", f"{hourly_sales.idxmax()}:00 - {hourly_sales.idxmax()+1}:00")
    
    st.subheader("Your Star Performers")
    col1, col2 = st.columns(2)
    col1.metric("Top Product", top_product.index[0], f"${top_product.values[0]:,.0f} in sales")
    col2.metric("Top Country", top_country.index[0], f"${top_country.values[0]:,.0f} in sales")
    
    st.subheader("Actionable Advice")
    st.markdown(f"""
    - **Seasonal Strategy:** Your sales peak in **{monthly_sales.idxmax()}**. Plan your marketing campaigns and stock levels to take full advantage of this period.
    - **Weekly Promotions:** **{daily_sales.idxmax()}** is your strongest sales day. Consider running special promotions or flash sales on slower days to even out weekly revenue.
    - **Focus on Winners:** Your top product is **'{top_product.index[0]}'**. Ensure it's always in stock and consider bundling it with less popular items to boost their sales.
    - **Market Focus:** **{top_country.index[0]}** is your biggest market. Think about targeted advertising or country-specific deals to further grow this key area.
    """)

# --- 5. RFM & K-MEANS ANALYSIS FUNCTIONS ---

def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values."""
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (analysis_date - date.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns RFM scores and segments customers."""
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rf_score_str = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str)
    segment_map = {
        r'[1-2][1-2]': 'Sleeping', r'[1-2][3-4]': 'At Risk', r'[1-2]5': "Can't Lose Them",
        r'3[1-2]': 'Fading', r'33': 'Need Attention', r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising', r'51': 'New Customers', r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm_df['Segment'] = rf_score_str.replace(segment_map, regex=True)
    return rfm_df

def assign_business_actions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns suggested business actions to each segment."""
    business_actions = {
        "Sleeping": "Try to win them back with special offers or reminders.",
        "Loyal Customers": "Reward them with exclusive deals and loyalty points.",
        "Champions": "Treat them like VIPs! Offer early access and special gifts.",
        "At Risk": "Send personalized emails and discounts to bring them back.",
        "Potential Loyalists": "Encourage them to join a membership or loyalty program.",
        "Fading": "Send 'we miss you' emails with a special offer.",
        "Need Attention": "Ask for their feedback and suggest popular products.",
        "Can't Lose Them": "Offer a great deal to keep them from leaving.",
        "Promising": "Suggest other products they might like.",
        "New Customers": "Give them a great first experience and a welcome discount."
    }
    rfm_df["Action"] = rfm_df["Segment"].map(business_actions)
    return rfm_df

def plot_rfm_distribution(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart showing the distribution of customers across RFM segments."""
    segment_counts = rfm_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig = px.bar(
        segment_counts, 
        x='Count', 
        y='Segment', 
        orientation='h',
        title='<b>How many customers are in each group?</b>',
        labels={'Count': 'Number of Customers', 'Segment': 'Customer Group'},
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_rfm_sales_by_segment(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart showing total sales by RFM segment."""
    segment_sales = rfm_df.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        segment_sales,
        x='Segment',
        y='Monetary',
        title='<b>Which customer groups generate the most sales?</b>',
        labels={'Monetary': 'Total Sales', 'Segment': 'Customer Group'},
        color_discrete_sequence=[COLOR_PALETTE['success']]
    )
    return fig

def plot_rfm_pie_charts(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates pie charts for customer count and revenue share by RFM segment."""
    summary = rfm_df.groupby('Segment').agg(
        Customer_Count=('Recency', 'count'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=("Share of Total Customers", "Share of Total Sales")
    )
    fig.add_trace(go.Pie(
        labels=summary['Segment'], 
        values=summary['Customer_Count'], 
        name="Customers"
    ), 1, 1)
    fig.add_trace(go.Pie(
        labels=summary['Segment'], 
        values=summary['Total_Revenue'], 
        name="Sales"
    ), 1, 2)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="<b>How big and valuable is each customer group?</b>",
        legend_orientation="h"
    )
    return fig

def generate_business_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary table for each segment."""
    summary = rfm_df.groupby('Segment').agg(
        Customer_Count=('Recency', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Revenue=('Monetary', 'mean')
    ).round(2)
    total_customers = summary['Customer_Count'].sum()
    total_revenue = summary['Total_Revenue'].sum()
    summary['%_of_Customers'] = ((summary['Customer_Count'] / total_customers) * 100).round(2)
    summary['%_of_Revenue'] = ((summary['Total_Revenue'] / total_revenue) * 100).round(2)
    return summary.sort_values(by='Total_Revenue', ascending=False).reset_index()

def display_rfm_insights(rfm_df: pd.DataFrame):
    """Generates a text summary of insights from the RFM analysis."""
    st.header("💡 Insights from Simple Customer Groups")
    summary = rfm_df.groupby('Segment')['Monetary'].agg(['count', 'sum']).sort_values(by='sum', ascending=False)
    total_revenue = rfm_df['Monetary'].sum()
    total_customers = len(rfm_df)
    
    st.subheader(f"Your Most Valuable Group: **{summary.index[0]}**")
    st.markdown(f"""
    - The **'{summary.index[0]}'** group is your business's powerhouse.
    - Although they make up only **{summary.iloc[0]['count'] / total_customers:.1%}** of your customers, they generate **${summary.iloc[0]['sum']:,.0f}**, which is **{summary.iloc[0]['sum'] / total_revenue:.1%}** of your total sales!
    - **Action:** These are your VIPs. Focus your efforts on keeping them happy with exclusive rewards and personalized attention.
    """)
    
    st.subheader(f"Group to Re-engage: **{summary.index[-1]}**")
    st.markdown(f"""
    - The **'{summary.index[-1]}'** group represents customers who haven't purchased in a while and may be at risk of leaving.
    - **Action:** Launch a targeted 'we miss you' campaign with a special offer to win them back before they're gone for good.
