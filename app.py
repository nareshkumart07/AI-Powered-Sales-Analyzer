"""
This script creates a comprehensive, interactive Streamlit dashboard for 
Exploratory Data Analysis (EDA), RFM analysis with K-Means clustering, 
Time-Series Sales Forecasting, and Dynamic Pricing recommendations.

To run this application:
1. Make sure you have the required libraries installed:
   pip install pandas openpyxl plotly streamlit scikit-learn seaborn squarify kaleido torch holidays
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
    with st.spinner("Running full preprocessing pipeline..."):
        st.write("Step 1: Handling negative values...")
        df = handle_negative_values(df)
        st.write("Step 2: Removing outliers...")
        df = remove_outliers_iqr(df, ['Quantity', 'Price'])
        st.write("Step 3: Imputing missing values...")
        df = impute_missing_values(df)
        st.write("Step 4: Engineering features...")
        df = engineer_features(df)
    return df

# --- 4. EDA ANALYSIS & PLOTTING FUNCTIONS ---

def analyze_top_performers(df: pd.DataFrame, top_n: int = 10) -> List[go.Figure]:
    """Analyzes and returns figures for top-performing products and countries."""
    figs = []
    # Top Revenue Products
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n).sort_values(ascending=True)
    fig1 = px.bar(top_revenue_products, y=top_revenue_products.index, x=top_revenue_products.values,
                  orientation='h', title=f'Top {top_n} Products by Revenue', labels={'x': 'Revenue', 'y': 'Product'},
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    figs.append(fig1)

    # Top Quantity Products
    top_quantity_products = df.groupby('Description')['Quantity'].sum().nlargest(top_n).sort_values(ascending=True)
    fig2 = px.bar(top_quantity_products, y=top_quantity_products.index, x=top_quantity_products.values,
                  orientation='h', title=f'Top {top_n} Products by Quantity Sold', labels={'x': 'Quantity Sold', 'y': 'Product'},
                  color_discrete_sequence=[COLOR_PALETTE['info']])
    figs.append(fig2)

    # Top Revenue Countries
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(top_n).sort_values(ascending=True)
    fig3 = px.bar(top_revenue_countries, y=top_revenue_countries.index, x=top_revenue_countries.values,
                  orientation='h', title=f'Top {top_n} Countries by Revenue', labels={'x': 'Revenue', 'y': 'Country'},
                  color_discrete_sequence=[COLOR_PALETTE['success']])
    figs.append(fig3)
    
    return figs

def analyze_temporal_trends(df: pd.DataFrame) -> List[go.Figure]:
    """Analyzes and returns figures for monthly and weekly trends."""
    figs = []
    # Monthly Revenue
    monthly_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum().reset_index()
    fig1 = px.line(monthly_revenue, x='InvoiceYearMonth', y='Revenue', markers=True,
                   title='Monthly Revenue Trend', labels={'InvoiceYearMonth': 'Month', 'Revenue': 'Total Revenue'},
                   color_discrete_sequence=[COLOR_PALETTE['primary']])
    figs.append(fig1)

    # Weekday Revenue
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum().reindex(weekday_order).reset_index()
    fig2 = px.line(weekday_revenue, x='InvoiceWeekday', y='Revenue', markers=True,
                   title='Weekday Revenue Trend', labels={'InvoiceWeekday': 'Day of the Week', 'Revenue': 'Total Revenue'},
                   color_discrete_sequence=[COLOR_PALETTE['secondary']])
    figs.append(fig2)
    
    return figs

def analyze_price_and_basket(df: pd.DataFrame) -> List[go.Figure]:
    """Analyzes and returns figures for price and basket size distributions."""
    figs = []
    # Price Distribution
    fig1 = px.histogram(df, x='Price', nbins=50, title='Distribution of Unit Prices',
                        labels={'Price': 'Unit Price'}, color_discrete_sequence=[COLOR_PALETTE['primary']])
    figs.append(fig1)

    # Basket Size Distribution
    basket_sizes = df.groupby('Invoice')['StockCode'].count()
    fig2 = px.histogram(basket_sizes, x=basket_sizes.values, nbins=40, title='Distribution of Basket Sizes',
                        labels={'x': 'Number of Items in Basket'}, color_discrete_sequence=[COLOR_PALETTE['info']])
    fig2.update_xaxes(range=[0, basket_sizes.quantile(0.99)])
    figs.append(fig2)
    
    return figs

def analyze_customer_behavior(df: pd.DataFrame) -> go.Figure:
    """Analyzes and returns a figure for new vs. returning customer revenue."""
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    customer_invoice_count = known_customers_df.groupby('Customer ID')['Invoice'].nunique()
    returning_customer_ids = customer_invoice_count[customer_invoice_count > 1].index
    returning_revenue = known_customers_df[known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    new_revenue = known_customers_df[~known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    revenue_data = pd.DataFrame({'Customer Type': ['Returning', 'New'], 'Revenue': [returning_revenue, new_revenue]})
    
    fig = px.bar(revenue_data, x='Customer Type', y='Revenue', color='Customer Type',
                 title='Revenue from New vs. Returning Customers',
                 color_discrete_map={'Returning': COLOR_PALETTE['primary'], 'New': COLOR_PALETTE['light']})
    return fig

def plot_geographic_heatmap(df: pd.DataFrame) -> go.Figure:
    """Returns a Plotly choropleth map figure."""
    country_revenue = df.groupby('Country')['Revenue'].sum().reset_index()
    fig = px.choropleth(country_revenue,
                        locations="Country", locationmode='country names',
                        color="Revenue", hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Blues,
                        title='Geographic Distribution of Revenue')
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
        z=co_occurrence_matrix.values,
        x=co_occurrence_matrix.columns,
        y=co_occurrence_matrix.index,
        colorscale='Blues'))
    fig.update_layout(title=f'Co-occurrence Matrix of Top {top_n} Products',
                      xaxis_title="Products", yaxis_title="Products")
    return fig

def display_eda_business_insights(df: pd.DataFrame):
    """Generates and displays the final business insights summary for EDA."""
    st.header("📊 EDA Business Insights & Recommendations")
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(1)
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(1)
    monthly_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum()
    hourly_revenue = df.groupby('InvoiceHour')['Revenue'].sum()
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    customer_invoice_count = known_customers_df.groupby('Customer ID')['Invoice'].nunique()
    returning_customer_ids = customer_invoice_count[customer_invoice_count > 1].index
    returning_rev = known_customers_df[known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    new_rev = known_customers_df[~known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    total_rev = returning_rev + new_rev
    returning_share = (returning_rev / total_rev) * 100 if total_rev > 0 else 0

    st.subheader("💡 Top Performers & Peak Times:")
    st.markdown(f"- **Top Revenue Product:** '{top_revenue_products.index[0]}'")
    st.markdown(f"- **Top Market (Country):** '{top_revenue_countries.index[0]}'")
    st.markdown(f"- **Peak Sales Month:** {monthly_revenue.idxmax()}")
    st.markdown(f"- **Busiest Day of the Week:** {weekday_revenue.idxmax()}")
    st.markdown(f"- **Peak Sales Hour:** {hourly_revenue.idxmax()}:00 - {hourly_revenue.idxmax()+1}:00")

    st.subheader("💡 Customer Behavior Insights:")
    st.markdown(f"- Returning customers drive **{returning_share:.1f}%** of identified customer revenue.")

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
        r'[1-2][1-2]': 'Hibernating', r'[1-2][3-4]': 'At Risk', r'[1-2]5': "Can't Lose Them",
        r'3[1-2]': 'About to Sleep', r'33': 'Need Attention', r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising', r'51': 'New Customers', r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm_df['Segment'] = rf_score_str.replace(segment_map, regex=True)
    return rfm_df

def assign_business_actions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns suggested business actions to each segment."""
    business_actions = {
        "Hibernating": "Re-engage with reminder emails and seasonal promotions.",
        "Loyal Customers": "Strengthen relationship with exclusive offers and loyalty rewards.",
        "Champions": "Reward with VIP programs, personalized gifts, and ambassador opportunities.",
        "At Risk": "Launch win-back campaigns, offer special discounts, and request feedback.",
        "Potential Loyalists": "Nurture with targeted recommendations and membership incentives.",
        "About to Sleep": "Send wake-up campaigns and birthday/anniversary offers.",
        "Need Attention": "Use personalized outreach, highlight trending products, and send surveys.",
        "Can't Lose Them": "Offer strong retention incentives and reactivation bundles.",
        "Promising": "Encourage repeat purchases with targeted recommendations and cross-selling.",
        "New Customers": "Deliver a strong onboarding experience and welcome discounts."
    }
    rfm_df["Action"] = rfm_df["Segment"].map(business_actions)
    return rfm_df

def generate_business_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary table for each segment."""
    summary = rfm_df.groupby('Segment').agg(
        Customer_Count=('Recency', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary_Value=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).round(2)
    total_customers = summary['Customer_Count'].sum()
    total_revenue = summary['Total_Revenue'].sum()
    summary['%_of_Customers'] = ((summary['Customer_Count'] / total_customers) * 100).round(2)
    summary['%_of_Revenue'] = ((summary['Total_Revenue'] / total_revenue) * 100).round(2)
    summary = summary.sort_values(by='Total_Revenue', ascending=False)
    return summary

def create_rfm_visualizations(rfm_df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Generates and returns all RFM visualizations."""
    figures = {}
    segment_summary = generate_business_summary(rfm_df)

    # Plot 1: Segment Distribution
    segment_counts = rfm_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig1 = px.bar(segment_counts, y='Segment', x='Count', orientation='h',
                  title='Customer Segment Distribution',
                  labels={'Count': 'Number of Customers', 'Segment': 'Segment'},
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
    figures['segment_distribution'] = fig1

    # Plot 2: Recency vs Frequency
    fig2 = px.scatter(rfm_df, x='Recency', y='Frequency', color='Segment',
                      title='Recency vs. Frequency by Segment',
                      hover_data=['Monetary'],
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    figures['recency_vs_frequency'] = fig2

    # Plot 3: Treemap
    segment_summary_for_treemap = segment_summary.reset_index()
    fig3 = px.treemap(segment_summary_for_treemap, path=['Segment'], values='Customer_Count',
                      title='Treemap of Customer Segments by Count',
                      hover_data=['%_of_Revenue'],
                      color_continuous_scale=px.colors.sequential.Blues)
    figures['segment_treemap'] = fig3

    return figures

def find_optimal_clusters(df: pd.DataFrame) -> go.Figure:
    """Calculates and plots the elbow curve to find the optimal number of clusters."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)
    
    wcss = []
    k_range = range(1, 11)
    for i in k_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    
    fig = px.line(x=list(k_range), y=wcss, markers=True,
                  title='Elbow Method for Optimal K',
                  labels={'x': 'Number of clusters', 'y': 'WCSS'},
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    return fig

def perform_kmeans_clustering(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Performs K-Means clustering on the RFM data."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    return df_clustered

def plot_kmeans_clusters(df: pd.DataFrame) -> go.Figure:
    """Creates an interactive 3D scatter plot of the K-Means clusters."""
    fig = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary',
                        color='Cluster', symbol='Cluster',
                        size_max=18, opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), title='3D K-Means Cluster Visualization')
    return fig

def display_kmeans_business_insights(df: pd.DataFrame):
    """Generates and displays business insights for K-Means clusters."""
    st.header("📊 Advanced Segmentation Business Insights")
    cluster_summary = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Customer_Count'}).round(2)

    st.write("#### Cluster Profiles:")
    for i, row in cluster_summary.iterrows():
        st.write(f"**Cluster {i}:**")
        st.write(f"- **Characteristics:** Customers with an average recency of {row['Recency']} days, {row['Frequency']} purchases, and a total spend of ${row['Monetary']:.2f}.")
        # Simple interpretation logic
        if row['Monetary'] > cluster_summary['Monetary'].mean() and row['Frequency'] > cluster_summary['Frequency'].mean():
            st.write("- **Interpretation:** High-Value, Frequent Buyers.")
            st.write("- **Recommendation:** Treat as VIPs. Offer loyalty rewards and exclusive access.")
        elif row['Recency'] > cluster_summary['Recency'].mean():
            st.write("- **Interpretation:** At-Risk or Churned Customers.")
            st.write("- **Recommendation:** Launch win-back campaigns with special offers.")
        else:
            st.write("- **Interpretation:** Standard or New Customers.")
            st.write("- **Recommendation:** Nurture with targeted marketing to increase frequency and spend.")


# --- 6. FORECASTING MODEL DEFINITIONS & PIPELINE FUNCTIONS ---

# --- 6a. Model Architectures ---
class LSTMModel(nn.Module):
    """Defines the structure of the LSTM neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """Defines the structure of the GRU neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# --- 6b. Forecasting Helper Functions ---

def prepare_and_engineer_features_forecast(
    df: pd.DataFrame,
    product_stock_code: str,
    competitor_df: Optional[pd.DataFrame] = None,
    customer_segment_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """Filters, prepares, merges external data, and engineers features for forecasting."""
    
    product_df = df[df['StockCode'] == product_stock_code].copy()
    if product_df.empty:
        st.error(f"Error: No data found for StockCode {product_stock_code}.")
        return None

    product_df.set_index('InvoiceDate', inplace=True)
    daily_sales_df = product_df.resample('D')['Quantity'].sum().to_frame()

    if competitor_df is not None:
        try:
            competitor_df['Date'] = pd.to_datetime(competitor_df['Date'], errors='coerce')
            competitor_df.set_index('Date', inplace=True)
            daily_comp_prices = competitor_df.resample('D').mean()
            daily_sales_df = daily_sales_df.merge(daily_comp_prices, left_index=True, right_index=True, how='left')
            price_cols = ['our_price', 'competitor_A', 'competitor_B', 'competitor_C']
            for col in price_cols:
                if col in daily_sales_df.columns:
                    daily_sales_df[col].ffill(inplace=True)
                    daily_sales_df[col].bfill(inplace=True)
        except Exception as e:
            st.warning(f"Could not process competitor data. Skipping. Error: {e}")

    if customer_segment_df is not None:
        try:
            customer_segment_df['Date'] = pd.to_datetime(customer_segment_df['Date'], errors='coerce')
            daily_segment_sales = customer_segment_df.pivot_table(index='Date', columns='Segment', values='Quantity', aggfunc='sum')
            daily_segment_sales = daily_segment_sales.resample('D').sum()
            daily_sales_df = daily_sales_df.merge(daily_segment_sales, left_index=True, right_index=True, how='left')
            daily_sales_df[daily_segment_sales.columns] = daily_sales_df[daily_segment_sales.columns].fillna(0)
        except Exception as e:
            st.warning(f"Could not process customer segment data. Skipping. Error: {e}")

    df_featured = daily_sales_df.copy()
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['month'] = df_featured.index.month
    df_featured['week_of_year'] = df_featured.index.isocalendar().week.astype(int)
    
    for i in [1, 7, 14, 30]:
        df_featured[f'lag_{i}_days'] = df_featured['Quantity'].shift(i)
    
    for window in [7, 14, 30]:
        df_featured[f'rolling_mean_{window}_days'] = df_featured['Quantity'].shift(1).rolling(window=window).mean()
        df_featured[f'rolling_std_{window}_days'] = df_featured['Quantity'].shift(1).rolling(window=window).std()

    uk_holidays = holidays.UK(years=df_featured.index.year.unique())
    df_featured['is_holiday'] = df_featured.index.map(lambda x: 1 if x in uk_holidays else 0)
    
    df_featured.fillna(0, inplace=True)
    return df_featured

def scale_and_create_sequences(daily_df: pd.DataFrame, seq_length: int, forecast_horizon: int):
    """Scales data and creates sequences for model training."""
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_df)
    target_col_idx = daily_df.columns.get_loc('Quantity')
    X, y = [], []
    for i in range(len(scaled_features) - seq_length - forecast_horizon + 1):
        X.append(scaled_features[i:i+seq_length])
        y.append(scaled_features[i+seq_length:i+seq_length+forecast_horizon, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler, target_col_idx

def split_data_and_create_loaders(X: np.ndarray, y: np.ndarray, train_split_ratio: float, val_split_ratio: float, batch_size: int = 32):
    """Splits data and creates PyTorch DataLoaders."""
    total_samples = len(X)
    train_size = int(total_samples * train_split_ratio)
    val_size = int(total_samples * val_split_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, y_test, X_test

def train_model(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, training_params: Dict) -> nn.Module:
    """Generic model training function with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    best_val_loss = float('inf')
    patience_counter = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(training_params['num_epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_loss += criterion(model(batch_X), batch_y).item()
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        progress_bar.progress((epoch + 1) / training_params['num_epochs'])
        status_text.text(f"Epoch {epoch+1}/{training_params['num_epochs']} | Validation Loss: {avg_val_loss:.5f}")

        if patience_counter >= training_params['patience']:
            st.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    progress_bar.empty()
    status_text.empty()
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler: MinMaxScaler, y_test: np.ndarray, target_col_idx: int, num_features: int) -> Tuple[pd.DataFrame, Dict]:
    """Evaluates the model and returns predictions and metrics."""
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            all_predictions.append(model(batch_X).cpu().numpy())
    predictions_scaled = np.concatenate(all_predictions)
    predictions_full = np.zeros((predictions_scaled.shape[0], num_features))
    predictions_full[:, target_col_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(predictions_full)[:, target_col_idx]
    y_test_full = np.zeros((len(y_test), num_features))
    y_test_full[:, target_col_idx] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(y_test_full)[:, target_col_idx]
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    metrics = {'MAE': mae, 'RMSE': rmse}
    results_df = pd.DataFrame({'Actual': y_test_actual, 'Predicted': predictions})
    return results_df, metrics

def generate_future_forecasts(model: nn.Module, daily_df: pd.DataFrame, scaler: MinMaxScaler, seq_length: int, target_col_idx: int, num_features: int, num_days: int) -> pd.DataFrame:
    """Generates forecasts for a specified number of future days."""
    model.eval()
    last_sequence_full = daily_df.tail(seq_length).values
    last_sequence_scaled = scaler.transform(last_sequence_full)
    current_sequence = torch.from_numpy(last_sequence_scaled).unsqueeze(0).float()
    future_predictions_scaled = []
    with torch.no_grad():
        for _ in range(num_days):
            prediction = model(current_sequence)
            future_predictions_scaled.append(prediction.item())
            new_row_scaled = current_sequence.numpy().squeeze()[-1].copy()
            new_row_scaled[target_col_idx] = prediction.item()
            new_sequence_np = np.vstack([current_sequence.numpy().squeeze()[1:], new_row_scaled])
            current_sequence = torch.from_numpy(new_sequence_np).unsqueeze(0).float()
    future_predictions_full = np.zeros((len(future_predictions_scaled), num_features))
    future_predictions_full[:, target_col_idx] = future_predictions_scaled
    future_predictions = scaler.inverse_transform(future_predictions_full)[:, target_col_idx]
    last_date = daily_df.index[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, num_days + 1)])
    future_df = pd.DataFrame({'Date': future_dates, 'Future_Forecast': future_predictions})
    future_df.set_index('Date', inplace=True)
    return future_df

def plot_forecast_dashboard(daily_df: pd.DataFrame, results_df: pd.DataFrame, future_df: pd.DataFrame, product_stock_code: str) -> go.Figure:
    """Creates a multi-panel dashboard for business insights using Plotly with enhanced visibility."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sales Forecast vs. Actuals (Test & Future)',
            'Price Elasticity of Demand',
            'Average Sales by Day of Week',
            'Sales Trends and Rolling Averages'
        ),
        vertical_spacing=0.15 # Add some vertical spacing
    )

    # --- Panel 1: Forecast Timeline (Enhanced Visibility) ---
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Actual'], name='Actual Sales (Test)',
        mode='lines+markers',
        line=dict(color=COLOR_PALETTE['primary'], width=2),
        marker=dict(size=5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Predicted'], name='Predicted Sales (Test)',
        mode='lines',
        line=dict(color=COLOR_PALETTE['danger'], dash='dash', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=future_df.index, y=future_df['Future_Forecast'], name='Future Forecast',
        mode='lines',
        line=dict(color=COLOR_PALETTE['success'], dash='dot', width=2)
    ), row=1, col=1)

    # --- Panel 2: Price Elasticity ---
    if 'our_price' in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df['our_price'], y=daily_df['Quantity'], mode='markers',
            name='Price vs Quantity',
            marker=dict(opacity=0.6, color=COLOR_PALETTE['info'])
        ), row=1, col=2)
    else:
        fig.add_annotation(text="Price data not available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=1, col=2)

    # --- Panel 3: Sales by Day of Week ---
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_sales = daily_df.groupby('day_of_week')['Quantity'].mean().reindex(range(7))
    day_of_week_sales.index = day_names
    fig.add_trace(go.Bar(
        x=day_of_week_sales.index, y=day_of_week_sales.values,
        name='Avg Sales',
        marker_color=COLOR_PALETTE['primary']
    ), row=2, col=1)

    # --- Panel 4: Rolling Averages (Enhanced Visibility) ---
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df['Quantity'], name='Daily Sales',
        mode='lines',
        line=dict(color='lightgrey', width=1)
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df['Quantity'].rolling(7).mean(), name='7-Day Rolling Mean',
        mode='lines',
        line=dict(color=COLOR_PALETTE['secondary'], width=2.5)
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df['Quantity'].rolling(30).mean(), name='30-Day Rolling Mean',
        mode='lines',
        line=dict(color=COLOR_PALETTE['dark'], width=2.5)
    ), row=2, col=2)

    # --- Update Layout for Better Visibility ---
    fig.update_layout(
        height=900, # Increased height
        title_text=f'<b>Business Dashboard for Product: {product_stock_code}</b>',
        title_font_size=24,
        template='plotly_white', # Cleaner template
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, b=40, t=100) # Adjust margins
    )
    
    # Update subplot title fonts
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=16)

    return fig


def display_forecast_insights(daily_df: pd.DataFrame, metrics: Dict):
    """Generates and displays actionable business insights."""
    st.header("💡 Actionable Business Insights from Forecast")

    st.subheader("1. Model Performance & Inventory Management:")
    st.markdown(f"- The model forecasts sales with a **Mean Absolute Error (MAE) of {metrics['MAE']:.2f} units**.")
    st.markdown(f"- **INSIGHT:** On any given day, the forecast could be off by approximately **{metrics['MAE']:.0f} units**. Consider setting safety stock levels to at least this amount to mitigate stockouts.")

    st.subheader("2. Marketing & Staffing Optimization:")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_sales = daily_df.groupby('day_of_week')['Quantity'].mean()
    if not day_of_week_sales.empty:
        best_day_name = day_names[day_of_week_sales.idxmax()]
        worst_day_name = day_names[day_of_week_sales.idxmin()]
        st.markdown(f"- **Weekly Sales Pattern:** Highest average sales occur on **{best_day_name}**, while the lowest are on **{worst_day_name}**.")
        st.markdown(f"- **INSIGHT:** Ensure optimal staffing and inventory on **{best_day_name}s**. Consider running targeted promotions on **{worst_day_name}s** to boost sales.")

# --- 6c. Dynamic Pricing Functions ---

def recommend_optimal_price(model: nn.Module, daily_df: pd.DataFrame, scaler: MinMaxScaler, seq_length: int, target_col_idx: int, price_col_idx: int, num_features: int, num_days_to_simulate: int) -> Tuple[pd.Series, pd.DataFrame]:
    """Simulates different prices over a future period to find the one that maximizes total revenue."""
    model.eval()
    last_sequence_full = daily_df.tail(seq_length).values
    
    current_price = last_sequence_full[-1, price_col_idx]
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20) # Test prices from -20% to +20%
    
    results = []
    with torch.no_grad():
        for price in price_range:
            # For each price, run a multi-day forecast
            future_quantities = []
            temp_sequence_full = last_sequence_full.copy()
            
            # Set the price for the simulation period
            temp_sequence_full[:, price_col_idx] = price
            
            scaled_sequence = scaler.transform(temp_sequence_full)
            current_sequence_tensor = torch.from_numpy(scaled_sequence).unsqueeze(0).float()

            for _ in range(num_days_to_simulate):
                predicted_quantity_scaled = model(current_sequence_tensor).item()
                
                # Inverse transform the single predicted quantity
                dummy_array = np.zeros((1, num_features))
                dummy_array[0, target_col_idx] = predicted_quantity_scaled
                predicted_quantity = scaler.inverse_transform(dummy_array)[0, target_col_idx]
                predicted_quantity = max(0, predicted_quantity)
                future_quantities.append(predicted_quantity)

                # Prepare the next input sequence
                new_row_scaled = current_sequence_tensor.numpy().squeeze()[-1].copy()
                new_row_scaled[target_col_idx] = predicted_quantity_scaled
                
                new_sequence_np = np.vstack([current_sequence_tensor.numpy().squeeze()[1:], new_row_scaled])
                current_sequence_tensor = torch.from_numpy(new_sequence_np).unsqueeze(0).float()

            total_predicted_quantity = sum(future_quantities)
            total_predicted_revenue = price * total_predicted_quantity
            results.append({'Price': price, 'Total_Predicted_Quantity': total_predicted_quantity, 'Total_Predicted_Revenue': total_predicted_revenue})

    results_df = pd.DataFrame(results)
    optimal_row = results_df.loc[results_df['Total_Predicted_Revenue'].idxmax()]
    return optimal_row, results_df

def plot_price_recommendation(results_df: pd.DataFrame, optimal_row: pd.Series, num_days_to_simulate: int) -> go.Figure:
    """Visualizes the price vs. revenue curve and highlights the optimal point."""
    fig = px.line(results_df, x='Price', y='Total_Predicted_Revenue',
                  title=f'Predicted Total Revenue over {num_days_to_simulate} Days by Price Point',
                  labels={'Price': 'Price ($)', 'Total_Predicted_Revenue': f'Total Predicted Revenue over {num_days_to_simulate} Days ($)'},
                  markers=True, color_discrete_sequence=[COLOR_PALETTE['primary']])
    
    fig.add_trace(go.Scatter(
        x=[optimal_row['Price']],
        y=[optimal_row['Total_Predicted_Revenue']],
        mode='markers',
        marker=dict(color=COLOR_PALETTE['danger'], size=12, symbol='star'),
        name='Optimal Price'
    ))
    
    fig.add_annotation(
        x=optimal_row['Price'],
        y=optimal_row['Total_Predicted_Revenue'],
        text=f"Optimal: ${optimal_row['Price']:.2f}<br>Total Revenue: ${optimal_row['Total_Predicted_Revenue']:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-40
    )
    return fig

def display_pricing_insights(optimal_row: pd.Series, current_price: float, num_days_to_simulate: int):
    """Generates and displays business insights for the pricing recommendation."""
    st.header("📈 Dynamic Pricing Insights")
    st.subheader("Recommendation Summary:")
    
    price_change_pct = ((optimal_row['Price'] - current_price) / current_price) * 100
    change_direction = "increase" if price_change_pct > 0 else "decrease"
    
    st.metric(label="Recommended Optimal Price", value=f"${optimal_row['Price']:.2f}", delta=f"{price_change_pct:.2f}% vs. Current")
    st.metric(label=f"Expected Total Sales Quantity over {num_days_to_simulate} days", value=f"{optimal_row['Total_Predicted_Quantity']:.0f} units")
    st.metric(label=f"Expected Total Revenue over {num_days_to_simulate} days", value=f"${optimal_row['Total_Predicted_Revenue']:.2f}")

    st.subheader("Actionable Business Strategy:")
    st.markdown(f"""
    - **The Action:** The model suggests a price **{change_direction} of {abs(price_change_pct):.1f}%** from the current price of ${current_price:.2f} to **${optimal_row['Price']:.2f}**.
    - **The Rationale:** This new price point is predicted to find the 'sweet spot' that maximizes your total revenue over the next **{num_days_to_simulate} days**.
    - **Implementation:** Consider A/B testing this new price point for a short period to validate the model's prediction in a real-world scenario before a full rollout. Monitor sales and customer feedback closely.
    """)


# --- 6d. Main Forecasting Pipeline Orchestrator ---

def run_forecasting_pipeline(
    model_type: str,
    df: pd.DataFrame,
    product_stock_code: str,
    competitor_df: Optional[pd.DataFrame] = None,
    customer_segment_df: Optional[pd.DataFrame] = None,
    future_forecast_days: int = 30,
    seq_length: int = 60,
    forecast_horizon: int = 1,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    seed: int = 42
):
    """Orchestrates the complete time-series forecasting pipeline."""
    st.info(f"Starting {model_type} Forecast Pipeline for Product: {product_stock_code}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Step 1: Data Prep & Feature Engineering
    daily_sales_df = prepare_and_engineer_features_forecast(
        df, product_stock_code, competitor_df, customer_segment_df
    )
    if daily_sales_df is None or daily_sales_df.empty:
        st.error("Pipeline halted due to data preparation issues.")
        return

    # Step 2: Data Processing
    X, y, scaler, target_col_idx = scale_and_create_sequences(daily_sales_df, seq_length, forecast_horizon)
    train_loader, val_loader, test_loader, y_test, X_test = split_data_and_create_loaders(X, y, train_split_ratio, val_split_ratio)
    
    if len(X_test) == 0:
        st.error("Not enough data to create a test set. Please check data size and split ratios.")
        return

    # Step 3: Model Training
    model_params = {'input_size': X.shape[2], 'hidden_size': 128, 'num_layers': 2, 'output_size': forecast_horizon}
    training_params = {'num_epochs': 50, 'learning_rate': 0.005, 'patience': 10}
    
    model = LSTMModel(**model_params) if model_type == 'LSTM' else GRUModel(**model_params)
    
    st.write("Model training in progress...")
    model = train_model(train_loader, val_loader, model, training_params)
    st.success("Model training complete!")

    # Step 4: Evaluation
    results_df, metrics = evaluate_model(model, test_loader, scaler, y_test, target_col_idx, daily_sales_df.shape[1])
    results_df.index = daily_sales_df.index[-len(results_df):]

    # Step 5: Future Forecasting
    future_df = generate_future_forecasts(
        model=model, daily_df=daily_sales_df, scaler=scaler, seq_length=seq_length,
        target_col_idx=target_col_idx, num_features=daily_sales_df.shape[1], num_days=future_forecast_days
    )

    # Step 6: Visualization and Reporting
    st.subheader(f"{model_type} Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f}")
    
    fig = plot_forecast_dashboard(daily_sales_df, results_df, future_df, product_stock_code)
    st.plotly_chart(fig, use_container_width=True)
    
    display_forecast_insights(daily_sales_df, metrics)
    
    st.subheader(f"Future Forecast Data ({model_type})")
    st.dataframe(future_df)
    st.info(f"Pipeline Finished for Product: {product_stock_code}")
    
    # Store the trained model and necessary data for dynamic pricing
    st.session_state.trained_model = model
    st.session_state.daily_sales_df = daily_sales_df
    st.session_state.scaler = scaler
    st.session_state.seq_length = seq_length
    st.session_state.target_col_idx = target_col_idx
    st.session_state.model_trained = True


# --- 7. STREAMLIT APPLICATION LAYOUT ---
def main():
    st.title('🛒 All-in-One Retail Analysis Dashboard')

    # Session State Initialization
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # --- File Uploader ---
    with st.sidebar:
        st.sidebar.title("🛒 Retail Dashboard")
        st.header("1. Data Upload")
        uploaded_file = st.file_uploader("Upload your retail data file (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
        if uploaded_file:
            st.session_state.raw_df = load_data(uploaded_file)
            if st.session_state.raw_df is not None:
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")

    if not st.session_state.data_loaded:
        st.info("Please upload a data file to begin analysis.")
        return
        
    with st.expander("ℹ️ How to Use This Dashboard"):
        st.markdown("""
        **Step 1: Upload Your Data**
        - Use the sidebar to upload your retail sales data in CSV or Excel format.

        **Step 2: Preprocess the Data**
        - Click the "Run Full Preprocessing Pipeline" button to clean and prepare your data for analysis.

        **Step 3: Explore Your Data (EDA)**
        - Navigate to the "📊 Exploratory Data Analysis (EDA)" tab to gain insights into top performers, trends, and customer behavior.

        **Step 4: Segment Your Customers**
        - In the "👥 Customer Segmentation" tab, use RFM and K-Means clustering to group your customers and understand their value.

        **Step 5: Forecast Future Sales**
        - Go to the "📈 Sales Forecasting & Pricing" tab to predict future sales for a selected product.
        - To enable dynamic pricing, make sure to upload a competitor prices CSV with a column named 'our_price'.

        **Step 6: Get Price Recommendations**
        - After running a sales forecast, the "Dynamic Pricing Recommendation" section will appear.
        - Adjust the simulation period and click "Recommend Optimal Price" to find the best price for maximizing revenue.
        """)

    st.header("Raw Data Preview")
    st.dataframe(st.session_state.raw_df.head())
    st.markdown("---")

    # --- Preprocessing Section ---
    st.header("⚙️ 2. Data Preprocessing")
    if st.button("Run Full Preprocessing Pipeline"):
        st.session_state.df_cleaned = preprocess_pipeline(st.session_state.raw_df)
        st.success("Preprocessing complete!")
        st.dataframe(st.session_state.df_cleaned.head())
    
    st.markdown("---")

    if st.session_state.df_cleaned is None:
        st.warning("Please run the preprocessing pipeline to proceed with the analysis.")
        return

    # --- Main Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "👥 Customer Segmentation", "📈 Sales Forecasting & Pricing"])

    with tab1:
        st.header("📊 Exploratory Data Analysis (EDA)")
        
        with st.expander("Explore Top Performers & Trends"):
            if st.button("Analyze Top Performers"):
                with st.spinner("Analyzing..."):
                    figs = analyze_top_performers(st.session_state.df_cleaned)
                    for fig in figs: st.plotly_chart(fig, use_container_width=True)

            if st.button("Analyze Temporal Trends"):
                with st.spinner("Analyzing..."):
                    figs = analyze_temporal_trends(st.session_state.df_cleaned)
                    for fig in figs: st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Explore Price, Basket, and Customer Behavior"):
            if st.button("Analyze Price and Basket Size"):
                with st.spinner("Analyzing..."):
                    figs = analyze_price_and_basket(st.session_state.df_cleaned)
                    for fig in figs: st.plotly_chart(fig, use_container_width=True)

            if st.button("Analyze Customer Behavior"):
                with st.spinner("Analyzing..."):
                    fig = analyze_customer_behavior(st.session_state.df_cleaned)
                    st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Explore Geographic & Product Co-occurrence"):
            if st.button("Analyze Geographic Distribution"):
                with st.spinner("Analyzing..."):
                    fig = plot_geographic_heatmap(st.session_state.df_cleaned)
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Analyze Market Basket"):
                with st.spinner("Analyzing..."):
                    fig = analyze_market_basket(st.session_state.df_cleaned)
                    st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Generate Final EDA Business Insights"):
            with st.spinner("Generating insights..."):
                display_eda_business_insights(st.session_state.df_cleaned)

    with tab2:
        st.header("👥 Customer Segmentation")
        segment_tab1, segment_tab2 = st.tabs(["Standard Segmentation (RFM)", "Advanced Segmentation (K-Means)"])

        with segment_tab1:
            st.subheader("Standard RFM Segmentation")
            if st.button("Run Standard Segmentation"):
                with st.spinner("Running Standard Segmentation..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    rfm_segmented = segment_customers(rfm_metrics)
                    rfm_final = assign_business_actions(rfm_segmented)
                    st.dataframe(generate_business_summary(rfm_final))
                    actions = rfm_final[['Segment', 'Action']].drop_duplicates().set_index('Segment')
                    st.table(actions)
                    figs = create_rfm_visualizations(rfm_final)
                    for fig_name, fig in figs.items(): st.plotly_chart(fig, use_container_width=True)

        with segment_tab2:
            st.subheader("Advanced K-Means Segmentation")
            if st.button("Find Optimal K (Elbow Method)"):
                with st.spinner("Calculating..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    fig = find_optimal_clusters(rfm_metrics)
                    st.plotly_chart(fig, use_container_width=True)

            k_clusters = st.slider("Select number of clusters (K)", 2, 10, 4, key='kmeans_k')
            if st.button("Run Advanced Clustering"):
                with st.spinner("Clustering..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    kmeans_clustered = perform_kmeans_clustering(rfm_metrics, k_clusters)
                    st.dataframe(kmeans_clustered.head())
                    fig = plot_kmeans_clusters(kmeans_clustered)
                    st.plotly_chart(fig, use_container_width=True)
                    display_kmeans_business_insights(kmeans_clustered)

    with tab3:
        st.header("📈 Sales Forecasting")
        
        # User Inputs for Forecasting
        product_list = st.session_state.df_cleaned['StockCode'].unique()
        selected_product = st.selectbox("Select a Product StockCode to Forecast", product_list)
        
        forecast_days = st.number_input("How many days to forecast into the future?", min_value=7, max_value=90, value=30, step=7)
        
        st.markdown("**(Optional) Upload additional data to improve forecast accuracy:**")
        col1_upload, col2_upload = st.columns(2)
        with col1_upload:
            competitor_file = st.file_uploader("Upload Competitor Prices CSV", type=['csv'], key="competitor")
        with col2_upload:
            segment_file = st.file_uploader("Upload Customer Segments CSV", type=['csv'], key="segment")

        # Load optional files if provided
        competitor_data = load_data(competitor_file) if competitor_file else None
        segment_data = load_data(segment_file) if segment_file else None

        # Forecasting model buttons
        col1_model, col2_model = st.columns(2)
        with col1_model:
            if st.button("Forecast using LSTM Model"):
                run_forecasting_pipeline(
                    model_type='LSTM',
                    df=st.session_state.df_cleaned,
                    product_stock_code=selected_product,
                    future_forecast_days=forecast_days,
                    competitor_df=competitor_data,
                    customer_segment_df=segment_data
                )
        with col2_model:
            if st.button("Forecast using GRU Model"):
                run_forecasting_pipeline(
                    model_type='GRU',
                    df=st.session_state.df_cleaned,
                    product_stock_code=selected_product,
                    future_forecast_days=forecast_days,
                    competitor_df=competitor_data,
                    customer_segment_df=segment_data
                )

        if st.session_state.model_trained:
            st.markdown("---")
            st.header("💰 Dynamic Pricing Recommendation")
            if 'our_price' in st.session_state.daily_sales_df.columns:
                st.subheader("Configure Pricing Simulation")
                pricing_forecast_days = st.number_input(
                    "How many days into the future to simulate for pricing?",
                    min_value=1, max_value=90, value=7, step=1, key="pricing_days"
                )

                if st.button("Recommend Optimal Price"):
                    with st.spinner(f"Simulating prices over {pricing_forecast_days} days..."):
                        price_col_idx = st.session_state.daily_sales_df.columns.get_loc('our_price')
                        optimal_row, price_results_df = recommend_optimal_price(
                            model=st.session_state.trained_model,
                            daily_df=st.session_state.daily_sales_df,
                            scaler=st.session_state.scaler,
                            seq_length=st.session_state.seq_length,
                            target_col_idx=st.session_state.target_col_idx,
                            price_col_idx=price_col_idx,
                            num_features=st.session_state.daily_sales_df.shape[1],
                            num_days_to_simulate=pricing_forecast_days
                        )
                        
                        price_fig = plot_price_recommendation(price_results_df, optimal_row, pricing_forecast_days)
                        st.plotly_chart(price_fig, use_container_width=True)
                        
                        current_price = st.session_state.daily_sales_df['our_price'].iloc[-1]
                        display_pricing_insights(optimal_row, current_price, pricing_forecast_days)
            else:
                st.warning("To enable dynamic pricing, please upload competitor data including an 'our_price' column.")


if __name__ == '__main__':
    main()
