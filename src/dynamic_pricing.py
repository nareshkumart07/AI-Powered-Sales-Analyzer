"""
This module contains all functions related to dynamic pricing recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from typing import Tuple

# Import shared resources
from utilities import COLOR_PALETTE

def recommend_optimal_price(model: torch.nn.Module, daily_df: pd.DataFrame, scaler, seq_length: int, target_col_idx: int, price_col_idx: int, num_features: int, num_days_to_simulate: int) -> Tuple[pd.Series, pd.DataFrame]:
    """Simulates different prices over a future period to find the one that maximizes total revenue."""
    model.eval()
    
    original_columns = daily_df.columns
    last_sequence_df = daily_df.tail(seq_length)
    
    current_price = last_sequence_df.iloc[-1, price_col_idx]
    if current_price == 0:
        st.warning("Current price is zero, cannot simulate price changes. Please check your data.")
        return None, None
        
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
    
    results = []
    with torch.no_grad():
        for price in price_range:
            future_quantities = []
            temp_sequence_df = last_sequence_df.copy()
            
            temp_sequence_df.iloc[:, price_col_idx] = price
            
            scaled_sequence = scaler.transform(temp_sequence_df)
            current_sequence_tensor = torch.from_numpy(scaled_sequence).unsqueeze(0).float()

            for _ in range(num_days_to_simulate):
                predicted_quantity_scaled = model(current_sequence_tensor).item()
                
                dummy_df = pd.DataFrame(np.zeros((1, num_features)), columns=original_columns)
                dummy_df.iloc[0, target_col_idx] = predicted_quantity_scaled
                
                predicted_quantity = scaler.inverse_transform(dummy_df)[0, target_col_idx]
                predicted_quantity = max(0, predicted_quantity)
                future_quantities.append(predicted_quantity)

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
                  title=f'<b>How Changing the Price Might Affect Your Total Sales</b>',
                  labels={'Price': 'Price ($)', 'Total_Predicted_Revenue': f'Predicted Total Sales in the next {num_days_to_simulate} Days ($)'},
                  markers=True, color_discrete_sequence=[COLOR_PALETTE['primary']])
    
    fig.add_trace(go.Scatter(
        x=[optimal_row['Price']],
        y=[optimal_row['Total_Predicted_Revenue']],
        mode='markers',
        marker=dict(color=COLOR_PALETTE['danger'], size=12, symbol='star'),
        name='Best Price'
    ))
    
    fig.add_annotation(
        x=optimal_row['Price'],
        y=optimal_row['Total_Predicted_Revenue'],
        text=f"Best Price: ${optimal_row['Price']:.2f}<br>Total Sales: ${optimal_row['Total_Predicted_Revenue']:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-40
    )
    return fig

def display_pricing_insights(optimal_row: pd.Series, current_price: float, num_days_to_simulate: int, results_df: pd.DataFrame):
    """Generates and displays business insights for the pricing recommendation."""
    st.header("📈 Smart Pricing Insights")
    
    if optimal_row is None or results_df is None:
        return

    revenue_at_current_price_row = results_df.iloc[(results_df['Price']-current_price).abs().argsort()[:1]]
    if revenue_at_current_price_row.empty:
        st.warning("Could not calculate revenue uplift.")
        return
        
    revenue_at_current_price = revenue_at_current_price_row['Total_Predicted_Revenue'].values[0]
    potential_uplift = optimal_row['Total_Predicted_Revenue'] - revenue_at_current_price
    uplift_pct = (potential_uplift / revenue_at_current_price) * 100 if revenue_at_current_price > 0 else 0

    st.subheader("Recommendation Summary:")
    price_change_pct = ((optimal_row['Price'] - current_price) / current_price) * 100
    change_direction = "increase" if price_change_pct > 0 else "decrease"
    
    col1, col2 = st.columns(2)
    col1.metric(label="Recommended Best Price", value=f"${optimal_row['Price']:.2f}", delta=f"{price_change_pct:.2f}% vs. Current")
    col2.metric(label=f"Potential Extra Sales ({num_days_to_simulate} days)", value=f"${potential_uplift:,.2f}", help=f"An estimated increase of {uplift_pct:.2f}% compared to keeping the current price.")
    
    col3, col4 = st.columns(2)
    col3.metric(label=f"Expected Items Sold", value=f"{optimal_row['Total_Predicted_Quantity']:.0f} items")
    col4.metric(label=f"Expected Total Sales", value=f"${optimal_row['Total_Predicted_Revenue']:,.2f}")

    st.subheader("Your Action Plan:")
    st.markdown(f"""
    - **The Action:** The model suggests you **{change_direction}** the price by **{abs(price_change_pct):.1f}%**, from **${current_price:.2f}** to **${optimal_row['Price']:.2f}**.
    - **Why?** This new price is the 'sweet spot' that is predicted to make you the most money over the next **{num_days_to_simulate} days**, potentially increasing your sales by **${potential_uplift:,.2f}**.
    - **What this means:** If a price `{change_direction}` leads to higher total sales, it suggests customers are **{'not very sensitive' if change_direction == 'increase' else 'very sensitive'}** to price changes for this product right now.
    - **How to test it:** Try out the new price for a week or two to see if it works as predicted before making it permanent. Keep an eye on sales and customer feedback.
    """)

