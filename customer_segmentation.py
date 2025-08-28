"""
This script performs a complete RFM (Recency, Frequency, Monetary) analysis
on customer data. It is designed for business use, generating a detailed
customer segmentation report, summary metrics, and saving all visualizations to files.
"""

# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import squarify  # A library for creating treemaps
from typing import Optional, Dict, Tuple
import os

# --- 1. DATA LOADING UTILITY ---

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from various file formats (CSV, Excel) into a pandas DataFrame.
    """
    print(f"Step 1: Loading data from '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file at '{file_path}' was not found.")
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        print("-> Data loaded successfully.")
        return df
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        print("-> Data loaded successfully.")
        return df
    else:
        raise ValueError(f"Unsupported file type: '{file_extension}'.")

# --- 2. DATA PREPARATION ---

def prepare_rfm_data(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str,
    date_format: str
) -> pd.DataFrame:
    """
    Prepares the input DataFrame for RFM analysis.
    """
    print("Step 2: Preparing and validating data...")
    local_df = df.copy()
    required_cols = [customer_id_col, invoice_date_col, revenue_col]
    
    missing_cols = [col for col in required_cols if col not in local_df.columns]
    if missing_cols:
        raise AssertionError(f"Error: The following required columns are missing: {missing_cols}")

    local_df[invoice_date_col] = pd.to_datetime(local_df[invoice_date_col], format=date_format)
    print("-> Data preparation complete.")
    return local_df

# --- 3. RFM CALCULATION AND SEGMENTATION ---

def calculate_rfm_metrics(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str
) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values for each customer."""
    analysis_date = df[invoice_date_col].max() + pd.Timedelta(days=1)
    recency = df.groupby(customer_id_col)[invoice_date_col].max().apply(lambda x: (analysis_date - x).days)
    frequency = df.groupby(customer_id_col)[invoice_date_col].count()
    monetary = df.groupby(customer_id_col)[revenue_col].sum()
    rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    return rfm

def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns RFM scores and segments customers based on R and F scores."""
    # Assign scores
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    # Define segments
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
    """Assigns suggested business actions to each customer segment."""
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

# --- 4. BUSINESS REPORTING AND VISUALIZATION ---

def generate_business_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary table with key business metrics for each segment."""
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

def plot_and_save(fig: Figure, save_path: str):
    """Saves a matplotlib figure to a file."""
    try:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Could not save figure to {save_path}. Error: {e}")

def create_visualizations(rfm_df: pd.DataFrame, output_dir: str) -> Dict[str, Figure]:
    """Generates, saves, and returns all standard RFM visualizations."""
    print("Step 4: Generating and saving visualizations...")
    
    figures = {}
    segment_summary = generate_business_summary(rfm_df)
    
    # Plot 1: Segment Distribution (Bar Chart)
    fig1 = plt.figure(figsize=(12, 7))
    sns.countplot(data=rfm_df, y='Segment', order=rfm_df['Segment'].value_counts().index, palette='muted')
    plt.title('Customer Segment Distribution', fontsize=16)
    plt.xlabel('Number of Customers', fontsize=12)
    plt.ylabel('Segment', fontsize=12)
    plt.tight_layout()
    plot_and_save(fig1, os.path.join(output_dir, '1_segment_distribution.png'))
    figures['segment_distribution'] = fig1

    # Plot 2: Recency vs. Frequency (Scatter Plot)
    fig2 = plt.figure(figsize=(12, 8))
    sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Segment', palette='muted', s=80, alpha=0.7)
    plt.title('Recency vs. Frequency by Segment', fontsize=16)
    plt.xlabel('Recency (Days since last purchase)', fontsize=12)
    plt.ylabel('Frequency (Total purchases)', fontsize=12)
    plt.legend(title='Customer Segments')
    plt.grid(True)
    plt.tight_layout()
    plot_and_save(fig2, os.path.join(output_dir, '2_recency_vs_frequency.png'))
    figures['recency_vs_frequency'] = fig2

    # Plot 3: Segment Proportions (Treemap)
    fig3 = plt.figure(figsize=(14, 9))
    squarify.plot(
        sizes=segment_summary['Customer_Count'],
        label=[f'{label}\n{count} customers\n({percent}%)' for label, count, percent in zip(segment_summary.index, segment_summary['Customer_Count'], segment_summary['%_of_Customers'])],
        color=sns.color_palette("muted", len(segment_summary)),
        alpha=0.8
    )
    plt.title('Treemap of Customer Segments by Count', fontsize=16)
    plt.axis('off')
    plot_and_save(fig3, os.path.join(output_dir, '3_segment_treemap.png'))
    figures['segment_treemap'] = fig3
    
    # Plot 4: Revenue Contribution by Segment (Pie Chart)
    fig4 = plt.figure(figsize=(15, 15))
    plt.pie(segment_summary['Total_Revenue'], labels=segment_summary.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("muted", len(segment_summary)))
    plt.title('Revenue Contribution by Segment', fontsize=16)
    plt.ylabel('') 
    plt.tight_layout()
    plot_and_save(fig4, os.path.join(output_dir, '4_revenue_by_segment_pie.png'))
    figures['revenue_pie_chart'] = fig4

    # Plot 5: Average Monetary Value by Segment (Bar Chart)
    avg_monetary_summary = segment_summary.sort_values('Avg_Monetary_Value', ascending=False)
    fig5 = plt.figure(figsize=(12, 7))
    sns.barplot(data=avg_monetary_summary, x='Avg_Monetary_Value', y=avg_monetary_summary.index, palette='muted')
    plt.title('Average Spend per Customer by Segment', fontsize=16)
    plt.xlabel('Average Monetary Value ($)', fontsize=12)
    plt.ylabel('Segment', fontsize=12)
    plt.tight_layout()
    plot_and_save(fig5, os.path.join(output_dir, '5_avg_monetary_value_bar.png'))
    figures['avg_monetary_bar'] = fig5
    
    print(f"-> All visualizations saved to '{output_dir}' directory.")
    return figures


# --- 5. MAIN PIPELINE ORCHESTRATOR ---

def rfm_analysis_pipeline(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str,
    date_format: str,
    output_csv_path: str = "customer_segmentation_report.csv",
    output_plot_dir: str = "plots"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Figure]]:
    """
    Orchestrates the complete RFM analysis pipeline for business reporting.
    """
    # Step 3: Calculate RFM and segment customers
    print("Step 3: Calculating RFM metrics and segmenting customers...")
    rfm_metrics = calculate_rfm_metrics(df, customer_id_col, invoice_date_col, revenue_col)
    rfm_segmented = segment_customers(rfm_metrics)
    rfm_final = assign_business_actions(rfm_segmented)
    print("-> Segmentation complete.")

    # Step 4: Generate visualizations and save them
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
    figures = create_visualizations(rfm_final, output_plot_dir)
    
    # Step 5: Generate business summary
    business_summary = generate_business_summary(rfm_final)
    
    # Step 6: Save detailed results to CSV
    if output_csv_path:
        rfm_final.to_csv(output_csv_path, index=True)
        print(f"\n✅ Detailed analysis saved to: {output_csv_path}")

    return rfm_final, business_summary, figures

# # --- HOW TO USE THE SCRIPT ---
# if __name__ == '__main__':
#     try:
#         # --- Configuration ---
#         FILE_PATH = '/content/refine_file.csv'
#         CUSTOMER_ID_COL = 'Customer ID'
#         INVOICE_DATE_COL = 'InvoiceDate'
#         REVENUE_COL = 'Revenue'
#         DATE_FORMAT = '%d-%m-%Y %H:%M' # Format for date conversion

#         # 1. Load data
#         input_df = load_data(FILE_PATH)
        
#         # 2. Prepare data
#         prepared_df = prepare_rfm_data(
#             df=input_df,
#             customer_id_col=CUSTOMER_ID_COL,
#             invoice_date_col=INVOICE_DATE_COL,
#             revenue_col=REVENUE_COL,
#             date_format=DATE_FORMAT
#         )

#         # 3. Run the main analysis pipeline
#         detailed_results_df, business_summary_df, generated_figures = rfm_analysis_pipeline(
#             df=prepared_df,
#             customer_id_col=CUSTOMER_ID_COL,
#             invoice_date_col=INVOICE_DATE_COL,
#             revenue_col=REVENUE_COL,
#             date_format=DATE_FORMAT
#         )

#         # 4. Display the high-level business summary
#         print("\n--- RFM Business Summary ---")
#         print(business_summary_df)
        
#         print("\n--- Recommended Actions per Segment ---")
#         actions_summary = detailed_results_df[['Segment', 'Action']].drop_duplicates().sort_values('Segment').set_index('Segment')
#         print(actions_summary)
        
#         # This will now display all the generated plots
#         print("\nDisplaying visualizations...")
#         plt.show()


#     except (FileNotFoundError, ValueError, AssertionError) as e:
#         print(f"\n--- ERROR ---")
#         print(e)
#         print("---------------")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
