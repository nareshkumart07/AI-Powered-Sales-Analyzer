"""
This is the main application file for the Streamlit Sales & Pricing Helper.
To run this app, use the command: streamlit run app.py
"""
import streamlit as st
import pandas as pd

# Import modules
from data_processing import load_data, preprocess_pipeline
from eda import (
    plot_monthly_sales, plot_daily_sales, plot_hourly_sales,
    plot_top_products, plot_geographical_sales, plot_worst_performers,
    plot_new_vs_returning_customers, plot_average_order_value,
    analyze_market_basket, display_eda_insights
)
from customer_segmentation import (
    calculate_rfm_metrics, segment_customers, assign_business_actions,
    generate_business_summary, plot_rfm_pie_charts, plot_rfm_sales_by_segment,
    plot_rfm_distribution, display_rfm_insights, find_optimal_clusters,
    perform_kmeans_clustering, get_cluster_names, generate_kmeans_summary_table,
    plot_kmeans_pie_charts, plot_kmeans_sales_by_segment, plot_kmeans_bar_charts,
    display_kmeans_business_insights, merge_data_with_segments
)
from forecasting import run_forecasting_pipeline
from dynamic_pricing import recommend_optimal_price, plot_price_recommendation, display_pricing_insights

# --- APP CONFIGURATION & INITIALIZATION ---
st.set_page_config(layout="wide", page_title="AI-Powered Sales Analyzer & Forecasting Tool")

def main():
    st.title('AI-Powered Sales Analyzer & Forecasting Tool')

    with st.expander("ℹ️ How to Use This Tool & Data Requirements", expanded=True):
        st.markdown("""
        ### **Required Data Format**
        Before you start, make sure your data file (CSV or Excel) contains the following columns with these exact names:
        - **Invoice**: The unique ID for each transaction (e.g., 536365).
        - **StockCode**: The unique ID for each product (e.g., 85123A).
        - **Description**: The name of the product (e.g., WHITE HANGING HEART T-LIGHT HOLDER).
        - **Quantity**: How many items were bought (e.g., 6).
        - **InvoiceDate**: The date and time of the sale (e.g., 12/1/2010 8:26).
        - **Price**: The price of a single item (e.g., 2.55).
        - **Customer ID**: The unique ID for each customer (e.g., 17850).
        - **Country**: The country where the sale was made (e.g., United Kingdom).
        
        ---

        ### **How to Use the Dashboard**
        **Step 1: Upload Your Data**
        - Use the sidebar to upload your sales data. The tool works best with files that have a full year of sales history.

        **Step 2: Check & Prepare Data**
        - The tool will automatically check if you have the required columns.
        - Click the **"Prepare My Data"** button to clean it up for analysis.

        **Step 3: Get Business Insights**
        - **Business Overview:** Get a big-picture look at your sales, top products, and customers.
        - **Understand Your Customers:** Group your customers to see who your best ones are and who needs attention.
        - **Future Predictions & Pricing:** Predict future sales for any product and get smart suggestions on the best price to set.
        """)

    # --- SESSION STATE INITIALIZATION ---
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # --- SIDEBAR & FILE UPLOAD ---
    with st.sidebar:
        st.sidebar.title("Controls")
        st.header("1. Upload Your Data")
        uploaded_file = st.file_uploader("Upload your sales data file (Excel or CSV)", type=['xlsx', 'xls', 'csv'])
        if uploaded_file:
            st.session_state.raw_df = load_data(uploaded_file)
            if st.session_state.raw_df is not None:
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")

    if not st.session_state.data_loaded:
        st.info("👋 Welcome! Please upload a sales data file to get started.")
        # st.image("https://i.imgur.com/uFLyk3z.png", caption="Use the sidebar on the left to upload your file.")
        return
        
    st.header("Step 1: Your Data at a Glance")
    st.dataframe(st.session_state.raw_df.head())
    st.markdown("---")

    st.header("Step 2: Check & Prepare Your Data")
    with st.container(border=True):
        st.subheader("Does your data have the right columns?")
        st.markdown("For this tool to work, your file needs these columns. The names must match exactly.")
        
        required_cols = {
            'Invoice': 'The unique ID for each transaction (e.g., 536365).',
            'StockCode': 'The unique ID for each product (e.g., 85123A).',
            'Description': 'The name of the product (e.g., WHITE HANGING HEART T-LIGHT HOLDER).',
            'Quantity': 'How many items were bought in the transaction (e.g., 6).',
            'InvoiceDate': 'The date and time of the sale (e.g., 12/1/2010 8:26).',
            'Price': 'The price of a single item (e.g., 2.55).',
            'Customer ID': 'The unique ID for each customer (e.g., 17850).',
            'Country': 'The country where the sale was made (e.g., United Kingdom).'
        }
        
        cols_in_data = st.session_state.raw_df.columns
        all_cols_present = True
        
        for col, desc in required_cols.items():
            if col in cols_in_data:
                st.markdown(f"✅ **{col}**: *{desc}*")
            else:
                st.markdown(f"❌ **{col}**: *{desc}* --- **MISSING!**")
                all_cols_present = False
        
        if not all_cols_present:
            st.error("Your file is missing one or more required columns. Please check your file and upload it again.")
            return

    if st.button("Prepare My Data", type="primary"):
        st.session_state.df_cleaned = preprocess_pipeline(st.session_state.raw_df)
        st.success("Your data is clean and ready for analysis!")
        st.dataframe(st.session_state.df_cleaned.head())
    
    st.markdown("---")

    if st.session_state.df_cleaned is None:
        st.warning("Please click the 'Prepare My Data' button to continue.")
        return

    # --- MAIN ANALYSIS TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Business Overview", "👥 Understand Your Customers", "🔮 Future Predictions & Pricing"])

    with tab1:
        st.header("A Big-Picture Look at Your Business")
        st.markdown("Click the buttons below to explore different parts of your business.")

        if st.button("Analyze Sales Performance (When?)"):
            with st.spinner("Analyzing sales trends..."):
                st.plotly_chart(plot_monthly_sales(st.session_state.df_cleaned), use_container_width=True)
                st.plotly_chart(plot_daily_sales(st.session_state.df_cleaned), use_container_width=True)
                st.plotly_chart(plot_hourly_sales(st.session_state.df_cleaned), use_container_width=True)

        if st.button("Analyze Performers (Best & Worst)"):
            with st.spinner("Analyzing performers..."):
                st.plotly_chart(plot_top_products(st.session_state.df_cleaned), use_container_width=True)
                st.plotly_chart(plot_geographical_sales(st.session_state.df_cleaned), use_container_width=True)
                st.markdown("---")
                st.plotly_chart(plot_worst_performers(st.session_state.df_cleaned), use_container_width=True)
        
        if st.button("Analyze Customer & Product Behavior (How & Why?)"):
            with st.spinner("Analyzing customer and product behavior..."):
                st.plotly_chart(plot_new_vs_returning_customers(st.session_state.df_cleaned), use_container_width=True)
                st.plotly_chart(plot_average_order_value(st.session_state.df_cleaned), use_container_width=True)
                st.plotly_chart(analyze_market_basket(st.session_state.df_cleaned), use_container_width=True)

        st.markdown("---")
        
        if st.button("Generate Overall Business Insights Summary", type="primary"):
            with st.spinner("Analyzing your business..."):
                display_eda_insights(st.session_state.df_cleaned)
        
        st.markdown("---")
        
        st.header("Drill-Down: Product-Specific Analysis")
        product_list = st.session_state.df_cleaned['Description'].unique()
        selected_product = st.selectbox("Select a Product to Analyze", product_list, key="product_eda")
        
        if st.button("Analyze Selected Product"):
            with st.spinner(f"Analyzing {selected_product}..."):
                product_df = st.session_state.df_cleaned[st.session_state.df_cleaned['Description'] == selected_product]
                
                st.subheader(f"Performance for: {selected_product}")
                total_sales = product_df['Revenue'].sum()
                units_sold = product_df['Quantity'].sum()
                avg_price = product_df['Price'].mean()
                num_countries = product_df['Country'].nunique()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Sales", f"${total_sales:,.2f}")
                col2.metric("Total Units Sold", f"{units_sold:,}")
                col3.metric("Average Price", f"${avg_price:,.2f}")
                col4.metric("Sold in # Countries", num_countries)

                st.plotly_chart(plot_monthly_sales(product_df, title_prefix=f"{selected_product}: "), use_container_width=True)
                st.plotly_chart(plot_daily_sales(product_df, title_prefix=f"{selected_product}: "), use_container_width=True)
                st.plotly_chart(plot_geographical_sales(product_df, title_prefix=f"{selected_product}: "), use_container_width=True)
                
                display_eda_insights(product_df, title_prefix=f"for {selected_product}")

    with tab2:
        st.header("Get to Know Your Customers Better")
        segment_tab1, segment_tab2 = st.tabs(["Simple Customer Groups", "Smart Customer Groups (AI-Powered)"])

        with segment_tab1:
            st.subheader("Group Customers by Their Buying Habits")
            st.markdown("This method groups customers based on how **Recently** they bought, how **Often** they buy, and how much **Money** they spend.")
            if st.button("Group My Customers"):
                with st.spinner("Grouping your customers..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    rfm_segmented = segment_customers(rfm_metrics)
                    rfm_final = assign_business_actions(rfm_segmented)
                    
                    st.subheader("Segment Summary")
                    summary_df = generate_business_summary(rfm_final)
                    st.dataframe(summary_df)
                    st.download_button(
                        label="Download Simple Segment Summary (CSV)",
                        data=summary_df.to_csv(index=False).encode('utf-8'),
                        file_name='simple_customer_segments_summary.csv',
                        mime='text/csv',
                    )

                    st.subheader("Your Data with Segments Added")
                    df_with_segments = merge_data_with_segments(st.session_state.df_cleaned, rfm_final, 'Segment')
                    st.dataframe(df_with_segments.head())
                    st.download_button(
                        label="Download Full Data with Simple Segments (CSV)",
                        data=df_with_segments.to_csv(index=False).encode('utf-8'),
                        file_name='full_data_simple_segments.csv',
                        mime='text/csv',
                    )

                    st.plotly_chart(plot_rfm_pie_charts(rfm_final), use_container_width=True)
                    st.plotly_chart(plot_rfm_sales_by_segment(rfm_final), use_container_width=True)
                    st.plotly_chart(plot_rfm_distribution(rfm_final), use_container_width=True)

                    display_rfm_insights(rfm_final)

                    actions = rfm_final[['Segment', 'Action']].drop_duplicates().set_index('Segment')
                    st.subheader("Recommended Actions for Each Group")
                    st.table(actions)

        with segment_tab2:
            st.subheader("Let AI Find the Best Customer Groups")
            st.markdown("This advanced method uses AI to look at your customer data and find the most natural groupings.")
            if st.button("Help Me Find the Best Number of Groups"):
                with st.spinner("Calculating..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    fig = find_optimal_clusters(rfm_metrics)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("Look for the 'elbow' in the chart above – the point where the line starts to flatten out. That's usually the best number of groups to choose.")

            k_clusters = st.slider("How many customer groups should we create?", 2, 10, 4, key='kmeans_k')
            if st.button("Create Smart Groups"):
                with st.spinner("The AI is thinking..."):
                    rfm_metrics = calculate_rfm_metrics(st.session_state.df_cleaned)
                    kmeans_clustered = perform_kmeans_clustering(rfm_metrics, k_clusters)
                    
                    cluster_names = get_cluster_names(kmeans_clustered)
                    kmeans_clustered['Cluster_Name'] = kmeans_clustered['Cluster'].map(cluster_names)
                    
                    st.subheader("Smart Group Summary")
                    kmeans_summary_df = generate_kmeans_summary_table(kmeans_clustered)
                    st.dataframe(kmeans_summary_df)
                    st.download_button(
                        label="Download Smart Segment Summary (CSV)",
                        data=kmeans_summary_df.to_csv(index=False).encode('utf-8'),
                        file_name='smart_customer_segments_summary.csv',
                        mime='text/csv',
                    )
                    
                    st.subheader("Your Data with Segments Added")
                    df_with_segments_kmeans = merge_data_with_segments(st.session_state.df_cleaned, kmeans_clustered, 'Cluster_Name')
                    st.dataframe(df_with_segments_kmeans.head())
                    st.download_button(
                        label="Download Full Data with Smart Segments (CSV)",
                        data=df_with_segments_kmeans.to_csv(index=False).encode('utf-8'),
                        file_name='full_data_smart_segments.csv',
                        mime='text/csv',
                    )


                    pie_fig = plot_kmeans_pie_charts(kmeans_clustered)
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                    sales_bar_fig = plot_kmeans_sales_by_segment(kmeans_clustered)
                    st.plotly_chart(sales_bar_fig, use_container_width=True)
                    
                    bar_fig = plot_kmeans_bar_charts(kmeans_clustered)
                    st.plotly_chart(bar_fig, use_container_width=True)
                    
                    display_kmeans_business_insights(kmeans_clustered, cluster_names)

    with tab3:
        st.header("🔮 Predict Future Sales")
        
        product_list = st.session_state.df_cleaned['StockCode'].unique()
        selected_product = st.selectbox("Select a Product ID to Predict Sales For", product_list)
        
        forecast_days = st.number_input("How many days do you want to predict into the future?", min_value=7, max_value=90, value=30, step=7)
        
        st.markdown("**(Optional) Upload these files to make predictions even better:**")
        st.info("""
        - **Competitor Prices File:** Must have a `InvoiceDate` column and price columns like `our_price`, `competitor_A`, etc.
        - **Customer Segments File:** Must have `InvoiceDate`, `Segment`, and `Quantity` columns.
        """)
        col1_upload, col2_upload = st.columns(2)
        with col1_upload:
            competitor_file = st.file_uploader("Upload Competitor Prices (CSV)", type=['csv'], key="competitor")
        with col2_upload:
            segment_file = st.file_uploader("Upload Customer Group Sales (CSV)", type=['csv'], key="segment")

        competitor_data = load_data(competitor_file) if competitor_file else None
        segment_data = load_data(segment_file) if segment_file else None

        col1_model, col2_model = st.columns(2)
        with col1_model:
            if st.button("Predict with Model 1"):
                run_forecasting_pipeline(
                    model_type='LSTM',
                    df=st.session_state.df_cleaned,
                    product_stock_code=selected_product,
                    future_forecast_days=forecast_days,
                    competitor_df=competitor_data,
                    customer_segment_df=segment_data
                )
        with col2_model:
            if st.button("Predict with Model 2"):
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
            st.header("💰 Smart Price Suggestions")
            if 'our_price' in st.session_state.daily_sales_df.columns:
                st.subheader("Find the Best Price to Maximize Sales")
                pricing_forecast_days = st.number_input(
                    "How many days to test prices for?",
                    min_value=1, max_value=90, value=7, step=1, key="pricing_days"
                )

                if st.button("Recommend the Best Price", type="primary"):
                    with st.spinner(f"Testing different prices over {pricing_forecast_days} days..."):
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
                        
                        if price_results_df is not None:
                            price_fig = plot_price_recommendation(price_results_df, optimal_row, pricing_forecast_days)
                            st.plotly_chart(price_fig, use_container_width=True)
                            
                            current_price = st.session_state.daily_sales_df['our_price'].iloc[-1]
                            display_pricing_insights(optimal_row, current_price, pricing_forecast_days, price_results_df)
            else:
                st.warning("Dynamic pricing requires price data. The 'our_price' column could not be found.")


if __name__ == '__main__':
    main()

