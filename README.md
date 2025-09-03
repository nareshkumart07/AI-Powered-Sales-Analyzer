**AI-Powered Market Trend Forecasting & Dynamic Pricing System For Revenue Optimization**

This is a comprehensive, multi-page Streamlit dashboard designed to provide deep insights into retail sales data. It empowers users to clean and analyze their sales history, understand customer behavior through advanced segmentation, predict future product sales using deep learning models, and receive AI-driven recommendations for optimal pricing.

**Features**
This tool is broken down into three main analysis tabs, each offering a suite of features:

**📊 Business Overview (EDA)**
Sales Performance Analysis: Visualize sales trends over time (monthly, daily, hourly).

Performer Analysis: Instantly identify best-selling products and top-performing countries, as well as underperforming products and regions that may need attention.

Customer & Product Behavior: Analyze the value of new vs. returning customers, track the average order value over time, and discover which products are frequently bought together with a market basket analysis.

Product-Specific Deep Dive: Select any product to get a detailed breakdown of its specific sales trends and geographical performance.

**👥 Understand Your Customers**
Simple Customer Groups (RFM): Segments customers based on Recency, Frequency, and Monetary value, providing clear, actionable insights for each group.

AI-Powered Smart Groups (K-Means): Uses machine learning to find natural clusters in your customer base, revealing non-obvious customer personas.

Downloadable Data: Download both the segment summaries and your full transactional data enriched with the new customer segment labels for use in targeted marketing campaigns.

**🔮 Future Predictions & Pricing**
AI Sales Forecasting: Utilizes advanced deep learning models (LSTM and GRU) to predict future daily sales for any product.

Forecast Breakdown: Includes easy-to-understand visuals like a pie chart of predicted busy days and a bar chart comparing past vs. future sales.

Smart Price Suggestions: After running a forecast, this feature simulates different price points to recommend the optimal price for maximizing future revenue.

**Project Structure**

The application is organized into several modules for clarity and maintainability:

**app.py**: The main file that runs the Streamlit application and controls the UI layout.

**data_processing.py**: Handles loading data (from a file or a database) and the entire cleaning and preprocessing pipeline.

**eda.py**: Contains all functions for the "Business Overview" tab.

**customer_segmentation.py**: Includes all logic for both RFM and K-Means segmentation.

**forecasting.py**: Contains the deep learning models (LSTM, GRU) and the forecasting pipeline.

**dynamic_pricing.py**: Houses the logic for price simulation and revenue optimization.

**utils.py**: A utility file for shared resources, such as the color palette for plots.

**requirements.txt**: Lists all the necessary Python libraries for the project.

**Setup and Installation**
Follow these steps to get the application running on your local machine.

**Prerequisites**

Python 3.8 or newer

Git

**Installation**

Clone the repository:

git clone <your-repository-url>

cd <your-repository-folder>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install -r requirements.txt

How to Run the Application
Once the installation is complete, you can run the app with a single command:

streamlit run app.py

Your web browser should open with the application running locally at http://localhost:8501.

**Deployment**

This application is ready for deployment on the Streamlit Community Cloud.

Upload to GitHub: Push all the project files (.py files and requirements.txt) to a public GitHub repository.

Connect to Streamlit Cloud: Sign up or log in to share.streamlit.io with your GitHub account.

Deploy: Click "New app", select your repository, and ensure the "Main file path" is set to app.py. Click "Deploy!".

The platform will automatically install the dependencies from requirements.txt and host your application on a shareable URL.
