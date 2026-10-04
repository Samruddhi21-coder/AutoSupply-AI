🚗 AutoSupply AI
AI-Powered Vehicle Demand Forecasting & Inventory Planning for India

AutoSupply AI is a machine-learning powered dashboard that helps estimate vehicle demand for a selected Indian state, vehicle category, month, and year.

Instead of relying on fixed numbers, the application uses historical monthly vehicle-registration data and a trained forecasting model to generate a demand estimate. It then calculates a data-driven safety stock and provides a recommended planning quantity.

Know the demand. Plan the vehicles.

✨ What Problem Does It Solve?

Automobile demand varies significantly depending on:

📍 Location / State
🚘 Vehicle category
📅 Month
🗓️ Year
📈 Historical demand patterns
🛡️ Demand uncertainty

Planning inventory without considering these factors can lead to:

Over-stocking
Under-stocking
Poor supply planning
Increased inventory costs
Missed demand

AutoSupply AI provides a simple interface where a user selects the required parameters and receives a model-based planning recommendation.

🎯 Key Features
🔮 Demand Forecasting

Select:

State
Vehicle Type
Month
Year

The application generates a demand estimate using the project's forecasting model.

The selected month and year are actual model inputs, so changing them changes the forecast rather than displaying a fixed hard-coded value.

📍 State-Aware Forecasting

The application considers the selected Indian state as part of the forecasting process.

This allows demand to differ between locations instead of assuming that every state has the same automobile demand.

🚘 Multiple Vehicle Categories

AutoSupply AI provides user-friendly vehicle groups including:

Two Wheeler
Three Wheeler
Passenger Vehicle
Passenger Transport
Goods Vehicle
Other Motor Vehicle

These groups are mapped from the underlying vehicle-registration categories.

🛡️ Safety Stock

Demand forecasts contain uncertainty.

AutoSupply AI therefore calculates a safety stock using historical demand variability.

The application uses a 90% one-sided service-level factor and applies a reasonable upper bound to prevent extreme buffers.

📦 Recommended Inventory

The core planning logic is:

Recommended Inventory
        =
Predicted Demand
        +
Safety Stock

This gives the user a practical planning quantity rather than only a raw forecast.

📊 Interactive Dashboard

The dashboard provides visual insights around the selected forecast, including demand trends and planning metrics.

The interface is designed to make forecasting understandable even for users without a machine-learning background.

📄 PDF Report

After generating a forecast, users can download a compact PDF report containing:

Location
Vehicle type
Target month/year
Forecast / actual demand
Safety stock
Recommended inventory
Inventory buffer
Planning logic

🧠 How It Works
              Historical Data
                    │
                    ▼
          Data Preparation
                    │
                    ▼
          Feature Engineering
                    │
                    ▼
        Machine Learning Model
                    │
                    ▼
          Demand Forecast
                    │
                    ▼
           Safety Stock
                    │
                    ▼
      Recommended Inventory
                    │
                    ▼
             Dashboard

The application follows this overall pipeline:

Historical Data → Time-Series Features → Trained Model → Forecast → Safety Stock → Planning Recommendation.

🤖 Machine Learning

The forecasting engine uses a Random Forest Regressor with preprocessing through a Scikit-learn pipeline.

The model works with historical vehicle-registration information and forecasting features to estimate demand for the selected combination of:

State × Vehicle Type × Month × Year
🗂️ Project Structure
AutoSupply-AI/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   └── vahan_vehicle_registrations.csv
│   │
│   ├── processed/
│   │   └── processed datasets
│   │
│   └── models/
│       └── model-related files
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_demand_forecasting.ipynb
│   ├── 06_forecast_analytics.ipynb
│   ├── 07_business_insights.ipynb
│   ├── 08_inventory_optimization.ipynb
│   ├── 09_supply_planning.ipynb
│   ├── 10_decision_insights.ipynb
│   ├── 11_dashboard_preparation.ipynb
│   ├── 12_dashboard_data.ipynb
│   └── 13_dashboard.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
🛠️ Tech Stack
Technology	Purpose
🐍 Python	Core programming
🐼 Pandas	Data manipulation
🔢 NumPy	Numerical computation
🤖 Scikit-learn	Machine learning
🌊 Streamlit	Interactive dashboard
📊 Plotly	Interactive visualizations
📄 ReportLab	PDF report generation
📓 Jupyter Notebook	Data analysis & experimentation
🌳 Random Forest	Demand forecasting
🔧 Git & GitHub	Version control

The current application dependencies include Streamlit, Pandas, NumPy, Scikit-learn and Plotly; the PDF-enabled version additionally uses ReportLab.

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Samruddhi21-coder/AutoSupply-AI.git

Then:

cd AutoSupply-AI
2️⃣ Create a Virtual Environment

A virtual environment is recommended so that the project's Python packages remain isolated from other projects.

Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate

You should see something like:

(.venv) PS F:\AutoSupply-AI>
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run AutoSupply AI

Start the Streamlit application:

python -m streamlit run app.py

The terminal will provide a local address similar to:

Local URL: http://localhost:8501

Open that address in your browser.

🖥️ Using the Application
Step 1 — Overview

The landing page explains:

What AutoSupply AI does
What problem it solves
How the forecasting system works

Click:

✨ LET'S EXPLORE
Step 2 — Select Forecast Inputs

Choose:

📍 Location / State
🚘 Vehicle Type
📅 Month
🗓️ Year
Step 3 — Generate Forecast

Click:

✨ GENERATE FORECAST

The application processes the selected parameters and generates the forecast.

Step 4 — Understand the Results

The dashboard provides metrics such as:

🎯 Predicted / Actual Demand

🛡️ Safety Stock

📦 Recommended Inventory

📊 Inventory Buffer

For historical periods, the application can distinguish between historical actual demand and a future forecast.

Step 5 — Download the Report

Generate the forecast and download the resulting PDF planning report for documentation or presentation.

📈 Example

Suppose the user selects:

State: Goa
Vehicle Type: Two Wheeler
Month: July
Year: 2029

AutoSupply AI processes the selected combination and generates:

Predicted Demand
        +
Safety Stock
        ↓
Recommended Inventory

The result is therefore specific to the selected state + vehicle + month + year, rather than being a single fixed number.

📊 Why This Project Is Different

Many dashboards simply display historical data.

AutoSupply AI attempts to go one step further:

Historical Data
       ↓
Machine Learning
       ↓
Demand Forecast
       ↓
Uncertainty Buffer
       ↓
Inventory Recommendation

So the project combines:

Data Science + Machine Learning + Forecasting + Inventory Planning + Business Intelligence + Interactive Dashboard

⚠️ Important Note

AutoSupply AI produces forecast estimates, not guaranteed future sales.

The forecast should be considered a decision-support tool and should be validated against real-world factors such as:

Market conditions
Economic changes
Manufacturer supply
Regional policies
Seasonality
Promotions
Consumer behavior
Actual inventory levels

The application's own report notes that forecasts should be validated against operational constraints before procurement decisions.

🔮 Future Improvements

Possible future versions could include:

 XGBoost / LightGBM forecasting comparison
 ARIMA / Prophet model comparison
 Automated model evaluation
 Confidence intervals
 Real-time registration data
 State-level demand heatmaps
 Dealer-level forecasting
 Supply-chain optimization
 Automated alerts for high-demand periods
 Cloud deployment
 User authentication
 API for external applications
👩‍💻 Author
Samruddhi Shrawagi

BE Engineering Student | Data Science | Machine Learning | Cybersecurity

Project: AutoSupply AI
Domain: Automobile Demand Forecasting & Inventory Planning

⭐ Project Goal

Turn historical automobile data into actionable demand and inventory insights.

AutoSupply AI is built to demonstrate how machine learning can be applied to a practical supply-chain problem and transformed into an intuitive business-facing application.

📌 Quick Start

If you already have Python installed:

git clone https://github.com/Samruddhi21-coder/AutoSupply-AI.git
cd AutoSupply-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py

Then open:

http://localhost:8501
