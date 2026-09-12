# Predictive Analytics Using Historical Data

A complete Python + Streamlit project for forecasting future sales from historical data.

## Files
- `app.py` — interactive dashboard and prediction model
- `generate_data.py` — creates sample historical data
- `historical_sales.csv` — sample dataset
- `requirements.txt` — required packages
- `README.md` — instructions

## Run with Git Bash on Windows

```bash
python --version
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

Then open `http://localhost:8501` if the browser does not open automatically.

## Model
The project uses Linear Regression with a time-index feature. The last portion of historical data is held out for testing. Accuracy is reported using MAE, RMSE and R².

## Interview explanation
I developed a predictive analytics dashboard using Python. I processed historical sales data, created a time-based feature, trained a Linear Regression model, evaluated it using MAE, RMSE and R², visualized actual versus predicted values, and forecast future sales through an interactive Streamlit dashboard.
