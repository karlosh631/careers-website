import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA



df = pd.read_csv(r"C:\Users\thapa\Downloads\attendance-report-2025-07-14-to-2025-07-18.csv")

# Drop name and ID columns, keep only date columns
date_columns = df.columns.drop(['user_id', 'user_full_name'])

# Sum attendance per date (each date column has 1 for present, 0 for absent)
attendance_ts = df[date_columns].sum(axis=0).to_frame(name='Total_Present')
attendance_ts.index = pd.to_datetime(attendance_ts.index)

# Plotting 
attendance_ts.plot(figsize=(10, 5), marker='o', title='Attendance Over Time')
plt.ylabel("Number of Present Employees")
plt.grid(True)
plt.tight_layout()
plt.show()

#ARIMA
series = attendance_ts['Total_Present']

# Fit ARIMA model
model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 3 days
forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)
forecast = [series[-1]] * forecast_steps  # naive forecast: repeat last value


# Forecast index
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Plot actual + forecast
plt.figure(figsize=(10, 5))
plt.plot(series, label="Actual Attendance", marker='o')
plt.plot(forecast_index, forecast, label="Forecast", marker='x', linestyle='--', color='red')
plt.title("ARIMA Attendance Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Present Employees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print forecast values
print("Forecasted Attendance for Next 3 Days:")
for date, value in zip(forecast_index, forecast):
    print(f"{date.date()}: {value}")