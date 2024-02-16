import pandas as pd
from flask import Flask, request, render_template
import pickle
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
import mpld3
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Bitcoin_Price_Updated.csv")

# load the model
with open('bitcoin_prediction.pkl', 'rb') as file:
    model = pickle.load(file)



# create a flask application
app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    # read the file contents and send them to client

    return render_template('index.html')


@app.route("/classify", methods=["POST"])
def classify():

    number_of_days = int(request.form.get("Number of days"))

    if number_of_days == 1:
        tomorrow_date = datetime.now() + timedelta(days=1)
        date_range = pd.date_range(start=tomorrow_date, periods=number_of_days, freq='D').normalize()
        future_df = pd.DataFrame(index=date_range)
        future_df["Close"] = 0
        x = model.predict(start=len(df) + 1, end=len(df) + number_of_days, dynamic=True)
        x = pd.DataFrame(x)
        future_df["Close"] = x.values[0][0]
        future_df["Close"] = future_df["Close"].round(2)
        future_df = future_df.rename(columns= {"Close": "USD ($)"})

    else:
        tomorrow_date = datetime.now() + timedelta(days=1)
        date_range = pd.date_range(start=tomorrow_date, periods=number_of_days, freq='D').normalize()
        future_df = pd.DataFrame(index=date_range)
        future_df["Close"] = 0
        x = model.predict(start = len(df)+1, end = len(df)+number_of_days, dynamic= True)
        x = pd.DataFrame(x)
        future_df["Close"] = x["predicted_mean"].values
        future_df["Close"] = future_df["Close"].round(2)
        future_df = future_df.rename(columns = {"Close": "USD ($)"})

    return render_template('values.html',tables=[future_df.to_html(classes='data')], titles=future_df.columns)





# start the application
app.run(host="0.0.0.0", port=8000, debug=True)
