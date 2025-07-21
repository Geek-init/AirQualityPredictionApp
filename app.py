from shiny import App, ui, render, reactive
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle

from a6_ex4 import PollutionRegressor

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("data_file", "Upload CSV"),
        ui.output_ui("select_pollutants_ui"),
        ui.output_ui("smooth_slider_ui"),
        ui.input_checkbox("enable_forecast", "Show PM2.5 Forecast"),
        ui.output_ui("scaler_upload_ui"),
        ui.output_ui("weights_upload_ui"),
    ),
    ui.output_plot("main_plot"),
    title="Air Quality Explorer"
)

def server(input, output, session):
    @reactive.Calc
    def uploaded_data():
        file_info = input.data_file()
        if file_info is None:
            return None
        df = pd.read_csv(Path(file_info[0]["datapath"]))
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        return df

    @output
    @render.ui
    def select_pollutants_ui():
        if uploaded_data() is None:
            return "Please upload your dataset first."
        columns = uploaded_data().columns
        available = [col for col in columns if col not in ["date", "year", "month", "day", "No", "station"]]
        return ui.input_selectize(
            "pollutant_choices",
            "Choose Variables",
            choices=available,
            selected=["PM2.5"],
            multiple=True
        )

    @output
    @render.ui
    def smooth_slider_ui():
        if uploaded_data() is None:
            return None
        return ui.input_slider("smooth_window", "Rolling Window (days)", 1, 30, 7)

    @output
    @render.ui
    def scaler_upload_ui():
        if input.enable_forecast():
            return ui.input_file("scaler_file", "Upload StandardScaler (.pkl)")
        return None

    @output
    @render.ui
    def weights_upload_ui():
        if input.enable_forecast():
            return ui.input_file("weights_file", "Upload Model Weights (.pt)")
        return None

    @output
    @render.plot
    def main_plot():
        df = uploaded_data()
        if df is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Upload a CSV file to display the plot.",
                    ha="center", va="center")
            return fig

        selected_vars = input.pollutant_choices()
        window_size = input.smooth_window()

        fig, ax = plt.subplots(figsize=(12, 6))

        for var in selected_vars:
            daily = df.groupby("date")[var].mean().rolling(window=window_size).mean()
            ax.plot(daily.index, daily, label=var)

        if input.enable_forecast() and input.scaler_file() and input.weights_file():
            scaler_meta = input.scaler_file()[0]
            scaler = pickle.load(open(Path(scaler_meta["datapath"]), "rb"))

            predictors = df.select_dtypes(include=["float64", "int64"]).drop(columns=["PM2.5"])
            X_scaled = scaler.transform(predictors)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            input_size = X_tensor.shape[1]
            model = PollutionRegressor(input_size)
            weights_meta = input.weights_file()[0]
            model.load_state_dict(torch.load(Path(weights_meta["datapath"])))
            model.eval()

            with torch.no_grad():
                predictions = model(X_tensor).numpy().flatten()

            df["Forecast"] = predictions
            forecast_daily = df.groupby("date")["Forecast"].mean().rolling(window=window_size).mean()
            ax.plot(forecast_daily.index, forecast_daily, label="Predicted PM2.5")

        ax.set_title("Air Pollution Trends")
        ax.legend()
        return fig

app = App(app_ui, server)
