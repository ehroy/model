import json
import os
import pickle
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR, SVC, NuSVC
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY


binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)


def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:,['open','high','low','close']].dropna()
    df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def train_model(timeframe):
    if MODEL == "LinearRegression":
            # Load the price data
        price_data = pd.read_csv(training_price_data_path)
        df = load_frame(price_data, timeframe)

        print(df.tail())

        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        model = LinearRegression()
    elif MODEL == "SVR":
            # Load the price data
        price_data = pd.read_csv(training_price_data_path)
        df = load_frame(price_data, timeframe)

        print(df.tail())

        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        model = SVR()
    elif MODEL == "SVC":
        price_data = pd.read_csv(training_price_data_path)
        df = load_frame(price_data, timeframe)
        y_train = df['close'].shift(-1).dropna().values
        X_train = df.iloc[:-1].values
        y_train = pd.cut(y_train, bins=3, labels=[0, 1, 2])
        model = SVC()
    elif MODEL == "KernelRidge":
            # Load the price data
        price_data = pd.read_csv(training_price_data_path)
        df = load_frame(price_data, timeframe)

        print(df.tail())

        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
            # Load the price data
        price_data = pd.read_csv(training_price_data_path)
        df = load_frame(price_data, timeframe)

        print(df.tail())

        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        model = BayesianRidge()
    elif MODEL == "NuSVC":
        price_data = pd.read_csv(training_price_data_path)

        # Mengambil frame data berdasarkan time frame
        df = load_frame(price_data, timeframe)

        # Menggeser 'close' untuk prediksi target dan menghapus NaN
        y_train = df['close'].shift(-1).dropna().values

        # Menghapus baris terakhir di X_train agar jumlah baris sesuai dengan y_train
        X_train = df.iloc[:-1].values

        # Memastikan tidak ada nilai NaN di X_train atau y_train
        # Jika masih ada NaN, hilangkan baris yang mengandung NaN
        X_train = X_train[~np.isnan(y_train)]
        y_train = y_train[~np.isnan(y_train)]

        # Konversi y_train menjadi kategori diskrit (misalkan 3 kategori)
        y_train = pd.cut(y_train, bins=3, labels=[0, 1, 2])
        model = NuSVC()
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    model.fit(X_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")


def get_inference(token, timeframe, region, data_provider):
    """Load model and predict current price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)
    
    print(X_new.tail())
    print(X_new.shape)

    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0]
