from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
from functools import wraps
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import bcrypt
from binance.client import Client

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# MongoDB Atlas setup
MONGODB_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_URI)
db = client.get_database('crypto_prediction')
users = db['users']

# Binance client setup
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
binance_client = Client(api_key, api_secret)

# Load the model and scaler
model = load_model('lstm_binance_model.h5')
scaler = joblib.load('scaler_binance.pkl')

def get_current_bitcoin_price():
    try:
        # Get the current Bitcoin price from Binance
        ticker = binance_client.get_symbol_ticker(symbol="BTCUSDT")
        current_price = float(ticker['price'])
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_price, last_update
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None, None

# Get historical data for prediction
klines = binance_client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2017")
data = pd.DataFrame(klines, columns=[
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
])
data['Date'] = pd.to_datetime(data['Open Time'], unit='ms')
data['Close'] = data['Close'].astype(float)
data = data[['Date', 'Close']]

# Get the last sequence and date
scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
best_timestep = 30
last_sequence = scaled_data[-best_timestep:]
last_known_date = data['Date'].iloc[-1]

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def predict_future(target_date):
    target_date = pd.to_datetime(target_date)
    days_ahead = (target_date - last_known_date).days
    
    if days_ahead <= 0:
        return None, "Target date must be in the future!"

    current_input = last_sequence.copy()
    prediction = None
    
    for _ in range(days_ahead):
        prediction = model.predict(current_input.reshape(1, best_timestep, 1))
        current_input = np.append(current_input[1:], prediction, axis=0)

    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    return predicted_price, None

@app.route('/')
def landing():
    current_price, last_update = get_current_bitcoin_price()
    if current_price is None:
        current_price = 0
        last_update = "Failed to fetch price"
    return render_template('landing.html', current_price=current_price, last_update=last_update)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if users.find_one({'email': email}):
            flash('Email already exists')
            return redirect(url_for('signup'))
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create user
        users.insert_one({
            'email': email,
            'password': hashed_password
        })
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = users.find_one({'email': email})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user'] = email
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction = None
    error = None
    input_date = None
    current_price, last_update = get_current_bitcoin_price()
    
    if request.method == 'POST':
        input_date = request.form['date']
        try:
            prediction, error = predict_future(input_date)
        except Exception as e:
            error = str(e)
    
    return render_template('dashboard.html', 
                         prediction=prediction, 
                         error=error, 
                         input_date=input_date,
                         last_known_date=last_known_date.strftime('%Y-%m-%d'),
                         user=session['user'],
                         current_price=current_price,
                         last_update=last_update)

if __name__ == '__main__':
    app.run(debug=True)
