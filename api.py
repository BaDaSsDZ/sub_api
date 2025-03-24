from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np
from pydantic import BaseModel
import psycopg2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os  # Add this import

app = FastAPI()

# Load max_length
with open("max_length.txt", "r") as f:
    max_length = int(f.read())

# LSTM Model
class SubscriptionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SubscriptionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Load model
model = SubscriptionLSTM(input_size=5, hidden_size=64, num_layers=1)
model.load_state_dict(torch.load("subscription_lstm.pth", map_location=torch.device('cpu')))
model.eval()

# Pydantic model for incoming data
class EventLog(BaseModel):
    session_id: str
    event_name: str
    step: int | None
    member_data: dict | None
    product_data: dict | None
    additional_data: dict | None

# Fetch session data from DB
def get_session_data(session_id: str):
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    query = "SELECT * FROM analytics WHERE session_id = %s ORDER BY timestamp"
    df = pd.read_sql(query, conn, params=(session_id,))
    conn.close()
    return df

# Preprocess session for prediction
def preprocess_session(df, current_event: EventLog):
    df['member_data'] = df['member_data'].apply(lambda x: x if x is not None else {})
    df['additional_data'] = df['additional_data'].apply(lambda x: x if x is not None else {})
    df['product_data'] = df['product_data'].apply(lambda x: x if x is not None else {})
    
    # Add the current event to the session history
    new_row = pd.DataFrame([{
        'session_id': current_event.session_id,
        'event_name': current_event.event_name,
        'step': current_event.step or 0,
        'timestamp': pd.Timestamp.now(),
        'member_data': current_event.member_data or {},
        'product_data': current_event.product_data or {},
        'additional_data': current_event.additional_data or {}
    }])
    df = pd.concat([df, new_row], ignore_index=True).sort_values('timestamp')
    
    steps = df['step'].fillna(0).values
    events = df['event_name'].values
    time_spent = df['additional_data'].apply(lambda x: x.get('timeSpentOnStep', 0)).values
    interactions = df['additional_data'].apply(lambda x: x.get('interactionCount', 0)).values
    session_time = df['additional_data'].apply(lambda x: x.get('sessionTimeSpent', 0)).values
    
    le = LabelEncoder()
    event_encoded = le.fit_transform(events)
    features = np.column_stack((steps, event_encoded, time_spent, interactions, session_time))
    
    if len(features) < max_length:
        features = np.pad(features, ((0, max_length - len(features)), (0, 0)), 'constant', constant_values=0)
    else:
        features = features[-max_length:]  # Take last max_length events
    
    return features

@app.post("/predict")
async def predict(event: EventLog):
    df = get_session_data(event.session_id)
    features = preprocess_session(df, event)
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, max_length, 5]
    with torch.no_grad():
        prob = model(X).item()
    return {"probability": prob}

@app.get("/")
async def root():
    return {"status": "ok"}