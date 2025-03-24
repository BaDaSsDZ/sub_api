import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Fetch Data
def fetch_data():
    conn = psycopg2.connect(
        dbname="neondb",
        user="neondb_owner",
        password="npg_D3ydx6RuzZLo",
        host="ep-late-base-a5udr15v-pooler.us-east-2.aws.neon.tech",
        port="5432"
    )
    query = "SELECT * FROM analytics_events ORDER BY session_id, timestamp"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    df['member_data'] = df['member_data'].apply(lambda x: x if x is not None else {})
    df['additional_data'] = df['additional_data'].apply(lambda x: x if x is not None else {})
    df['product_data'] = df['product_data'].apply(lambda x: x if x is not None else {})

    # Group by session_id
    sessions = df.groupby('session_id')
    features = []
    labels = []
    max_length = 0

    for session_id, group in sessions:
        group = group.sort_values('timestamp')
        steps = group['step'].fillna(0).values
        events = group['event_name'].values
        time_spent = group['additional_data'].apply(lambda x: x.get('timeSpentOnStep', 0)).values
        interactions = group['additional_data'].apply(lambda x: x.get('interactionCount', 0)).values
        session_time = group['additional_data'].apply(lambda x: x.get('sessionTimeSpent', 0)).values

        le = LabelEncoder()
        event_encoded = le.fit_transform(events)
        session_features = np.column_stack((steps, event_encoded, time_spent, interactions, session_time))
        features.append(session_features)
        max_length = max(max_length, len(session_features))
        label = 1 if 'subscription_created' in group['event_name'].values else 0
        labels.append(label)

    padded_features = []
    for seq in features:
        if len(seq) < max_length:
            padded = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant', constant_values=0)
        else:
            padded = seq
        padded_features.append(padded)

    X = np.array(padded_features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, max_length

# Step 3: Define Model
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

# Step 4: Dataset
class SubscriptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 5: Train Model
def train_model(X, y):
    input_size = X.shape[2]  # 5 features
    hidden_size = 64
    num_layers = 1
    dataset = SubscriptionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SubscriptionLSTM(input_size, hidden_size, num_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

# Main execution
if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_data()
    print(f"Fetched {len(df)} log entries across {len(df['session_id'].unique())} sessions")

    print("Preprocessing data...")
    X, y, max_length = preprocess_data(df)
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}, Max session length: {max_length}")

    print("Training model...")
    model = train_model(X, y)

    print("Saving model and max_length...")
    torch.save(model.state_dict(), "subscription_lstm.pth")
    with open("max_length.txt", "w") as f:
        f.write(str(max_length))

    print("Testing model...")
    model.eval()
    sample = X[0:1]
    with torch.no_grad():
        pred = model(torch.tensor(sample, dtype=torch.float32)).item()
        print(f"Predicted probability for session 0: {pred:.4f}, Actual label: {y[0]}")

    print("Done! Model saved as 'subscription_lstm.pth', max_length as 'max_length.txt'")