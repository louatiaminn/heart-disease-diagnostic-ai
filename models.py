import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

SEED = 42


def get_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=SEED)


def get_xgboost():
    return xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=SEED
    )


class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(DeepMLP, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.pipeline(x).squeeze(-1)


def train_pytorch(X_train, y_train, epochs=80, lr=0.01):
    model = DeepMLP(input_dim=X_train.shape[1], hidden_dim=32, output_dim=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def predict_pytorch(model, X_test):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
    return preds, probs

