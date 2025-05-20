import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net import gtnet

# Параметры
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes = 10
in_dim = 5
seq_length = 8
out_dim = 2
total_time = 100
batch_size = 16
epochs = 20
learning_rate = 0.001

# СИНТЕТИЧЕСКИЕ ДАННЫЕ
np.random.seed(42)
data = np.random.randn(num_nodes, in_dim, total_time).astype(np.float32)  # (10, 5, 100)

def create_dataset(data, seq_len, pred_len):
    x_list, y_list = [], []
    for t in range(data.shape[2] - seq_len - pred_len + 1):
        x = data[:, :, t:t+seq_len]
        y = data[:, :, t+seq_len:t+seq_len+pred_len]
        x_list.append(x.transpose(1, 0, 2))  # -> (in_dim, num_nodes, seq_len)
        y_list.append(y.transpose(1, 0, 2))  # -> (in_dim, num_nodes, pred_len)

    # Преобразуем списки в numpy.ndarray перед созданием тензоров
    return torch.from_numpy(np.array(x_list)), torch.from_numpy(np.array(y_list))

X, Y = create_dataset(data, seq_length, out_dim)
print("Dataset shape:", X.shape, Y.shape)


train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]


train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = gtnet(
    gcn_true=True,
    buildA_true=True,
    gcn_depth=2,
    num_nodes=num_nodes,
    device=device,
    dropout=0.3,
    subgraph_size=5,
    node_dim=10,
    dilation_exponential=2,
    conv_channels=32,
    residual_channels=32,
    skip_channels=64,
    end_channels=128,
    seq_length=seq_length,
    in_dim=in_dim,
    out_dim=out_dim,
    layers=3
).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        out = model(batch_x)  # [B, out_dim, num_nodes, 1]

        # Подготовка target
        target = batch_y.permute(0, 3, 2, 1)  # [B, out_dim, num_nodes, in_dim]
        target = target[..., 0]               # Берём только первый признак для сравнения

        loss = loss_fn(out.squeeze(-1), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_x)
            target = batch_y.permute(0, 3, 2, 1)[..., 0]
            loss = loss_fn(out.squeeze(-1), target)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {total_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(test_loader):.4f}")
