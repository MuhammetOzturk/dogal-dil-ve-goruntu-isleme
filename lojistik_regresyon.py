import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader

X_moon,y_moon = make_moons(n_samples=300,noise=0.2,random_state=42)

scaler = StandardScaler()
X_moon = scaler.fit_transform(X_moon)

plt.scatter(X_moon[:,0],X_moon[:,1],c=y_moon, cmap='coolwarm',edgecolor='k',alpha=0.7,s=100)
plt.grid(True,linestyle='--',linewidth=2,alpha=0.5)
plt.xlabel('x1')
plt.ylabel('y1')
plt.show()

X_tensor = torch.FloatTensor(X_moon)
y_tensor = torch.FloatTensor(y_moon).unsqueeze(1) #(300,1)

dataset = TensorDataset(X_tensor,y_tensor)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

def plot_decision_boundary(model, X, y, title, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

    # Izgara oluştur
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing='ij'
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)

    # Model tahminleri
    model.eval()
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape).numpy()

    # Kontur çizimi
    ax.contourf(xx.numpy(), yy.numpy(), preds, levels=50, cmap='coolwarm', alpha=0.4)
    ax.contour(xx.numpy(), yy.numpy(), preds, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.5)

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Model, kayıp fonksiyonu ve optimizer
model = LogisticRegressionModel()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 500
loss_history_log = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_history_log.append(epoch_loss / len(dataloader))

print(f"Lojistik Regresyon - Son Epoch Kayıp: {loss_history_log[-1]:.4f}")


