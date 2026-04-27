import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

matplotlib.use('kitcat')
# PyTorch ile Evrensel Fonksiyon Yakınsaması

# Hedef fonksiyon
def hedef_fonksiyon(x_np):
    return np.sin(x_np) * np.cos(2*x_np) + 0.5 * np.sin(5*x_np)

# Veri oluşturma
x_np = np.linspace(-np.pi, np.pi, 300).astype(np.float32)
y_np = hedef_fonksiyon(x_np).astype(np.float32)

X_train = torch.FloatTensor(x_np).unsqueeze(1)  # (300, 1)
y_train = torch.FloatTensor(y_np).unsqueeze(1)  # (300, 1)

# Farklı kapasiteli modeller
def model_olustur(n_noron):
    return nn.Sequential(
        nn.Linear(1, n_noron),
        nn.Tanh(),
        nn.Linear(n_noron, n_noron),
        nn.Tanh(),
        nn.Linear(n_noron, 1)
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, n_noron in enumerate([4, 16, 64]):
    torch.manual_seed(42)
    model = model_olustur(n_noron)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Eğitim döngüsü
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tahmin = model(X_train).squeeze().numpy()

    axes[idx].plot(x_np, y_np, 'b-', linewidth=2, label='Hedef Fonksiyon', alpha=0.8)
    axes[idx].plot(x_np, tahmin, 'r--', linewidth=2, label=f'ANN ({n_noron} nöron/katman)', alpha=0.8)
    axes[idx].set_title(f'{n_noron} Nöron - Loss: {loss.item():.4f}', fontsize=11)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Evrensel Fonksiyon Yakınsaması - PyTorch ile Nöron Sayısının Etkisi', fontsize=13)
plt.tight_layout()
plt.show()
