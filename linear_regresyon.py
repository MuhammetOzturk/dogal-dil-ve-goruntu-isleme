import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(n_samples=100, w_true=2.0, b_true=1.0, noise_std=2.0, device="cpu"):
    """Sentetik lineer regresyon verisi üretir."""
    X = torch.linspace(0, 10, n_samples, device=device).unsqueeze(1)
    noise = torch.randn(n_samples, 1, device=device) * noise_std
    y = w_true * X + b_true + noise
    line = w_true * X + b_true
    return X, y, line


def plot_data(ax, X, y, plot_type="scatter", label=None, title=None, **kwargs):
    """Veriyi verilen eksende (ax) çizer."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if plot_type == "scatter":
        ax.scatter(X, y, label=label, **kwargs)
    elif plot_type == "plot":
        ax.plot(X, y, label=label, **kwargs)

    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, X):
        return self.linear(X)


def train_model(model, X, y, loss_fn, optimizer, epochs=200):
    """Modeli eğitir ve kayıp geçmişi ile anlık görüntüleri döndürür."""
    loss_history = []
    snapshots = {}  # epoch -> prediction

    for epoch in range(epochs):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Anlık görüntüler için seçili epochları sakla
        if epoch in {0, 10, 50, 100, epochs - 1}:
            #detach turevlenebilir tensoru normal tensore donusturur. 
            snapshots[epoch] = y_pred.detach().clone()

    return loss_history, snapshots


def plot_snapshots(X, y, line, snapshots, w_true, b_true):
    """Eğitim anlık görüntülerini yan yana çizer."""
    snapshot_epochs = sorted(snapshots.keys())
    n = len(snapshot_epochs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, epoch in zip(axes, snapshot_epochs):
        plot_data(ax, X, y, plot_type="scatter", label="Gerçek Veri", alpha=0.7,color="blue", s=20)
        plot_data(ax, X, line, plot_type="plot", label=f"Doğru: W={w_true}, b={b_true}", color="green", linewidth=2)
        plot_data(ax, X, snapshots[epoch], plot_type="plot", label=f"Tahmin (Epoch {epoch})", color="red", linewidth=2)
        ax.set_title(f"Epoch {epoch}")

    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history):
    """Kayıp geçmişini çizer."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="purple")
    plt.title("Kayıp (Loss) Geçmişi")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch sürümü: {torch.__version__}")
    print(f"Kullanılan cihaz: {device}")

    # Sabitler
    torch.manual_seed(42)
    w_true = 2.0
    b_true = 1.0
    n_samples = 100
    epochs = 200
    lr = 1e-3

    # Veri hazırlığı
    X, y, line = prepare_data(n_samples=n_samples, w_true=w_true, b_true=b_true, device=device)

    # Model, kayıp fonksiyonu ve optimizasyon
    model = LinearRegressionModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"Gerçek W={w_true}, b={b_true}")
    print(f"Modelin başlangıç W={model.linear.weight.item():.5f}, b={model.linear.bias.item():.5f}")

    # Eğitim
    loss_history, snapshots = train_model(model, X, y, loss_fn, optimizer, epochs=epochs)

    # Sonuçları yazdır
    print(f"Eğitim sonrası W={model.linear.weight.item():.5f}, b={model.linear.bias.item():.5f}")

    # Görselleştirme
    plot_snapshots(X, y, line, snapshots, w_true, b_true)
    plot_loss_history(loss_history)




if __name__ == "__main__":
    main()
