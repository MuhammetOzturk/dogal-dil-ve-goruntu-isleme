import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. TARGET FUNCTION & DATA GENERATION
# =============================================================================
def target_function(x):
    """Hedef fonksiyon: sin(x) * cos(2x) + 0.5 * sin(5x)"""
    return np.sin(x) * np.cos(2 * x) + 0.5 * np.sin(5 * x)

def create_data(n_samples=300):
    """Eğitim verisini oluşturur"""
    x = np.linspace(-np.pi, np.pi, n_samples).astype(np.float32)
    y = target_function(x).astype(np.float32)
    X = torch.FloatTensor(x).unsqueeze(1)      # (300, 1)
    Y = torch.FloatTensor(y).unsqueeze(1)      # (300, 1)
    return X, Y, x, y

# =============================================================================
# 2. MANUAL ACTIVATION FUNCTIONS
# =============================================================================
def tanh(z):
    """Tanh aktivasyonu"""
    return torch.tanh(z)

def tanh_derivative(z):
    """Tanh türevi: 1 - tanh(z)^2"""
    return 1.0 - torch.tanh(z) ** 2

# =============================================================================
# 3. MANUAL LOSS FUNCTION (MSE)
# =============================================================================
def compute_mse_loss(y_true, y_pred):
    """
    MSE Loss = (1/N) * sum((y_true - y_pred)^2)
    
    Loss'un y_pred'e göre türevi:
    dL/dy_pred = -(2/N) * (y_true - y_pred)
    """
    n = y_true.shape[0]
    loss = torch.mean((y_true - y_pred) ** 2)
    dloss_dy_pred = -(2.0 / n) * (y_true - y_pred)
    return loss, dloss_dy_pred

# =============================================================================
# 4. MANUAL NEURAL NETWORK CLASS
# =============================================================================
class ManualNN:
    """
    3 katmanlı tam bağlı (fully connected) yapay sinir ağı.
    Tüm matematiksel işlemler manuel olarak yapılır.
    """
    
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # --- Ağırlık Başlatımı (Xavier/Glorot) ---
        # W1: (input_size  -> hidden_size)
        self.W1 = torch.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = torch.zeros(1, hidden_size)
        
        # W2: (hidden_size -> hidden_size)
        self.W2 = torch.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = torch.zeros(1, hidden_size)
        
        # W3: (hidden_size -> output_size)
        self.W3 = torch.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b3 = torch.zeros(1, output_size)
    
    def forward(self, X):
        """
        İLERİ YAYILIM (Forward Pass)
        ============================
        Katman 1:  z1 = X @ W1 + b1     -> a1 = tanh(z1)
        Katman 2:  z2 = a1 @ W2 + b2    -> a2 = tanh(z2)
        Çıkış:     z3 = a2 @ W3 + b3    -> y_pred = z3  (lineer)
        """
        # Katman 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = tanh(self.z1)
        
        # Katman 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = tanh(self.z2)
        
        # Çıkış Katmanı (lineer aktivasyon - regresyon)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_pred = self.z3
        
        return self.y_pred
    
    def backward(self, X, dloss_dy_pred):
        """
        GERİ YAYILIM (Backward Pass) - Zincir Kuralı
        =============================================
        
        Adım adım gradyanları hesaplarız:
        
        1. Çıkış Katmanı:
           dL/dW3 = a2.T @ dL/dz3
           dL/db3 = sum(dL/dz3, axis=0)
           dL/da2 = dL/dz3 @ W3.T
        
        2. Gizli Katman 2:
           dL/dz2 = dL/da2 * tanh'(z2)   [Hadamard çarpımı]
           dL/dW2 = a1.T @ dL/dz2
           dL/db2 = sum(dL/dz2, axis=0)
           dL/da1 = dL/dz2 @ W2.T
        
        3. Gizli Katman 1:
           dL/dz1 = dL/da1 * tanh'(z1)   [Hadamard çarpımı]
           dL/dW1 = X.T @ dL/dz1
           dL/db1 = sum(dL/dz1, axis=0)
        """
        n_samples = X.shape[0]
        
        # --- Çıkış Katmanı ---
        # z3 = a2 @ W3 + b3  (lineer olduğu için dz3 = dy_pred)
        dz3 = dloss_dy_pred                              # (n, 1)
        self.dW3 = self.a2.T @ dz3                       # (hidden, 1)
        self.db3 = torch.sum(dz3, dim=0, keepdim=True)   # (1, 1)
        da2 = dz3 @ self.W3.T                            # (n, hidden)
        
        # --- Gizli Katman 2 ---
        # z2 = a1 @ W2 + b2,  a2 = tanh(z2)
        dz2 = da2 * tanh_derivative(self.z2)             # (n, hidden)
        self.dW2 = self.a1.T @ dz2                       # (hidden, hidden)
        self.db2 = torch.sum(dz2, dim=0, keepdim=True)   # (1, hidden)
        da1 = dz2 @ self.W2.T                            # (n, hidden)
        
        # --- Gizli Katman 1 ---
        # z1 = X @ W1 + b1,  a1 = tanh(z1)
        dz1 = da1 * tanh_derivative(self.z1)             # (n, hidden)
        self.dW1 = X.T @ dz1                             # (1, hidden)
        self.db1 = torch.sum(dz1, dim=0, keepdim=True)   # (1, hidden)
    
    def update_weights(self, lr):
        """
        AĞIRLIK GÜNCELLEME (Gradient Descent)
        =====================================
        W = W - learning_rate * dW
        b = b - learning_rate * db
        """
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3
    
    def train(self, X, Y, epochs=1000, lr=0.01, verbose=True):
        """
        EĞİTİM DÖNGÜSÜ
        ==============
        Her epoch için:
          1. Forward pass  -> tahmin
          2. Loss hesapla  -> hata ve başlangıç gradyanı
          3. Backward pass -> zincir kuralı ile tüm gradyanlar
          4. Update        -> ağırlıkları güncelle
        """
        loss_history = []
        
        for epoch in range(epochs):
            # 1. İleri Yayılım
            y_pred = self.forward(X)
            
            # 2. Loss ve başlangıç gradyanı (dL/dy_pred)
            loss, dloss_dy_pred = compute_mse_loss(Y, y_pred)
            loss_history.append(loss.item())
            
            # 3. Geri Yayılım (Zincir Kuralı)
            self.backward(X, dloss_dy_pred)
            
            # 4. Ağırlık Güncelleme
            self.update_weights(lr)
            
            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")
        
        return loss_history
    
    def predict(self, X):
        """Eğitim sonrası tahmin yapar"""
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred.squeeze().numpy()

# =============================================================================
# 5. TRAINING & EVALUATION FUNCTIONS
# =============================================================================
def train_model(model, X, Y, epochs=1000, lr=0.01):
    """Modeli eğitir ve son loss'u döndürür"""
    print(f"\n{'='*50}")
    print(f"Training Manual NN ({model.W1.shape[1]} hidden neurons)")
    print(f"{'='*50}")
    history = model.train(X, Y, epochs=epochs, lr=lr)
    final_loss = history[-1]
    return model, final_loss, history

def evaluate_model(model, X):
    """Model ile tahmin yapar"""
    predictions = model.predict(X)
    return predictions

# =============================================================================
# 6. MAIN EXPERIMENT & VISUALIZATION
# =============================================================================
if __name__ == "__main__":
    # Veriyi oluştur
    X_train, Y_train, x_np, y_np = create_data(n_samples=300)
    
    # 3 farklı model kapasitesi ile deney
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    neuron_counts = [4, 16, 64]
    
    for idx, n_neurons in enumerate(neuron_counts):
        # Model oluştur
        model = ManualNN(input_size=1, hidden_size=n_neurons, output_size=1, seed=42)
        
        # Eğit (tüm matematik manuel!)
        model, final_loss, history = train_model(model, X_train, Y_train, epochs=1000, lr=0.01)
        
        # Değerlendir
        predictions = evaluate_model(model, X_train)
        
        # Görselleştir
        axes[idx].plot(x_np, y_np, 'b-', linewidth=2, label='Target Function', alpha=0.8)
        axes[idx].plot(x_np, predictions, 'r--', linewidth=2, 
                       label=f'Manual NN ({n_neurons} neurons)', alpha=0.8)
        axes[idx].set_title(f'{n_neurons} Neurons\nFinal MSE: {final_loss:.6f}', fontsize=11)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('f(x)')
    
    plt.suptitle('Universal Function Approximation with Manual Backpropagation\n'
                 '(Forward Pass + Chain Rule + Gradient Descent)', fontsize=13)
    plt.tight_layout()
    plt.show()
