import torch
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. TARGET FUNCTION & DATA
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

def relu(z):
    """ReLU: max(0, z)"""
    return torch.maximum(torch.zeros_like(z), z)

def relu_derivative(z):
    """ReLU türevi: z > 0 ise 1, değilse 0"""
    return (z > 0).float()

def sigmoid(z):
    """
    Sigmoid: 1 / (1 + e^(-z))
    Çıkışı (0, 1) aralığına sıkıştırır.
    """
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_derivative(z):
    """
    Sigmoid türevi: sigmoid(z) * (1 - sigmoid(z))
    """
    s = sigmoid(z)
    return s * (1.0 - s)

def softmax(z):
    """
    Softmax: exp(z_i) / sum(exp(z_j))
    Çok sınıflı olasılık dağılımı üretir.
    Numerik stabilite için z - max(z) çıkarılır.
    """
    z_stable = z - torch.max(z, dim=1, keepdim=True).values
    exp_z = torch.exp(z_stable)
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)

# =============================================================================
# 3. MANUAL LOSS FUNCTIONS
# =============================================================================
def mse_loss(y_true, y_pred):
    """
    MSE Loss = (1/N) * sum((y_true - y_pred)^2)
    dL/dy_pred = -(2/N) * (y_true - y_pred)
    """
    n = y_true.shape[0]
    loss = torch.mean((y_true - y_pred) ** 2)
    dloss_dy_pred = -(2.0 / n) * (y_true - y_pred)
    return loss, dloss_dy_pred

def bce_loss(y_true, y_pred, eps=1e-7):
    """
    Binary Cross Entropy (Sigmoid çıkışı için)
    L = -(1/N) * sum[y*log(p) + (1-y)*log(1-p)]
    dL/dp = -(1/N) * [y/p - (1-y)/(1-p)]
    """
    n = y_true.shape[0]
    p = torch.clamp(y_pred, eps, 1 - eps)
    loss = -torch.mean(y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p))
    dloss_dp = -(1.0 / n) * (y_true / p - (1 - y_true) / (1 - p))
    return loss, dloss_dp

def cross_entropy_loss(y_true, y_pred, eps=1e-7):
    """
    Cross Entropy (Softmax çıkışı için)
    y_true: one-hot encoded (n, classes)
    y_pred: softmax probabilities (n, classes)
    
    Softmax + CE birlikte kullanıldığında geri yayılım basitleşir:
    dL/dz = (1/N) * (y_pred - y_true)   (z: softmax öncesi logitler)
    """
    n = y_true.shape[0]
    p = torch.clamp(y_pred, eps, 1.0)
    loss = -torch.mean(torch.sum(y_true * torch.log(p), dim=1))
    dloss_dz = (1.0 / n) * (p - y_true)
    return loss, dloss_dz

# =============================================================================
# 4. MANUAL NEURAL NETWORK
# =============================================================================
class ManualNN:
    """
    3 katmanlı tam bağlı yapay sinir ağı.
    Aktivasyon fonksiyonu seçilebilir: 'tanh', 'relu', 'sigmoid'
    """
    
    def __init__(self, input_size, hidden_size, output_size, activation='tanh', seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.activation = activation
        
        # Xavier (Glorot) başlatımı
        self.W1 = torch.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = torch.zeros(1, hidden_size)
        
        self.W2 = torch.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = torch.zeros(1, hidden_size)
        
        self.W3 = torch.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b3 = torch.zeros(1, output_size)
    
    def activate(self, z):
        """Seçilen aktivasyon fonksiyonunu uygular"""
        if self.activation == 'tanh':
            return tanh(z)
        elif self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        else:
            raise ValueError("activation must be 'tanh', 'relu', or 'sigmoid'")
    
    def activate_derivative(self, z):
        """Seçilen aktivasyon fonksiyonunun türevini uygular"""
        if self.activation == 'tanh':
            return tanh_derivative(z)
        elif self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        else:
            raise ValueError("activation must be 'tanh', 'relu', or 'sigmoid'")
    
    def forward(self, X):
        """
        İLERİ YAYILIM
        =============
        Katman 1:  z1 = X @ W1 + b1     -> a1 = activate(z1)
        Katman 2:  z2 = a1 @ W2 + b2    -> a2 = activate(z2)
        Çıkış:     z3 = a2 @ W3 + b3    -> y_pred = z3  (lineer)
        """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activate(self.z1)
        
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.activate(self.z2)
        
        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_pred = self.z3
        
        return self.y_pred
    
    def backward(self, X, dloss_dy_pred):
        """
        GERİ YAYILIM - Zincir Kuralı
        =============================
        
        Çıkış Katmanı (lineer):
            dW3 = a2.T @ dz3
            db3 = sum(dz3)
            da2 = dz3 @ W3.T
        
        Gizli Katman 2:
            dz2 = da2 * activate'(z2)   [Hadamard çarpımı]
            dW2 = a1.T @ dz2
            db2 = sum(dz2)
            da1 = dz2 @ W2.T
        
        Gizli Katman 1:
            dz1 = da1 * activate'(z1)   [Hadamard çarpımı]
            dW1 = X.T @ dz1
            db1 = sum(dz1)
        """
        # Çıkış Katmanı
        dz3 = dloss_dy_pred
        self.dW3 = self.a2.T @ dz3
        self.db3 = torch.sum(dz3, dim=0, keepdim=True)
        da2 = dz3 @ self.W3.T
        
        # Gizli Katman 2
        dz2 = da2 * self.activate_derivative(self.z2)
        self.dW2 = self.a1.T @ dz2
        self.db2 = torch.sum(dz2, dim=0, keepdim=True)
        da1 = dz2 @ self.W2.T
        
        # Gizli Katman 1
        dz1 = da1 * self.activate_derivative(self.z1)
        self.dW1 = X.T @ dz1
        self.db1 = torch.sum(dz1, dim=0, keepdim=True)
    
    def update_weights(self, lr):
        """Gradyan İnişi: W = W - lr * dW"""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3
    
    def train(self, X, Y, epochs=1000, lr=0.01, verbose=True):
        """Eğitim döngüsü"""
        loss_history = []
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss, dloss_dy_pred = mse_loss(Y, y_pred)
            loss_history.append(loss.item())
            
            self.backward(X, dloss_dy_pred)
            self.update_weights(lr)
            
            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")
        
        return loss_history
    
    def predict(self, X):
        """Tahmin yapar"""
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred.squeeze().numpy()

# =============================================================================
# 5. TRAIN & EVALUATE
# =============================================================================
def train_model(model, X, Y, epochs=1000, lr=0.01):
    print(f"\n{'='*50}")
    print(f"Training: {model.activation.upper()} | {model.W1.shape[1]} neurons")
    print(f"{'='*50}")
    history = model.train(X, Y, epochs=epochs, lr=lr)
    return model, history[-1]

def evaluate_model(model, X):
    return model.predict(X)

# =============================================================================
# 6. MAIN EXPERIMENT
# =============================================================================
if __name__ == "__main__":
    X_train, Y_train, x_np, y_np = create_data(n_samples=300)
    
    activations = ['tanh', 'relu', 'sigmoid']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, act in enumerate(activations):
        model = ManualNN(
            input_size=1, 
            hidden_size=16, 
            output_size=1,
            activation=act,
            seed=42
        )
        model, final_loss = train_model(model, X_train, Y_train, epochs=1000, lr=0.01)
        predictions = evaluate_model(model, X_train)
        
        axes[idx].plot(x_np, y_np, 'b-', linewidth=2, label='Target', alpha=0.8)
        axes[idx].plot(x_np, predictions, 'r--', linewidth=2, 
                       label=f'{act.upper()} (16 neurons)', alpha=0.8)
        axes[idx].set_title(f'Activation: {act.upper()}\nFinal MSE: {final_loss:.6f}', fontsize=11)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('f(x)')
    
    plt.suptitle('Manual Backpropagation: Tanh vs ReLU vs Sigmoid\n'
                 '(Forward Pass + Chain Rule + Gradient Descent)', fontsize=13)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("EK NOTLAR:")
    print("-" * 60)
    print("Sigmoid ve Softmax fonksiyonları ile BCE ve Cross Entropy")
    print("loss fonksiyonları kodda tanımlıdır.")
    print()
    print("Sigmoid: Gizli katmanlarda veya ikili sınıflandırma")
    print("         çıkış katmanında kullanılabilir.")
    print()
    print("Softmax: Çok sınıflı sınıflandırma problemlerinde çıkış")
    print("         katmanında kullanılır. CE loss ile birlikte")
    print("         kullanıldığında geri yayılım basitleşir:")
    print("         dL/dz = (1/N) * (softmax(z) - y_true)")
    print("="*60)
