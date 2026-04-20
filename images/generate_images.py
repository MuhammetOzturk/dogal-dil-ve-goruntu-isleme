import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. AI-ML-DL Hiyerarşisi
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# AI Dairesi (en dış)
ai = Circle((5, 5), 4.5, color='#E8F4FD', ec='#2E86AB', linewidth=3)
ax.add_patch(ai)
ax.text(5, 8.8, 'Yapay Zeka (AI)', fontsize=16, ha='center', va='center', fontweight='bold', color='#1A5276')

# ML Dairesi (orta)
ml = Circle((5, 4.5), 3.2, color='#D4EFDF', ec='#28B463', linewidth=3)
ax.add_patch(ml)
ax.text(5, 7.2, 'Makine Öğrenmesi (ML)', fontsize=14, ha='center', va='center', fontweight='bold', color='#1E8449')

# DL Dairesi (iç)
dl = Circle((5, 3.8), 1.8, color='#FADBD8', ec='#E74C3C', linewidth=3)
ax.add_patch(dl)
ax.text(5, 3.8, 'Derin Öğrenme\n(DL)', fontsize=12, ha='center', va='center', fontweight='bold', color='#C0392B')

# Örnekler
ax.text(1.5, 6.5, '• Satranç botu\n• Sesli asistan', fontsize=10, color='#1A5276')
ax.text(7.5, 5.5, '• Spam filtreleme\n• Ev fiyat tahmini', fontsize=10, color='#1E8449')
ax.text(5, 1.5, '• Yüz tanıma\n• ChatGPT', fontsize=10, ha='center', color='#C0392B')

plt.title('Yapay Zeka - Makine Öğrenmesi - Derin Öğrenme Hiyerarşisi', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('ai_ml_dl_hierarchy.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Feature Engineering vs Deep Learning
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sol panel: Traditional ML
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Geleneksel Makine Öğrenmesi', fontsize=14, fontweight='bold', color='#154360')

boxes = [
    (2, 8, 'Ham Veri', '#D6EAF8'),
    (2, 6, 'İnsan Uzmanı\n(Özellik Çıkarımı)', '#AED6F1'),
    (2, 4, 'Özellikler\n(Features)', '#85C1E9'),
    (2, 2, 'ML Modeli\n(SVM, RF)', '#5DADE2'),
]
for x, y, text, color in boxes:
    box = FancyBboxPatch((x-1.5, y-0.8), 3, 1.6, boxstyle="round,pad=0.1", facecolor=color, edgecolor='#154360', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#154360')
    if y > 2:
        ax.annotate('', xy=(x, y-0.9), xytext=(x, y-1.7), arrowprops=dict(arrowstyle='->', color='#154360', lw=2))

ax.text(5, 0.5, 'Elle tasarlanmış özellikler gerekir', fontsize=11, ha='center', style='italic', color='#154360')

# Sağ panel: Deep Learning
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Derin Öğrenme', fontsize=14, fontweight='bold', color='#922B21')

boxes_dl = [
    (2, 8, 'Ham Veri\n(Görüntü, Ses, Metin)', '#FADBD8'),
    (2, 5.5, 'Derin Sinir Ağı\n(Otomatik Özellik Öğrenme)', '#F5B7B1'),
    (2, 3, 'Çıktı\n(Tahmin / Sınıf)', '#F1948A'),
]
for x, y, text, color in boxes_dl:
    h = 1.8 if 'Sinir' in text else 1.4
    box = FancyBboxPatch((x-1.5, y-h/2), 3, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor='#922B21', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#922B21')
    if y > 3:
        ax.annotate('', xy=(x, y-1.2), xytext=(x, y-2.2), arrowprops=dict(arrowstyle='->', color='#922B21', lw=2))

ax.text(5, 0.5, 'Model verinin ham halinden kendi özelliklerini öğrenir', fontsize=11, ha='center', style='italic', color='#922B21')

plt.suptitle('Özellik Çıkarımı: Geleneksel ML vs Derin Öğrenme', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_engineering_vs_deep_learning.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Yapay Sinir Ağı Mimarisi
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

layers = [
    (1.5, ['x₁', 'x₂', 'x₃'], 'Girdi Katmanı'),
    (5.0, ['h₁', 'h₂', 'h₃', 'h₄'], 'Gizli Katman\n(ReLU)'),
    (8.5, ['ŷ'], 'Çıktı Katmanı\n(Sigmoid)'),
]

neurons = {}
for lx, labels, title in layers:
    n = len(labels)
    start_y = 5 + (n-1)*0.8
    for i, label in enumerate(labels):
        y = start_y - i*1.6
        circle = Circle((lx, y), 0.4, color='#AED6F1', ec='#154360', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(lx, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='#154360', zorder=4)
        neurons[(lx, y)] = label
    ax.text(lx, 8.8, title, ha='center', va='center', fontsize=12, fontweight='bold', color='#154360')

# Bağlantılar
for (x1, y1), label1 in neurons.items():
    for (x2, y2), label2 in neurons.items():
        if abs(x2 - x1 - 3.5) < 0.1:
            ax.plot([x1+0.4, x2-0.4], [y1, y2], color='#85929E', linewidth=1, alpha=0.6, zorder=1)
        elif abs(x2 - x1 - 3.5) < 0.1 and 'h' in label1 and 'ŷ' in label2:
            ax.plot([x1+0.4, x2-0.4], [y1, y2], color='#85929E', linewidth=1, alpha=0.6, zorder=1)

plt.title('Basit Bir Yapay Sinir Ağı (MLP) Mimarisi', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('neural_network_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 4. Sigmoid Fonksiyonu
fig, ax = plt.subplots(figsize=(8, 5))
z = np.linspace(-10, 10, 400)
sigma = 1 / (1 + np.exp(-z))
ax.plot(z, sigma, color='#E74C3C', linewidth=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('z', fontsize=12)
ax.set_ylabel('σ(z)', fontsize=12)
ax.set_title('Sigmoid Aktivasyon Fonksiyonu', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('sigmoid_function.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# 5. Gradyan İnişi (3D surface + contour)
fig = plt.figure(figsize=(12, 5))

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
# Gradyan iniş yolu
theta_x = [1.5, 1.2, 0.9, 0.6, 0.3, 0.0]
theta_y = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
theta_z = [tx**2 + ty**2 for tx, ty in zip(theta_x, theta_y)]
ax1.plot(theta_x, theta_y, theta_z, color='red', marker='o', markersize=6, linewidth=2, label='Gradyan İnişi')
ax1.set_xlabel('w₁')
ax1.set_ylabel('w₂')
ax1.set_zlabel('Loss')
ax1.set_title('3D Kayıp Yüzeyi')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=15, cmap='viridis')
ax2.plot(theta_x, theta_y, color='red', marker='o', markersize=6, linewidth=2)
ax2.scatter([0], [0], color='gold', s=100, zorder=5, marker='*', label='Global Minimum')
ax2.set_xlabel('w₁')
ax2.set_ylabel('w₂')
ax2.set_title('Kontur Grafiği')
ax2.legend()
plt.suptitle('Gradyan İnişi (Gradient Descent) Görselleştirmesi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Tüm görseller başarıyla oluşturuldu!")
