import json

with open('../Derin_Ogrenme_Temelleri.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def img_cell(path, caption=""):
    text = f"<p align=\"center\">\n  <img src=\"{path}\" alt=\"{caption}\" width=\"80%\">\n</p>\n\n*{caption}*"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

# 1. AI-ML-DL hiyerarşisi -> Hücre 1 (Temel Kavramlar) sonrası
# Hücre 1'in sonunda "### Hiyerarşik İlişki" var. Hücre 1'den hemen sonra ekleyelim.
cells.insert(2, img_cell('images/ai_ml_dl_hierarchy.png', 'Yapay Zeka, Makine Öğrenmesi ve Derin Öğrenme arasındaki hiyerarşik ilişki.'))

# 2. Feature Engineering vs Deep Learning -> Hücre 3 (ML vs DL karşılaştırması) sonrası
# Şimdi cells[3] eski cells[2] olmuş durumda. Eski indeks 2 şimdi 3.
cells.insert(4, img_cell('images/feature_engineering_vs_deep_learning.png', 'Geleneksel makine öğrenmesinde insan uzmanı tarafından özellik çıkarımı yapılırken, derin öğrenmede model ham veriden otomatik olarak özellikleri öğrenir.'))

# 3. Gradyan İnişi -> Adım 4 markdown hücresinden önce
# Adım 4 markdown hücresini bulalım. Metin içinde "### Adım 4: Eğitim Döngüsü" geçiyor.
idx_step4 = None
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown' and 'Adım 4: Eğitim Döngüsü' in ''.join(cell.get('source', [])):
        idx_step4 = i
        break
if idx_step4 is not None:
    cells.insert(idx_step4, img_cell('images/gradient_descent.png', 'Gradyan inişi algoritması, kayıp fonksiyonunun en düşük olduğu noktaya doğru iteratif olarak ilerler.'))

# 4. Sigmoid Fonksiyonu -> 5.1 Lojistik Regresyon Nedir? markdown hücresinden sonra
idx_logistic = None
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown' and '5.1. Lojistik Regresyon Nedir?' in ''.join(cell.get('source', [])):
        idx_logistic = i
        break
if idx_logistic is not None:
    cells.insert(idx_logistic + 1, img_cell('images/sigmoid_function.png', 'Sigmoid fonksiyonu, herhangi bir gerçek sayıyı 0 ile 1 arasına sıkıştırarak olasılık tahmini yapmayı sağlar.'))

# 5. Sinir Ağı Mimarisi -> 5.2 Basit Sinir Ağı (MLP) Nedir? markdown hücresinden sonra
idx_mlp = None
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown' and '5.2. Basit Sinir Ağı (MLP) Nedir?' in ''.join(cell.get('source', [])):
        idx_mlp = i
        break
if idx_mlp is not None:
    cells.insert(idx_mlp + 1, img_cell('images/neural_network_architecture.png', 'Çok Katmanlı Algılayıcı (MLP) mimarisi: girdi katmanı, gizli katman (ReLU) ve çıktı katmanından (Sigmoid) oluşur.'))

with open('../Derin_Ogrenme_Temelleri.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Görseller notebook'a başarıyla eklendi!")
