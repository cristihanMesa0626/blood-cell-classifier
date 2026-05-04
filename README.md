# 🔬 Blood Cell Leukemia Classifier

Pipeline de Machine Learning para clasificación de células sanguíneas 
en frotis de leucemia usando Transfer Learning con MobileNetV2.

## 📋 Dataset
- **Fuente:** Kaggle - [Leukemia ALL Dataset](https://www.kaggle.com/datasets/mehradaria/leukemia)
- **Clases:** Benign, Early, Pre, Pro
- **Técnica:** Transfer Learning con MobileNetV2 + RandomForest

## 🚀 Ejecución local

### 1. Clonar el repositorio
```bash
git clone https://github.com/cristihanMesa0626/blood-cell-classifier.git
cd blood-cell-classifier
```

### 2. Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias
```bash
make install
```

### 4. Configurar Kaggle
Colocar el archivo `kaggle.json` en `~/.kaggle/kaggle.json`

### 5. Ejecutar pipeline
```bash
make train
```

### 6. Ver resultados en MLflow
```bash
mlflow ui --backend-store-uri mlruns
```
Abrir: http://127.0.0.1:5000

## 📊 Métricas registradas
- Accuracy
- F1 Score (weighted)
- F1 Score (macro)

## 🔧 Tecnologías
- Python 3.13
- MLflow 2.19.0
- PyTorch + MobileNetV2
- scikit-learn RandomForest
- GitHub Actions CI/CD
