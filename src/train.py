import os
import sys
import yaml
import json
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.transform import resize
import joblib
import kagglehub

print("✅ Librerías cargadas correctamente")

# ============================================
# 1. CARGAR CONFIGURACION
# ============================================
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("✅ Configuracion cargada")
print(f"   Dataset: {config['dataset']['kaggle_id']}")
print(f"   Modelo: {config['model']['classifier']}")

# ============================================
# 2. CONFIGURAR KAGGLE Y DESCARGAR DATASET
# ============================================
kaggle_json_path = Path(__file__).parent.parent / "kaggle.json"
if kaggle_json_path.exists():
    with open(kaggle_json_path, "r") as f:
        kaggle_creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = kaggle_creds["username"]
    os.environ["KAGGLE_KEY"] = kaggle_creds["key"]
    print("✅ Credenciales Kaggle configuradas")
else:
    print("⚠️  kaggle.json no encontrado, usando variables de entorno")

print("\n📥 Descargando dataset de Kaggle...")
dataset_path = kagglehub.dataset_download(config['dataset']['kaggle_id'])
print(f"✅ Dataset descargado en: {dataset_path}")

# ============================================
# 3. CARGAR IMAGENES
# ============================================
print("\n🖼️  Cargando imagenes...")

def extract_hog_features(img):
    """Extrae features HOG de una imagen"""
    img_resized = resize(img, (128, 128), anti_aliasing=True)
    features, _ = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1
    )
    return features
def load_images_from_folder(base_path, max_per_class):
    images = []
    labels = []
    base = Path(base_path)
    
    # Buscar carpetas hoja con imágenes (evitar duplicados)
    class_dirs_dict = {}
    for item in sorted(base.rglob("*")):
        if item.is_dir() and any(item.iterdir()):
            # Verificar que es una carpeta hoja (sin subcarpetas con imágenes)
            subdirs_with_images = [d for d in item.iterdir() if d.is_dir() and any(d.glob("*.jpg"))]
            if not subdirs_with_images:
                img_files = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.bmp"))
                if len(img_files) >= 10:
                    class_name = item.name
                    if class_name not in class_dirs_dict:
                        class_dirs_dict[class_name] = img_files[:max_per_class]
    
    class_dirs = list(class_dirs_dict.keys())
    print(f"   Clases encontradas: {class_dirs}")
    
    for class_name in class_dirs:
        img_files = class_dirs_dict[class_name]
        count = 0
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                features = extract_hog_features(img_array)
                images.append(features)
                labels.append(class_name)
                count += 1
            except Exception:
                continue
        print(f"   {class_name}: {count} imagenes cargadas")
    
    return np.array(images), labels

X, labels = load_images_from_folder(
    dataset_path,
    config['dataset']['max_samples_per_class']
)

print(f"✅ Total imagenes: {len(X)}")
print(f"✅ Features por imagen: {X.shape[1]}")
print(f"✅ Clases: {list(set(labels))}")

# ============================================
# 4. PREPARAR DATOS
# ============================================
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['model']['test_size'],
    random_state=config['model']['random_state'],
    stratify=y
)

print(f"\n✅ Datos divididos:")
print(f"   Entrenamiento: {X_train.shape[0]} muestras")
print(f"   Prueba: {X_test.shape[0]} muestras")

# ============================================
# 5. ENTRENAR MODELO
# ============================================
print("\n🤖 Entrenando RandomForest...")

model = RandomForestClassifier(
    n_estimators=config['model']['n_estimators'],
    random_state=config['model']['random_state'],
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Modelo entrenado")

# ============================================
# 6. EVALUAR MODELO
# ============================================
print("\n📊 Evaluando modelo...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1 Score (weighted): {f1:.4f}")
print(f"   F1 Score (macro):    {f1_macro:.4f}")
print("\n📋 Reporte detallado:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================
# 7. REGISTRAR EN MLFLOW
# ============================================
print("\n📝 Registrando en MLflow...")

workspace_dir = Path(__file__).parent.parent
mlruns_dir = workspace_dir / "mlruns"
mlruns_dir.mkdir(exist_ok=True)

tracking_uri = "file://" + str(mlruns_dir.resolve()).replace("\\", "/")
if not tracking_uri.startswith("/"):
    tracking_uri = "/" + tracking_uri
tracking_uri = "file://" + tracking_uri

mlflow.set_tracking_uri(tracking_uri)

experiment_name = config['mlflow']['experiment_name']
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

input_example = X_test[:3]

with mlflow.start_run(experiment_id=experiment_id,
                      run_name=config['mlflow']['run_name']) as run:
    mlflow.log_param("model_type", config['model']['classifier'])
    mlflow.log_param("feature_extractor", "HOG")
    mlflow.log_param("n_estimators", config['model']['n_estimators'])
    mlflow.log_param("test_size", config['model']['test_size'])
    mlflow.log_param("max_samples_per_class", config['dataset']['max_samples_per_class'])
    mlflow.log_param("num_classes", len(le.classes_))
    mlflow.log_param("classes", str(list(le.classes_)))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("train_samples", X_train.shape[0])
    mlflow.log_metric("test_samples", X_test.shape[0])

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    joblib.dump(model, workspace_dir / "model.pkl")
    joblib.dump(le, workspace_dir / "label_encoder.pkl")

    print(f"✅ Experimento: {experiment_name}")
    print(f"✅ Run ID: {run.info.run_id}")

print("\n🎉 Pipeline completado exitosamente!")
print(f"   Accuracy final: {accuracy*100:.2f}%")
print(f"   F1 Score: {f1:.4f}")