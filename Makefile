install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	python -c "import torch; import mlflow; import sklearn; print('✅ Todas las librerias OK')"

lint:
	pip install flake8 && flake8 src/ --max-line-length=120 --ignore=E501,W503 || true

all: install train