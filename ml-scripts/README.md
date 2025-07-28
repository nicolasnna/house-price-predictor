# ML Scripts - House Price Predictor

Este directorio contiene los scripts y recursos para el entrenamiento, evaluación y manejo de modelos de machine learning para el proyecto *House Price Predictor*.


## 📁 Estructura de carpetas

```bash
ml-scripts/
├── data/
│ ├── raw/ # Datasets originales sin modificar
│ ├── processed/ # Datos limpios y transformados listos para usar
│ └── external/ # Datos externos o adicionales
│
├── notebooks/ # Notebooks para exploración y experimentación
│
├── models/ # Modelos entrenados guardados (*.joblib, *.pkl, *.json)
│
├── scripts/ # Scripts para entrenamiento, evaluación y preprocesamiento
│
├── utils/ # Funciones reutilizables y utilitarias
│
├── pyproject.toml # Archivo de configuración de Poetry
├── poetry.lock # Archivo que asegura versiones exactas de dependencias
├── README.md # Este archivo
└── .gitignore # Archivos y carpetas ignoradas por git
```


## ⚙️ Uso

Para instalar las dependencias:

```bash
poetry install
```

Luego puedes ejecutar los scripts desde el entorno activado, por ejemplo:

```bash
poetry run python scripts/train_model.py --dataset data/processed/california.csv --output models/modelo.joblib
```