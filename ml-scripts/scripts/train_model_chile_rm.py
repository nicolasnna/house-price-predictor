import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

RANDOM_STATE = 30

def filter_prices(X: pd.DataFrame, y: pd.Series, min_price: float = 1e5):
    mask = y > min_price
    return X[mask], y[mask]
  
def make_pipeline(model):
  preprocessor = SimpleImputer()  # aquí podrías añadir StandardScaler, OneHotEncoder, etc.
  return Pipeline([
      ("preprocessor", preprocessor),
      ("model", model)
  ])
    
def evaluate_model(pipe: Pipeline, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mape = mean_absolute_percentage_error(y_val, preds)
    precision = 100 - (mape * 100)
    return {"mae": mae, "precision": precision}

def get_obj_results(model, X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 42):
    X, y = filter_prices(X, y, min_price=1e5)
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipe = make_pipeline(
      model
    )
    results = evaluate_model(pipe, train_X, train_y, val_X, val_y)

    cv_mae = -cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    ).mean()
    results["cross_mae"] = cv_mae

    cv_mape = -cross_val_score(
        pipe, X, y,
        cv=cv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1
    ).mean()
    results["cross_precision"] = 100 - (cv_mape * 100)

    return results
  
if __name__ == "__main__":
  data_jul_23 = pd.read_csv('../data/processed/kaggle - Casas Chile RM jul-23 - proccessed.csv')
  data_mar_23 = pd.read_csv('../data/processed/kaggle - Casas Chile RM mar-23 - proccessed.csv')

  y_jul = data_jul_23['Price']
  X_jul = data_jul_23.drop(columns=['Price', 'TotalArea'])
  y_mar = data_mar_23['Price']
  X_mar = data_mar_23.drop(columns=['Price', 'TotalArea'])
  
  forest_model = RandomForestRegressor(random_state=RANDOM_STATE)
  xgb_model = XGBRegressor(
    random_state=RANDOM_STATE,
    n_estimators=1000,
    learning_rate=0.12,
    n_jobs=-1
  )

  results_jul = get_obj_results(forest_model, X_jul, y_jul)
  print('----- Data set Jul 2023 -----')
  print(f" Random forest regressor: \n{results_jul}")
  results_jul = get_obj_results(xgb_model, X_jul, y_jul)
  print(f" xgb model: \n{results_jul}")
  
  results_mar = get_obj_results(forest_model, X_mar, y_mar)
  print('\n----- Data set Mar 2023 -----')
  print(f" Random forest regressor: \n{results_mar}")
  results_mar = get_obj_results(xgb_model, X_mar, y_mar)
  print(f" xgb model: \n{results_mar}")
  
  X_union = pd.concat([X_jul, X_mar], ignore_index=True)
  y_union = pd.concat([y_jul, y_mar], ignore_index=True)
  results_union = get_obj_results(forest_model, X_union, y_union)
  print('\n----- Data set Mar-Jul 2023 -----')
  print(f" Random forest regressor: \n{results_union}")
  results_union = get_obj_results(xgb_model, X_union, y_union)
  print(f" xgb model: \n{results_union}")

