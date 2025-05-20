import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
# ベースライン（過去バージョン）モデルのパス
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_baseline.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target
        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    assert np.array_equal(
        model1.predict(X_test), model2.predict(X_test)
    ), "モデルの予測結果に再現性がありません"


def test_saved_model_performance(sample_data, preprocessor):
    """保存モデルの推論精度と推論時間を検証"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    start = time.time()
    y_pred = model.predict(X_test_split)
    end = time.time()
    inference_time = end - start
    accuracy = accuracy_score(y_test_split, y_pred)
    assert accuracy >= 0.75, f"保存モデルの精度が低すぎます: {accuracy}"
    assert inference_time < 1.0, f"保存モデルの推論時間が長すぎます: {inference_time}秒"


def test_performance_regression(sample_data, preprocessor):
    """過去バージョンモデルとの性能比較"""
    if not os.path.exists(BASELINE_MODEL_PATH):
        pytest.skip("ベースラインモデルが存在しないためスキップします")
    with open(MODEL_PATH, "rb") as f:
        new_model = pickle.load(f)
    with open(BASELINE_MODEL_PATH, "rb") as f:
        base_model = pickle.load(f)
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    _, X_test_reg, _, y_test_reg = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    new_acc = accuracy_score(y_test_reg, new_model.predict(X_test_reg))
    base_acc = accuracy_score(y_test_reg, base_model.predict(X_test_reg))
    assert new_acc >= base_acc, f"モデルの精度がベースラインを下回っています: 新: {new_acc}, ベースライン: {base_acc}"
    new_time_start = time.time()
    new_model.predict(X_test_reg)
    new_time = time.time() - new_time_start
    base_time_start = time.time()
    base_model.predict(X_test_reg)
    base_time = time.time() - base_time_start
    assert new_time <= base_time, f"推論時間がベースラインより遅い: 新: {new_time}秒, ベースライン: {base_time}秒"
