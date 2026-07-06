# ruff: noqa: F401
# pyright: ignore[reportMissingImports]
# type: ignore

import os

import pandas as pd
from category_encoders import (
    CountEncoder,
)
from sklearn.compose import (
    ColumnTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
)

from data_processing import (
    add_datetime_features,
    clean_model_from_sub_and_trim,
    preprocess_split,
    train_val_test_data_split,
)


def prepare_data(
    filepath="./data/training.csv",
    threshold=30,
    min_group_size=10,
):
    """
    Загружает и обрабатывает данные для задачи бинарной классификации.
    Возвращает готовые массивы X_train, X_val, X_test, y_train, y_val, y_test.

    Parameters
    ----------
    filepath : str, default='./data/training.csv'
        Путь к файлу с исходными данными.
    threshold : int, default=30
        Максимальное число уникальных значений для OHE; выше – CountEncoder.
    min_group_size : int, default=10
        Минимальный размер группы для CountEncoder (редкие объединяются).

    Returns
    -------
    tuple of np.ndarray
        (X_train, X_val, X_test, y_train, y_val, y_test)
        Все массивы имеют тип float64, y – int64.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df_train_raw = pd.read_csv(filepath)
    cat_cols = df_train_raw.select_dtypes(include=["object", "category"]).columns
    df_train_raw = df_train_raw.drop(columns=["RefId"])
    df_train_raw[cat_cols] = df_train_raw[cat_cols].fillna("Unknown")
    df_train_raw = add_datetime_features(df_train_raw, "PurchDate")
    df_train_raw["Model"] = df_train_raw.apply(clean_model_from_sub_and_trim, axis=1)
    df_train_raw["Model"] = (
        df_train_raw["Model"].replace("", "missing").fillna("missing")
    )

    train_df, val_df, test_df = train_val_test_data_split(df_train_raw)
    base_date = train_df["PurchDate"].min()
    train_df, val_df, test_df = [
        preprocess_split(df, base_date) for df in [train_df, val_df, test_df]
    ]

    drop_cols = [
        "IsBadBuy",
        "PurchDate",
        "year",
        "IsOnlineSale",
        "is_weekend",
        "PRIMEUNIT",
        "AUCGUART",
        "WheelTypeID",
        "days_since_start",
        "Color",
        "Auction",
        "Nationality",
        "Size",
    ]
    X_train_raw, X_val_raw, X_test_raw = [
        df.drop(columns=drop_cols) for df in [train_df, val_df, test_df]
    ]
    y_train, y_val, y_test = [df["IsBadBuy"] for df in [train_df, val_df, test_df]]

    to_object = {"BYRNO": "object", "VNZIP1": "object"}
    X_train_raw, X_val_raw, X_test_raw = [
        df.astype(to_object) for df in [X_train_raw, X_val_raw, X_test_raw]
    ]

    threshold = 30
    cat_cols = X_train_raw.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    num_cols = X_train_raw.select_dtypes(
        include=["int64", "int32", "float64"]
    ).columns.tolist()

    n_unique = X_train_raw[cat_cols].nunique()
    high_card_cols = n_unique[n_unique > threshold].index.tolist()
    low_card_cols = n_unique[n_unique <= threshold].index.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
                low_card_cols,
            ),
            (
                "count",
                CountEncoder(
                    handle_unknown="value",
                    normalize=False,
                    min_group_size=min_group_size,
                ),
                high_card_cols,
            ),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "scaler",
                QuantileTransformer(output_distribution="normal", random_state=42),
            ),
        ]
    )

    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    return X_train, X_val, X_test, y_train, y_val, y_test
