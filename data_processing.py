# ruff: noqa: F401
# pyright: ignore[reportMissingImports]
# type: ignore

import numpy as np
import pandas as pd


def train_val_test_data_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделяет DataFrame на обучающую, валидационную и тестовую выборки по времени
    в пропорции 33%/33%/33% (по количеству записей).
    """
    df = data.copy()
    df["PurchDate"] = pd.to_datetime(df["PurchDate"])
    df = df.sort_values("PurchDate").reset_index(drop=True)

    n = len(df)
    train_end = n // 3
    val_end = 2 * n // 3

    data_train = df.iloc[:train_end].copy()
    data_val = df.iloc[train_end:val_end].copy()
    data_test = df.iloc[val_end:].copy()

    return data_train, data_val, data_test


def clean_model_from_sub_and_trim(row: pd.Series) -> str:
    """
    Удаляет из названия модели слова, встречающиеся в подмодели (SubModel) и комплектации (Trim).

    Функция предназначена для построчной обработки DataFrame.
    Извлекает значения колонок 'SubModel' и 'Trim', разбивает их на слова,
    затем удаляет эти слова из строки колонки 'Model'.
    Если после очистки не осталось слов, возвращает пустую строку.

    Parameters
    ----------
    row : pd.Series
        Строка DataFrame, содержащая как минимум колонки 'SubModel', 'Trim' и 'Model'.
        Значения могут быть типа str, float (NaN) или другими.

    Returns
    -------
    str
        Очищенное значение модели. Если входное значение Model не является строкой,
        возвращается исходное значение, приведённое к строке, либо пустая строка.
    """
    words_to_remove = set()

    for col in ["SubModel", "Trim"]:
        val = row.get(col)
        if isinstance(val, str):
            words_to_remove.update(val.split())

    model_val = row.get("Model")

    if not isinstance(model_val, str):
        return model_val

    model_words = model_val.split()
    cleaned_words = [w for w in model_words if w not in words_to_remove]
    if not cleaned_words:
        return ""

    return " ".join(cleaned_words)


def add_datetime_features(
    df: pd.DataFrame, date_col: str = "PurchDate"
) -> pd.DataFrame:
    """
    Добавляет признаки на основе даты: год, месяц, день недели и флаг выходного дня.

    Извлекает компоненты из указанной колонки с датой и добавляет их в DataFrame.
    День недели кодируется как 0 (понедельник) - 6 (воскресенье).
    Выходной день (суббота/воскресенье) помечается как 1.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame, содержащий колонку с датой.
    date_col : str, default 'PurchDate'
        Название колонки с датой.

    Returns
    -------
    pd.DataFrame
        Копия исходного DataFrame с добавленными колонками:
        'year', 'month', 'dayofweek', 'is_weekend'.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["is_weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
    return df


def add_days_since_start(
    df: pd.DataFrame, min_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Добавляет колонку с количеством дней, прошедших с указанной минимальной даты.

    Если min_date не задана, вычисляет её как минимальную дату в колонке 'PurchDate'
    переданного DataFrame. Для корректной работы пайплайна с временным разбиением
    следует вычислять min_date на тренировочной выборке и передавать её явно.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame, обязательно содержащий колонку 'PurchDate' в формате,
        приводимом к datetime.
    min_date : pd.Timestamp, default None
        Базовая дата, относительно которой вычисляется количество дней.
        Если None, берётся минимальная дата из df['PurchDate'].

    Returns
    -------
    pd.DataFrame
        Копия исходного DataFrame с добавленной колонкой 'days_since_start'.
    """
    df = df.copy()
    if min_date is None:
        min_date = df["PurchDate"].min()
    df["days_since_start"] = (df["PurchDate"] - min_date).dt.days
    return df


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет циклические признаки для месяца (sin/cos) для использования в моделях машинного обучения.

    Преобразует числовую колонку 'month' (предполагается, что значения от 1 до 12)
    в две колонки: 'month_sin' и 'month_cos', кодирующие циклическую природу месяцев.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame, содержащий колонку 'month' с целочисленными значениями месяца.

    Returns
    -------
    pd.DataFrame
        Копия исходного DataFrame с добавленными колонками 'month_sin' и 'month_cos'.
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def preprocess_split(df, base_date):
    df = add_days_since_start(df, min_date=base_date)
    df = add_cyclic_month(df)
    return df
