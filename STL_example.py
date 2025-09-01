import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет STL-декомпозицию к данным о количестве рождений.
    
    Args:
        df (pd.DataFrame): Входной DataFrame, содержащий как минимум следующие колонки:
            - 'dt' : дата (pd.Timestamp)
            - 'cnt_people' : количество людей, родившихся в указанную дату

    Returns:
        pd.DataFrame: DataFrame, содержащий:
            - 'birth_day' : день года в формате 'MM-DD'
            - 'frac_people' : нормализованная доля рождений с учётом сезонности и тренда
    """
    # Удаляем 29 февраля для упрощения работы с STL
    df_crop_no_leap = df[pd.DatetimeIndex(df['dt']).strftime('%m-%d') != '02-29'].copy()

    # Применяем STL-декомпозицию
    stl = STL(
        df_crop_no_leap['cnt_people'],
        seasonal=7,
        period=365,
        trend=365 * 2 + 1,
        robust=True,
        seasonal_deg=0
    )
    res = stl.fit()

    # Нормализуем тренд: min(Trend)/Trend(t)
    df_crop_no_leap['norm_trend'] = np.min(res.trend) / res.trend

    # Возвращаем 29 февраля - интерполируем пропущенные значения
    df = df.join(df_crop_no_leap.set_index('dt'), on='dt', rsuffix='_n').interpolate()

    # Корректируем данные на основе нормализованного тренда
    df['tres'] = df['norm_trend'] * df['cnt_people']

    # Группируем по дням и нормализуем
    df['day'] = pd.DatetimeIndex(df['dt']).strftime('%m-%d')
    daily_frac = df.groupby('day')['tres'].sum() * (1 / df['tres'].sum())

    result_df = pd.DataFrame({
        'birth_day': sorted(df['day'].unique()),
        'frac_people': daily_frac.values
    })

    return result_df