import os
import random
import datetime

import pandas as pd
import numpy as np

_PRICE_LABEL = "price"
_TIMESTAMP_LABEL = "ts"

_HOURS_PER_DAY = 24
_DAYS_PER_WEEK = 7
_HOURS_PER_WEEK = _HOURS_PER_DAY * _DAYS_PER_WEEK


def load(path, years,
    filename_template="Anno {}.xlsx",
    sheet_name="Prezzi-Prices",
    price_name="PUN"):
    filepath = os.path.join(path, filename_template)

    data = []
    timestamps = []

    for i in years:
        df = pd.read_excel(filepath.format(i), sheet_name)
        data.append( np.array(df[price_name]) )
        timestamps.append( np.array(df.iloc[:, 0]) )

    data = np.concatenate(data)
    timestamps = np.concatenate(timestamps)

    return pd.DataFrame({_TIMESTAMP_LABEL: timestamps, _PRICE_LABEL: data})

def nth_week(data, w, d=0):
    i = w * _HOURS_PER_WEEK + d * _HOURS_PER_DAY
    week = np.array(data[_PRICE_LABEL][i : i + _HOURS_PER_WEEK])
    plus = np.array(data[_PRICE_LABEL][i + _HOURS_PER_WEEK : i + _HOURS_PER_WEEK + _HOURS_PER_DAY])
    first_day = datetime.datetime.strptime(str(data[_TIMESTAMP_LABEL][i]), "%Y%m%d").date()
    first_weekday = first_day.weekday()

    return {
        'week': week,
        'plus': plus,
        'f_day': first_day,
        'f_weekday': first_weekday
    }

def random_week(data, random_state=None):
    weeks = (len(data) // _HOURS_PER_WEEK) -1

    random.seed(random_state)
    w = random.randrange(0, weeks)
    d = random.randrange(0, _DAYS_PER_WEEK)

    return nth_week(data, w, d)

def sample(data, size, random_state=None):
    x = []
    y = []
    for i in range(size):
        week_data = random_week(data, random_state)
        x.append( week_data['week'].reshape(-1, _HOURS_PER_DAY).mean(axis=1) )
        y.append( week_data['plus'].mean() )
    
    return np.array(x), np.array(y)