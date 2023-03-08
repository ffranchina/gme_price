import os
import pickle

from . import dataset

import numpy as np
import statsmodels.tsa.api as tsa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

class Oracle:
    def __init__(self, model_filename=None, random_state=None):
        self.random_state = random_state
        self.model_filename = model_filename
        self.model = None

        if self.model_filename and os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(50, 50), random_state=random_state, max_iter=500)
            )

    def fit(self, path, years, dataset_size):
        data = dataset.load(path, years)
        X, y = dataset.sample(data, dataset_size)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)

        if self.model_filename:
            with open(self.model_filename, 'wb') as f:
                pickle.dump(self.model, f)

        return score

    def predict(self, week):
        stl_model = tsa.STL(week, period=dataset._HOURS_PER_DAY).fit()

        day_means = week.reshape(-1, dataset._HOURS_PER_DAY).mean(axis=1)
        forecasted_trend = self.model.predict(day_means.reshape(1, -1))

        px = np.array(np.arange(week.size+dataset._HOURS_PER_DAY).reshape(-1, dataset._HOURS_PER_DAY)).mean(axis=1)
        py = np.concatenate((day_means, forecasted_trend))

        x = np.linspace(week.size, week.size+dataset._HOURS_PER_DAY, dataset._HOURS_PER_DAY)
        y = np.interp(x, px, py)

        seasonal_mean = stl_model.seasonal.reshape(-1, dataset._HOURS_PER_DAY).mean(axis=0)

        return y + seasonal_mean