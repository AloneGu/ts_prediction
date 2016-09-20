from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense

import matplotlib.pyplot as plt



class ForecastModel(object):
    '''
    Abstract base class for all neuralforecast models to come.
    Subclasses have to provide a keras model as self.model
    '''

    def __init__(self):
        self.has_data = False

    def get_data(self):
        if not self.has_data:
            raise ("preprocess model with a time-series first")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def preprocess(self, ts, train_percentage):
        '''
        Return preprocessed data split in train and test set.
        To implement this method, set
            self.X_train
            self.X_test
            self.y_train
            self.y_test
        for the provided time-series ts.
        '''
        raise NotImplementedError

    def fit(self, ts, train_percentage=0.8, batch_size=32, nb_epoch=50):
        if not self.has_data:
            self.preprocess(ts, train_percentage)
        self.model.fit(self.X_train, self.y_train, batch_size=32, nb_epoch=50,
                       validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)

    def plot_predictions(self, out_file):
        n = len(self.X_test)

        original = self.y_test
        prediction = self.model.predict(self.X_test)

        fig = plt.figure()
        plt.plot(range(n - 1), original[:-1])
        plt.plot(range(n - 1), prediction[1:])
        fig.savefig(out_file)
