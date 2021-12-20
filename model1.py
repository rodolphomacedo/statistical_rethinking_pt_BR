import numpy as np
import matplotlib.pyplot as plt
import pystan


class Models:

    def model1(self):
        X = np.random.normal(5, 1, 1000)
        my_data = {'N': len(X), 'X': X}

        sm = pystan.StanModel(file='my_model1.stan')

        fit = sm.sampling(data=my_data, iter=1000, chains=4)

        return fit


if __name__ == '__main__':
    m = Models()
    result = m.model1()
    print(result)

