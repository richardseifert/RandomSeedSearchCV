from RandomSeedSearchCV import RandomSeedSearchCV, randomseed_rfr_maker
from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

if __name__ == '__main__':
    boston = load_boston()
    X = boston['data']
    y = boston['target']

    res = RandomSeedSearchCV(randomseed_rfr_maker,X,y,validation=0.2,N=100)
    best_model = randomseed_rfr_maker(int(res[0,0]))
    best_model.fit(X,y)
    y_pred = best_model.predict(X)

    fig,ax = plt.subplots()
    ax.plot(y,y_pred,'o',linestyle='none',color='cornflowerblue')
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    plt.show()