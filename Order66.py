import tensorflow as tf
import numpy as np

from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from Pipeline import DataPipeline
from Pipeline import model_test

def get_data():
    return DataPipeline.get_dataset()

def get_model():
    return model_test.create_vvg16()

def evaluate_model(cc):
    dataset = get_data()
    model = get_model()
    # evaluate model
    scores = cross_val_score(model, dataset, scoring="accuracy", cv=cc, n_jobs=-1)
    return mean(scores), scores.min(), scores.max()

"""# calculate ideal test condiction
ideal, _, _ = evaluate_model(LeaveOneOut())
print("Ideal: % 3f" % ideal)
folds = range(2, 18)
means, mins, maxs = list(), list(), list()
# evaluate for each k value
for k in folds:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    k_mean, k_min, k_max = evaluate_model(cv)
    print("> folds=%d, accuracy=%.3f (%.3f,%.3f)" % (k, k_mean, k_min, k_max))
    means.append(k_mean)
    means.append(k_mean - k_min)
    means.append(k_max - k_mean)
pyplot.errorbar(folds, means, yerr=[mins, maxs], fmt="o")
pyplot.plot(folds, [ideal for _ in range(len(folds))], color="r")
pyplot.show()"""