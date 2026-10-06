"""
ANN surrogates: fit a ReLU or tanh network to any labeled data set (x, y) and save it for
embedding in an optimization model. Data comes as a ``hybopt.data.Dataset`` or as arrays/DataFrames.
"""

from hybopt.ann.surrogate import ANNSurrogate, fit_ann
from hybopt.ann.train import TrainSettings

__all__ = ["ANNSurrogate", "TrainSettings", "fit_ann"]
