import numpy as np
from dataset import tau2019dev
from experiment import Experiment
from features.feature_vector import fv_lbp, fv_glcm, fv_mfcc, fv_lpq

dataset = tau2019dev()  # .frac(k=0.15)
feats = [fv_lbp, fv_lpq, fv_glcm, fv_mfcc]
rule = lambda r: np.prod(r, axis=0)
experiment = Experiment(dataset=dataset, feature_sets=feats, rule=rule)
experiment.run()
