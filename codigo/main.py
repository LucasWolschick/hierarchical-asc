import sys

sys.path.append("/home/lucas/pic")

import hiclass
import numpy as np
from dataset import tau2019dev
from experiment import Experiment
from features.feature_vector import (
    fv_lbp,
    fv_glcm,
    fv_mfcc,
    fv_lpq,
    fv_rp,
    fv_lbp_mfcc,
    fv_lpq_mfcc,
    fv_lbp_mfcc_glcm,
    fv_lbp_mfcc_glcm_rp,
    fv_lbp_mfcc_rp,
    fv_lpq_mfcc_rp,
)

import hierarchy

dataset = tau2019dev()  # .frac(k=0.15)

for i, path, classifier in [
    (0, hierarchy.paths1, hiclass.LocalClassifierPerNode),
    (1, hierarchy.paths2, hiclass.LocalClassifierPerNode),
]:
    print(f"hierarchy {i}")
    feats = [fv_lpq, fv_mfcc, fv_rp]
    print(", ".join(n.id for n in feats))
    vec = []
    for _ in range(1):
        rule = lambda r: np.prod(r, axis=0)
        experiment = Experiment(
            dataset=dataset, feature_sets=feats, rule=rule, log=True
        )
        vec.append(experiment.run_hier_inner(paths=path, classifier=classifier))
    print(str(classifier) + " / " + ", ".join(n.id for n in feats) + " RESULTS:")
    print(np.average(np.array(vec), axis=0))

# import codigo.genetic

# codigo.genetic.run()
