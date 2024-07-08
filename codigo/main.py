import sys

sys.path.append("/home/lucas/pic")

# import hiclass
# import numpy as np
# from dataset import tau2019dev
# from experiment import Experiment
# from features.feature_vector import fv_lbp, fv_glcm, fv_mfcc, fv_lpq, fv_rp

# import hierarchy

# dataset = tau2019dev().frac(k=0.15)
# feats = [fv_lbp]  # [fv_lbp, fv_lpq, fv_glcm, fv_mfcc]
# rule = lambda r: np.prod(r, axis=0)
# experiment = Experiment(dataset=dataset, feature_sets=feats, rule=rule)
# experiment.run_hier_inner(
#     paths=hierarchy.paths, classifier=hiclass.LocalClassifierPerNode
# )
# experiment.run_hier_outer(
#     paths=hierarchy.paths, classifier=hiclass.LocalClassifierPerNode
# )
# experiment.run()

import codigo.genetic

codigo.genetic.run()
