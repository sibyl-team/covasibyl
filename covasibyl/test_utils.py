import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

import covasim.utils as cvu
import covasim.defaults as cvd

import numba as nb

from .utils import choose_w_rng, choose_n_rng
nbint   = cvd.nbint


def get_symp_probs(sim, symp_prob, pdf=None, ):
    """
    Parse symp probab as covasim
    """
    t = sim.t
    # Find probablity for symptomatics to be tested
    symp_inds  = cvu.true(sim.people.symptomatic)
    if pdf:
        symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
        inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
        count = np.nan * np.ones(inv_count.shape)
        count[inv_count != 0] = 1/inv_count[inv_count != 0]
        symp_prob = np.ones(len(symp_time))
        inds = 1 > (symp_time*symp_prob)
        symp_prob[inds] = symp_prob/(1-symp_time[inds]*symp_prob)
        symp_prob = pdf.pdf(symp_time) * symp_prob * count[symp_time]

    return symp_prob

def find_ili_inds(sim, ili_prev, symp_inds, start_day, rng=None):
    pop_size = sim['pop_size']
    ili_inds = []
    t = sim.t
    if ili_prev is not None:
        rel_t = t - start_day
        if rel_t < len(ili_prev):
            n_ili = int(ili_prev[rel_t] * pop_size)  # Number with ILI symptoms on this day
            ili_inds = choose_n_rng(pop_size, n_ili, rng) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
            ili_inds = np.setdiff1d(ili_inds, symp_inds)
    return ili_inds

def get_random_indcs_test(sim, n_tests, rng, exclude_diag=True):
    people = sim.people
    probs = np.ones(len(people.age))
    if exclude_diag:
        idx_diagnosed = np.where(sim.people.diagnosed)[0]
        probs[idx_diagnosed] = 0.
    probs /= probs.sum()

    inds_test = choose_w_rng(probs=probs, n=n_tests, unique=True, 
            rng=rng)

    return inds_test