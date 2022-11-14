import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

import covasim.utils as cvu
import covasim.defaults as cvd

import numba as nb

from .utils import choose_w_rng, choose_n_rng, check_free_birds_EI
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
        ## TODO: add ili_prev symptoms before intervention day (start_day > t)
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

def make_symp_probs_covasim_def(sim, start_day, symp_test_p, pdf, ili_prev, rng):
    symp_inds = cvu.true(sim.people.symptomatic)
    symp_prob = get_symp_probs(sim, symp_test_p, pdf)
    
    # Define symptomatics, accounting for ILI prevalence
    ili_inds = find_ili_inds(sim, ili_prev, symp_inds, start_day, rng)

    diag_inds = cvu.true(sim.people.diagnosed)

    test_probs = np.zeros(sim['pop_size']) # Begin by assigning equal testing probability to everyone
    test_probs[symp_inds]       = symp_prob            # People with symptoms (true positive)
    test_probs[ili_inds]        = symp_test_p ## Ili inds, can be 0

    test_probs[diag_inds] = 0.0 # People who are diagnosed don't test

    return test_probs

def possible_find_infected(sim):
    """
    Check if it's possible that there can be new infected after this time instant
    """
    nfree_birds = len(check_free_birds_EI(sim.people))
    v= (nfree_birds == 0 and max(sim["iso_factor"].values()) <=0) or (sum(sim.people.exposed) ==0)
    return not v
    