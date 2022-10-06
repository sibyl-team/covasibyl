import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import gamma as gamma_f
import pandas as pd


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
         cwd=Path(__file__).resolve().parent
         ).decode('ascii').strip()

def remove_invalid_cts(p1, p2, beta):
    """
    Remove invalid contacts, then duplicate them
    """
    valid = (p1!=p2)
    i = p1[valid]
    j = p2[valid]
    m = beta[valid]
    return np.concatenate((i,j)), np.concatenate((j,i)), np.concatenate((m,m))

def get_contacts_day(people):
    N = len(people.sex)
    c = 0
    for n,lay in people.contacts.items():  
        w = people.pars["beta_layer"][n]
        u = remove_invalid_cts(**lay)
        #print(f" {n}-> {w}", end=" ")
        mat = sp.csr_matrix((u[2]*w,(u[0],u[1])), shape=(N,N))
        c += mat
    cend = c.tocoo()
    #print("")
    return pd.DataFrame(dict(zip(["i","j","m"],(cend.row, cend.col, cend.data))) )

def get_day_contacts_mat(people, exclude_lays=[], to_df=True):
    N = len(people.sex)
    c = 0
    for n,lay in people.contacts.items():
        if n in exclude_lays:
            continue
        w = people.pars["beta_layer"][n]
        u = remove_invalid_cts(**lay)
        #print(f" {n}-> {w}", end=" ")
        mat = sp.csr_matrix((u[2]*w,(u[0],u[1])), shape=(N,N))
        c += mat
    if to_df:
        cend = c.tocoo()
        #print("")
        return pd.DataFrame(dict(zip(["i","j","m"],(cend.row, cend.col, cend.data))) )
    else:
        return c

def cts_mat_to_df(c, idcs=["i","j","m"]):
    """
    Transform sparse contact matrix into dataframe of contacts
    """
    cend = c.tocoo()
    return pd.DataFrame(dict(zip(idcs,(cend.row, cend.col, cend.data))) )

def cts_df_to_sparse(cts):
    return sp.coo_matrix((cts["m"], (cts["i"], cts["j"]))).tocsr()

def filt_contacts_df(cts, idcs, multipl, N, only_i=False):
    mat  = sp.coo_matrix((cts["m"], (cts["i"], cts["j"])), shape=(N,N)).tocsr()
    d = np.ones(N)
    d[idcs] = multipl
    filt = sp.diags(d, format="csr")
    if not only_i:
        mat = mat.dot(filt) ##djj
    #if both: 
    mat = filt.dot(mat) #dii

    return cts_mat_to_df(mat)

def filt_contacts_mult(cts, weight:np.ndarray, N:int, only_i=False, only_j=False, return_df=True):
    """
    Filter contacts by multiplying the contacts value by a 
    factor dependent on the index

    Args:
        cts (dict-like, pandas Dataframe): the contacts, i,j,m
        weight (np.ndarray): the vector of weight of i
        N (int): number of individuals
        only_i (bool, optional): _description_. Defaults to False.

    Returns:
        dataframe of contacts reweighted
    """
    mat  = sp.coo_matrix((cts["m"], (cts["i"], cts["j"])),shape=(N,N)).tocsr()
    filt = sp.diags(weight, format="csr")
    if only_i and only_j:
        raise ValueError("Cannot multiply only i and only j")
    if not only_i:
        mat = mat.dot(filt) ##djj
    #if both:
    if not only_j:
        mat = filt.dot(mat) #dii
    if return_df:
        return cts_mat_to_df(mat)
    else:
        return mat

def sel_contacts_idx_df(cts, idcs, N:int, which="or"):
    """
    Filter contacts by index

    Args:
        cts (dict-like, pandas Dataframe): the contacts, i,j,m
        idcs (np.ndarray, list): indices to keep
        N (int): number of individuals
        which (string, optional): the operation to perform

    Returns:
        dataframe of contacts reweighted
    """
    mat  = sp.coo_matrix((cts["m"], (cts["i"], cts["j"])),shape=(N,N)).tocsr()
    v = np.zeros(N)
    v[idcs] = 1.
    filt = sp.diags(v, format="csr")
    mright = mat.dot(filt)
    
    if which == "or":
        mleft = filt.dot(mat)
        mat = mright + mleft - filt.dot(mright)
    else:
        raise ValueError("Invalid value for filter")

    return cts_mat_to_df(mat)


def check_free_birds(people):
    free_idx = (people.infectious & np.logical_not(people.diagnosed) ).nonzero()[0]
    return free_idx

def choose_n_rng(max_n:int, n:int, rng):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose
        rng : RandomState generator, optional

    **Example**::

        choices = cv.choose(5, 2) # choose 2 out of 5 people with equal probability (without repeats)
    '''
    if rng is None:
        rng = np.random
    return rng.choice(max_n, n, replace=False)

def choose_w_rng(probs, n, unique=True, rng=None): # No performance gain from Numba
    '''
    Choose n items (e.g. people), each with a probability from the distribution probs.

    FM: Added the rng random state argument
    Args:
        probs (array): list of probabilities, should sum to 1
        n (int): number of samples to choose
        unique (bool): whether or not to ensure unique indices
        rng: a numpy.RandomState instance, or None

    **Example**::

        choices = cv.choose_w([0.2, 0.5, 0.1, 0.1, 0.1], 2) # choose 2 out of 5 people with nonequal probability.
    '''
    probs = np.array(probs)
    n_choices = len(probs)
    n_samples = int(n)
    probs_sum = probs.sum()
    if probs_sum: # Weight is nonzero, rescale
        probs = probs/probs_sum
    else: # Weights are all zero, choose uniformly
        probs = np.ones(n_choices)/n_choices
    if rng is None:
        rng = np.random
    return rng.choice(n_choices, n_samples, p=probs, replace=not(unique))

def n_binomial(prob, n, rng=None):
    '''
    Perform multiple binomial (Bernolli) trials
    
    From Covasim, added rng argument

    Args:
        prob (float): probability of each trial succeeding
        n (int): number of trials (size of array)
        rng: a numpy.RandomState instance, or None


    Returns:
        Boolean array of which trials succeeded

    '''
    if rng is None:
        rng = np.random
    return rng.random(n) < prob

def choose_probs(probs: np.ndarray, rng=None):
    """
    Random sampling of implicit indices, according to the probability probs
    Equivalent to series of Bernoulli random trials
    
    This method saves random numbers by extracting all zeros out of the probability

    Args:
        probs (np.ndarray): probabilities
        rng (numpy RandomState, optional): random generator. Defaults to the default random number generator.
    """
    if rng is None:
        rng = np.random
    nz = np.where(probs)[0]
    pp = probs[probs>0]
    idcs = nz[rng.random(len(pp)) < pp]

    return idcs

def gamma_pdf_full(x, alfa, beta ):
    return beta**alfa*x**(alfa-1)*np.exp(-1*beta*x)/gamma_f(alfa)

def get_state(N, dates_save, day):
    state = np.zeros(N,dtype=np.int8)
    exp = (dates_save["date_exposed"] <= day)
    state[exp] = 1
    infect = (dates_save["date_infectious"]<=day)
    sympt = (dates_save["date_symptomatic"]<=day)
    nonsympt = infect & (~sympt)

    neversympt = np.isnan(dates_save["date_symptomatic"])
    asymp = (nonsympt & neversympt)
    presymp = nonsympt & (~neversympt)
    state[asymp] = 2
    state[presymp] = 3

    state[sympt] = 4
    severe = (dates_save["date_severe"] <= day)
    critical = (dates_save["date_critical"]<= day)
    state[severe] = 5
    state[critical] = 6

    state[dates_save["date_recovered"]<=day] = 7
    state[dates_save["date_dead"]<=day] = 8

    return state