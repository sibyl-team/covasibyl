import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .template_rank import AbstractRanker
from .mean_field_rank import contacts_rec_to_csr, get_rank, check_inputs


def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def ranking_tracing(t, transmissions, observations, tau, rng):
    """Naive contact tracing.

    Search for all individuals that have been in contact during [t-tau, t]
    with the individuals last tested positive (observations s=I at
    t-tau <= t_test < t) and count the number of encounters.

    Returns: scores = encounters. If t < delta returns random scores.
    """
    N = transmissions[0].shape[0]
    if (t < tau):
        scores = rng.rand(N)
        return scores
    # last_tested : observations s=I for t-tau <= t_test < t
    last_tested = set(
        obs["i"] for obs in observations
        if obs["s"] == 1 and (t - tau <= obs["t_test"]) and (obs["t_test"] <= t)
    )
    # contacts with last_tested people during [t - tau, t]
    #print("CT, last_tested ", last_tested)
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact)
        for t_contact in range(t - tau, t)
        for i, j, lamb in csr_to_list(transmissions[t_contact])
        if j in last_tested and lamb # lamb = 0 does not count
    )
    encounters = pd.DataFrame({"i": range(N)})
    # no encounters -> count = 0
    if (contacts.shape[0] == 0):
        encounters["count"] = 0
    else:
        counts = contacts.groupby("i").size() # number of encounters for all i
        encounters["count"] = encounters["i"].map(counts).fillna(0)
    scores = encounters["count"].values
    return scores


def ranking_tracing_faster(t, transmissions, observations, tau, rng):
    """Naive contact tracing.

    Search for all individuals that have been in contact during [t-tau, t]
    with the individuals last tested positive (observations s=I at
    t-tau <= t_test < t) and count the number of encounters.

    Faster versions, using the sparse matrices
    @author Fabio Mazza

    Returns: scores = encounters. If t < delta returns random scores.
    """
    N = transmissions[0].shape[0]
    if (t < tau):
        scores = rng.rand(N)
        return scores
    # last_tested : observations s=I for t-tau <= t_test < t
    last_tested = set(
        obs["i"] for obs in observations
        if obs["s"] == 1 and (t - tau <= obs["t_test"]) and (obs["t_test"] <= t)
    )
    nt = len(last_tested)
    ## Indicator matrix, 1 -> tested positive
    test_pos = csr_matrix((np.ones(nt), 
                          (np.zeros(nt,dtype=int), list(last_tested))), shape=(1,N) )
    c = 0
    for t_c in range(t-tau, t):
        # skip empty matrices
        if transmissions[t_c].nnz==0:
            continue
        ## number of contacts with the positive individual at time t_c
        x = (test_pos).dot(transmissions[t_c]>0)
        c+= x #(x>0).astype(float)
    """
    For a more accurate (?) version of the CT, we could also do:
        x = (test_pos).dot(transmissions[t_c])
        c+= (x>0).astype(float)
    """
    try:
        scores = c.toarray()[0]
    except AttributeError:
        ## c is still 0
        print("the sum of transmissions is: ",c)
        scores = rng.rand(N)*1e-3
    return scores


class TracingRanker(AbstractRanker):

    def __init__(self, tau, lamb):
        self.description = "class for naive contact tracing of openABM loop."
        self.author = "https://github.com/sphinxteam"
        self.tau = tau
        self.lamb = lamb
        self.rng = np.random.RandomState(1)

    def init(self, N, T):
        self.transmissions = []
        self.observations = []
        self.T = T
        self.N = N
        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        # check that t=t_day in daily_contacts and t=t_day-1 in daily_obs
        #check_inputs(t_day, daily_contacts, daily_obs)
        # append daily_contacts and daily_obs
        
        if len(daily_contacts) > 0:
            daily_transmissions = contacts_rec_to_csr(self.N, daily_contacts, self.lamb)
        else:
            daily_transmissions = csr_matrix((self.N, self.N))
        self.transmissions.append(daily_transmissions)
        self.observations += [
            dict(i=i, s=s, t_test=t_test) for i, s, t_test in daily_obs
        ]
        ## Trick to run ranking always
        tau_today = min(self.tau, t_day)
        # scores given by mean field run from t-delta to t
        scores = ranking_tracing_faster(
            t_day, self.transmissions, self.observations, tau_today, self.rng
        )
        # convert to list [(index, value), ...]
        rank = get_rank(scores)
        return rank
