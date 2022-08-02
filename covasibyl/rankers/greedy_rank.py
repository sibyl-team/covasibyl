import sys
import time
import numpy as np
import pandas as pd
from numba import njit
from .template_rank import AbstractRanker

TAU_INF = 10000000

class GreedyRanker(AbstractRanker):


    def __init__(self,
                include_S = True,
                tau = TAU_INF,
                sec_neigh = False,
                lamb = 0.99):
        self.description = "class for tracing greedy inference of openABM loop"
        self.include_S = include_S
        self.tau = tau
        self.sec_neigh = sec_neigh
        self.lamb = lamb
        self.rng = np.random.RandomState(1)
        self.debug = False

    def init(self, N, T):
        self.contacts = None
        #dummy obs, needed if the first time you add only one element
        self.obs = [(0,-1,0)] 
        self.T = T
        self.N = N
        self.rank_not_zero = np.zeros(T)

        return True

    '''def _save_contacts(self, daily_contacts):
        """
        Save contacts in a pandas dataframe
        This is slower than numpy but easier to handle
        """
        if isinstance(daily_contacts, np.recarray):
            daily_contacts.dtype.names = "i", "j", "t", "lambda"
            cts_d = pd.DataFrame(daily_contacts)
        else:
            cts_d = pd.DataFrame(np.array(daily_contacts), columns=["i", "j", "t", "lambda"])

        assert len(cts_d) == len(daily_contacts)
        print(f"{len(cts_d)} new contacts,", end=" ")
        if self.contacts is None:
            self.contacts = cts_d
        else:
            self.contacts = pd.concat((self.contacts, cts_d), ignore_index=True)
    '''
    def _save_contacts(self, daily_contacts):
        """
        Save contacts in a pandas dataframe
        This is slower than numpy but easier to handle
        """
        conts_dtype = np.dtype([(k, "int") for k in ["i","j","t"]]+[("lambda", "float")])

        if isinstance(daily_contacts, np.recarray):
            cts_d = daily_contacts.copy()
            cts_d.dtype.names = "i", "j", "t", "lambda"
            
        else:
            cts_d = np.array(daily_contacts,dtype=conts_dtype)

        assert len(cts_d) == len(daily_contacts)
        print(f"{len(cts_d)} new contacts,", end=" ")
        if self.contacts is None:
            self.contacts = cts_d
        else:
            self.contacts = np.concatenate((self.contacts, cts_d))
    
    def _clear_old_contacts(self, t_day):
        if self.contacts is not None:
            mday = (self.contacts["t"] >= t_day - self.tau)
            self.contacts = self.contacts[mday]

        """while len(self.contacts) > 0:
            if self.contacts[0][2] < t_day - self.tau:
                self.contacts.pop()
            else:
                break
        """

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        t0=time.time()
        for obs in daily_obs:
            self.obs.append(obs)
        tobs = time.time() - t0 
        if self.debug: print(f"tobs: {tobs:4.3e} s,",end=" ")
        t0=time.time()
        self._clear_old_contacts(t_day)
        tclr = time.time()-t0
        t0=time.time()
        self._save_contacts(daily_contacts)
        tc=time.time()-t0
        
        if self.debug: print(f" tclear: {tclr:4.3e}, t_savec: {tc:6.5f} s", end="\n")

        obs_df = pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
        contacts_df = pd.DataFrame({k: self.contacts[k] for k in ("i", "j", "t", "lambda")})

        if not self.include_S:
            obs_df = obs_df[obs_df.s != 0] # just infected
        t0 = time.time()
        if self.sec_neigh:
            rank_greedy = run_greedy_sec_neigh(self, obs_df, t_day, contacts_df, self.N, tau = self.tau, lamb = self.lamb,verbose=False) 
        else:
            rank_greedy = run_greedy(obs_df, t_day, contacts_df, self.N, self.rng, tau = self.tau, verbose=False, debug=self.debug)
        tgre=time.time() - t0
        print(f"t_greedy_tot: {tgre:6.3f}")

        dict_greedy = dict(rank_greedy)
        self.rank_not_zero[t_day] =  sum(1 for x in rank_greedy if x[1] > 0)
        data["rank_not_zero"] = self.rank_not_zero
        rank = list(sorted(rank_greedy, key=lambda tup: tup[1], reverse=True))

        return rank
@njit()
def _sum_score_contacts(contacts_arr, maxS, minR, score):
    for i, j, t in contacts_arr:
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                score[i] += 1

def run_greedy(observ, T:int, contacts, N, rng, noise = 1e-3, tau = TAU_INF, verbose=True, debug=False):

    t0 = time.time()
    observ = observ[(observ["t_test"] <= T)]
    contacts = contacts[(contacts["t"] <= T) & (contacts["t"] >= T-tau)]

    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I

    # debug
    #idx_I_at_T = observ[(observ['s'] == 1) & (observ['t_test'] == T)].to_numpy()
    #idx_I_assumed = np.setdiff1d(idx_I, idx_I_at_T)

    #idx_S_anyT = observ[(observ['s'] == 0) & (observ['t_test'] < T)]['i'] # observed S at time < T
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking

    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,N), idx_all) # these have no contacts -> tail of the ranking


    idx_to_inf = np.setdiff1d(idx_all, idx_I) # nor I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # nor S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # nor R anytime

    idcs_t = time.time()-t0
    if debug: print(f"tinds: {idcs_t:3.2e} s " ,end="")
    t0=time.time()


    maxS = -1 * np.ones(N)
    minR = T * np.ones(N)
    for i, s, t_test in  observ[["i", "s", "t_test"]].to_numpy():
        if s == 0 and t_test < T:
            maxS[i] = max(maxS[i], t_test)
        if s == 2:
            minR[i] = min(minR[i], t_test)
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)
    tloop = time.time()-t0
    if debug: print(f"tloop: {tloop:4.1e} s", end=" ")
    if verbose:
        print("! Assuming contacts as direct links !", file=sys.stderr)
        print("! Assuming that if i is infected at t < T (and not observed as R), it is still infected at T !", file=sys.stderr)

    #Score = pd.Series(np.zeros(N)) #dict([(i, 0) for i in range(N)])
    score_arr = np.zeros(N)
    print(f"all contacts: {len(contacts)}")
    t1 = time.time()
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]
    tcut = time.time() -t1
    print(f"all contacts cut: {len(contacts_cut)}, taken: {tcut:4.2f}")
    t0 = time.time()
    """for i, j, t in contacts_cut[["i", "j", "t"]].values:
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += 1
    """
    _sum_score_contacts(contacts_cut[["i", "j", "t"]].values, maxS, minR, score_arr)
    Score = pd.Series(score_arr)
    tloopsc = time.time() -t0
    t0=time.time()
    for i in range(0,N):
        if verbose:
            if i % 1000 == 0:
                v =i/N
                print(f"\rDone... {v:3.2%}", end="")
        if i in idx_non_obs:
            Score[i] = -1 + rng.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max()
        elif i in idx_S: #at time T
            Score[i] = -1 + rng.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + rng.rand() * noise
    if verbose: print("")
    #sorted_Score = list(sorted(Score.items(),key=lambda item: item[1], reverse=True))
    Score = Score.sort_values(ascending=False)
    sorted_Score = list(Score.to_dict().items())
    tscore = time.time() - t0
    if debug: print(f"tcs_score: {tloopsc:5.3f}, tscore {tscore:5.3f}", end=" ")
    return sorted_Score



def run_greedy_weighted(self, observ, T, contacts, N, noise = 1e-3, verbose=True):

    observ = observ[(observ["t_test"] <= T)]
    contacts = contacts[(contacts["t"] <= T)]

    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I

    # debug
    #idx_I_at_T = observ[(observ['s'] == 1) & (observ['t_test'] == T)].to_numpy()
    #idx_I_assumed = np.setdiff1d(idx_I, idx_I_at_T)

    idx_S_anyT = observ[(observ['s'] == 0) & (observ['t_test'] < T)]['i'] # observed S at time < T
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking

    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,N), idx_all) # these have no contacts -> tail of the ranking


    idx_to_inf = np.setdiff1d(idx_all, idx_I) # nor I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # nor S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # nor R anytime


    maxS = dict()
    minR = dict()
    for i in range(0,N):
        if i in idx_S_anyT:
            maxS[i] = observ[(observ['i'] == i) & (observ['s'] == 0)]['t_test'].max()
        else:
            maxS[i] = -1
        if i in idx_R:
            minR[i] = observ[(observ['i'] == i) & (observ['s'] == 2)]['t_test'].min()
        else:
            minR[i] = T
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)

    if verbose:
        print("! Assuming contacts as direct links !", file=sys.stderr)
        print("! Assuming that if i is infected at t < T (and not observed as R), it is still infected at T !", file=sys.stderr)

    Score = dict([(i, 0) for i in range(N)])
    print(f"all contacts: {len(contacts)}")
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]
    print(f"all contacts cut: {len(contacts_cut)}")
    
    for i, j, t, lamb in contacts_cut.to_numpy():
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += lamb
    
    for i in range(0,N):
        if verbose:
            if i % 1000 == 0:
                print("Done... "+ str(i) + "/" + str(N))
        if i in idx_non_obs:
            Score[i] = -1 + self.rng.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max()
        elif i in idx_S: #at time T
            Score[i] = -1 + self.rng.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + self.rng.rand() * noise
    sorted_Score = list(sorted(Score.items(),key=lambda item: item[1], reverse=True))
    return sorted_Score


def run_greedy_sec_neigh(self, observ, T, contacts, N, noise = 1e-7, tau = TAU_INF, lamb = 0.99, verbose=True):

 
    from collections import Counter

    observ = observ[(observ["t_test"] <= T)]
    contacts = contacts[(contacts["t"] <= T) & (contacts["t"] >= T-tau)]

    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I

    # debug
    #idx_I_at_T = observ[(observ['s'] == 1) & (observ['t_test'] == T)].to_numpy()
    #idx_I_assumed = np.setdiff1d(idx_I, idx_I_at_T)

    idx_S_anyT = observ[(observ['s'] == 0) & (observ['t_test'] < T)]['i'] # observed S at time < T
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking

    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,N), idx_all) # these have no contacts -> tail of the ranking


    idx_to_inf = np.setdiff1d(idx_all, idx_I) # rm I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # rm S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # rm R anytime


    maxS = -1 * np.ones(N)
    minR = T * np.ones(N)
    for i, s, t_test, in  observ[["i", "s", "t_test"]].to_numpy():
        if s == 0 and t_test < T:
            maxS[i] = max(maxS[i], t_test)
        if s == 2:
            minR[i] = min(minR[i], t_test)
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)

    if verbose:
        print("! Assuming contacts as direct links !", file=sys.stderr)
        print("! Assuming that if i is infected at t < T (and not observed as R), it is still infected at T !", file=sys.stderr)

    Score = dict([(i, 0) for i in range(N)])
    print(f"all contacts: {len(contacts)}")
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]

    print(f"first NN contacts cut: {len(contacts_cut)}")
    if len(contacts_cut) > 0:
        # (i,j) are both unknown
        contacts_cut2 = contacts[(contacts["i"].isin(idx_to_inf)) \
                                 & (contacts["j"].isin(idx_to_inf))]
    idxk = []
    sec_NN = 0
    for i, j, t in contacts_cut[["i", "j", "t"]].to_numpy():
        # i to be estimated, j is infected
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += lamb + self.rng.rand() * noise
                # get neighbors k from future contacts (i,k), from the set of the unknown nodes
                aux = contacts_cut2[(contacts_cut2["i"] == i) \
                                    & (contacts_cut2["t"] > max(t,maxS[i]))]["j"].to_numpy() 
                # collect indices
                idxk = np.concatenate((idxk, aux), axis=None)
          
    sec_NN = len(idxk)
    value_occ = Counter(idxk).items()
    # upd sec NN scores using occurencies
    for (k, occk) in value_occ:
        Score[k] += lamb*lamb*occk + self.rng.rand() * noise
        
    print(f"second NN contacts: {sec_NN}")

    for i in range(0,N):
        if verbose:
            if i % 1000 == 0:
                print("Done... "+ str(i) + "/" + str(N))
        if i in idx_non_obs:
            Score[i] = -1 + self.rng.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max() + self.rng.rand() * noise
        elif i in idx_S: #at time T
            Score[i] = -1 + self.rng.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + self.rng.rand() * noise
    sorted_Score = list(sorted(Score.items(),key=lambda item: item[1], reverse=True))
    return sorted_Score

