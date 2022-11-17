import numpy as np
import pandas as pd

import covasim.utils as cvu
import covasim.defaults as cvd

import warnings

from . import utils

from collections import defaultdict

def write_tests_state(people, stats_df, inds_test):
    """
    Write the actual state of people to the statistic
    """

    is_E = (people.exposed &(~people.infectious))
    stats_df.loc[cvu.itruei(is_E, inds_test), "t_state"] = "E"
    stats_df.loc[cvu.itruei(people.susceptible, inds_test), "t_state"] = "S"
    
    #today_tests.loc[cvu.itruei(people.infectious, inds_test), "true_state"] = "S"

    nonsympt = (people.infectious &(~people.symptomatic) )[inds_test]
    asympt = nonsympt & np.isnan(people.date_symptomatic[inds_test])
    presympt = nonsympt & np.logical_not(asympt)
    stats_df.loc[inds_test[asympt], "t_state"] = "AS"
    stats_df.loc[inds_test[presympt],"t_state"] = "PS"

    symptomat = people.symptomatic[inds_test]
    severe = people.severe[inds_test]
    critical = people.critical[inds_test]
    mild = symptomat & np.logical_not(severe | critical)
    stats_df.loc[inds_test[mild], "t_state"] = "MS"
    stats_df.loc[inds_test[critical], "t_state"] = "CS"
    stats_df.loc[inds_test[severe], "t_state"] = "SS"

    stats_df.loc[cvu.itruei(people.dead, inds_test), "t_state"] = "D"
    stats_df.loc[cvu.itruei(people.recovered, inds_test), "t_state"] = "R"


def random_all_p(p, n, rand=np.random):
    '''
    Perform multiple binomial (Bernolli) trials

    Args:
        p (float): probability of each trial succeeding
        n (int): number of trials (size of array)
        rand   : give the random number generator if you need to

    Returns:
        Boolean array of which trials succeeded
    '''
    return rand.random(n) < p

def get_default_test_p_sym(sim, t, symp_test, test_pdf, test_probs=None):
    """
    Default way of testing symptomatic individuals
    """
    # Calculate test probabilities for people with symptoms
    N = len(sim.people.age)
    symp_inds = cvu.true(sim.people.symptomatic)
    if test_probs is None:
        test_probs = np.ones(N)
    if test_pdf: # Handle the onset to swab delay
        symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
        inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
        count = np.nan * np.ones(inv_count.shape) # Initialize the count
        count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
        symp_test *= test_pdf.pdf(symp_time) * count[symp_time] # Put it all together
        ### comment on above: basically we use the probability from pdf, but
        ### then divide by the empirical frequency (number of people with delay =d / tot number of symptomatics)
        ### This way, the less frequent delays are more likely to be chosen

    test_probs[symp_inds] *= symp_test # Update the test probabilities
    return test_probs

def compact_test_dict(idcs_dict):
    """
    Read dict of tests and get the source for each test
    """
    test_all = None
    #sources_test = {}
    src_idx_all = []
    c=0
    for k in idcs_dict.keys():
        ii = np.unique(idcs_dict[k])
        if test_all is None:
            test_all = ii
        else:
            if len(np.intersect1d(ii,test_all)) != 0:
                raise ValueError("Testing same individual more than once")
            test_all = np.concatenate((test_all, ii))
        #sources_test[c] =k
        src_idx_all.append(np.full(len(ii), k))
        c+=1
   
    idcs= np.concatenate(src_idx_all)
    ordin=np.argsort(test_all)
    return test_all[ordin], idcs[ordin]
class CovasimTester:

    def __init__(self, sim, seed=None, contain=True, warn_diag=True):
        """
        A tester for covasim that saves the actual result of tests,
        not just the positive ones.

        contain: if True, actually diagnose individuals so they are isolated
        """
        
        pars = sim.pars
        self.N = pars["pop_size"]

        self.rand_seed = int(sim["rand_seed"])
                
        self.give_diagnosis = contain

        self.reinit()
        self.warn_diag=warn_diag

    
    def reinit(self):
        #self.date_diagn_state = np.full((3,self.N), np.nan)
        self.date_diagnosed = np.full(self.N, np.nan)
        self.diagnosed = np.zeros(self.N,dtype=bool)

        self.tests_stats = []

        self.date_posit_test = np.full_like(self.date_diagnosed, np.nan)
        self.randstate = np.random.RandomState(np.random.PCG64(self.rand_seed))

        self.test_results=defaultdict(list)

        self._warned = defaultdict(lambda: False)

        if not self.give_diagnosis:
            warnings.warn("WARNING: Not containing the tested individuals")

    
    def _not_diagnosed(self):
        return np.isnan(self.date_diagnosed)

    def run_tests(self, sim, inds, test_sensitivity=1.0, test_specificity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people, accounting for recovered 
        and susceptible individuals (SIR model)


        Args:
            inds: indices of who to test, can be a list-like or a dict of lists
            test_sensitivity (float): probability of a true positive
            test_specificity (float): probability of a true negative
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''
        try:
            inds_all, src_tests = compact_test_dict(inds)
            inds = inds_all
                #source_all.append([k]*)
        except AttributeError:
            ### we have a list
            inds = np.unique(inds)
            #sources_test = {0: "unk"}
            src_tests = np.full(len(inds), "u")
        people = sim.people
        today = sim.t
        ## inds now contains all tests
        num_tests = len(inds)
        if num_tests == 0:
            ### Do nothing
            return
        
        ## register with covasim
        sim.results['new_tests'][today] += num_tests
        ## check that there are no diagnosed (eg positive-tested) people
        diagnosed = people.diagnosed[inds] | self.diagnosed[inds]
        if self.warn_diag and diagnosed.sum() > 0:
            msg = f"at time {sim.t}: Asking to test people who have already been diagnosed"
            warnings.warn(msg)

        people.tested[inds] = True
        people.date_tested[inds] = sim.t # Only keep the last time they tested

        
        ## lost tests
        not_lost = utils.n_binomial(1.0-loss_prob, len(inds))
        inds_test = inds[not_lost]
        num_lost = np.logical_not(not_lost).sum()

        today_tests = pd.DataFrame(inds, columns=["i"])
        today_tests["date_req"] = sim.t
        today_tests["t_state"] = "U"
        today_tests["res_state"] = -2
        today_tests.set_index("i", inplace=True)
        today_tests["src_test"] = src_tests
        #if(num_lost > 1):
        #    print(f"Lost {num_lost} tests")
        ## exposed individuals remain exposed when infectious

        write_tests_state(people, today_tests, inds_test)

        ## check susceptibles
        ## exposed people remain exposed when infectious on covasim
        is_E = (people.exposed &(~people.infectious))
        susc_inds = cvu.itruei((people.susceptible), inds_test)
        neg_test = random_all_p(test_specificity, len(susc_inds), rand=self.randstate)
        res_susc = susc_inds[neg_test]
        
        res_infected = susc_inds[np.logical_not(neg_test)] # false positive
        if(test_specificity >= 1.):
            assert len(res_infected) == 0

        ### find infectious indices
        #inf_or_exp = people.infectious[inds_test] | people.exposed[inds_test]
        is_I_idcs = cvu.itruei(people.infectious | is_E, inds_test) #return indices
        ## find the ones which test positive
        pos_test      = random_all_p(test_sensitivity, len(is_I_idcs), rand=self.randstate)
        is_inf_pos    = is_I_idcs[pos_test]
        
        ### concatenate results
        res_susc = np.concatenate((res_susc, is_I_idcs[np.logical_not(pos_test)]))
        res_infected = np.concatenate((res_infected, is_inf_pos))
        
        # check for recovered
        ids_rec = cvu.itruei(people.recovered, inds_test)
        # Use specificity for recovered people -> they don't have the disease anymore
        test_R = random_all_p(test_specificity, len(ids_rec), rand=self.randstate)
        res_recov = np.concatenate((ids_rec[test_R], cvu.itruei(people.dead, inds_test))) #inds_test[rec_inds]
        res_infected = np.concatenate((res_infected, ids_rec[~test_R]))

        #if we are asked to contain the epidemic (put in isolation)
        if self.give_diagnosis:
            ## Keep this for compatibility
            # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
            self.date_diagnosed[res_infected] = sim.t + test_delay
            self.date_posit_test[res_infected] = sim.t
            ## COPY ON COVASIM: important!! -> diagnosed individuals are isolated
            sim.people.date_diagnosed[res_infected] = sim.t +test_delay
            sim.people.date_pos_test[res_infected] = sim.t

        day_res =  sim.t + test_delay
        if day_res not in self.test_results:
            self.test_results[day_res] = []
        
        res_tests_t = self.test_results[day_res]

        res_tests_t.extend((i, 0, today) for i in res_susc)
        res_tests_t.extend((i, 1, today) for i in res_infected)
        res_tests_t.extend((i, 2, today) for i in res_recov)
        

        today_tests.loc[res_susc, "res_state"] = 0
        today_tests.loc[res_infected, "res_state"] = 1
        today_tests.loc[res_recov, "res_state"] = 2
        today_tests["date_res"] = sim.t + test_delay

        ## handle diagnosed
        diag_inds  = people.check_inds(self.diagnosed, self.date_diagnosed, filter_inds=None) # Find who was actually diagnosed on this timestep
        self.diagnosed[diag_inds] = True

        self.tests_stats.append(today_tests.to_records())

        return num_lost

    def get_results(self, day):

        return np.array(self.test_results[day])
        #np.stack(np.where(self.date_diagn_state == day),1)[:,::-1]
    
    def get_all_stats(self):
        if len(self.tests_stats)>0:
            return np.concatenate(self.tests_stats)
        else:
            return []

    def _observe_sources(self, sim):

        people = sim.people
        issrc = people.date_exposed == 0
        warnings.warn("Asked to observe the sources, ONLY RUN FOR DEVELOPMENT PURPOSE")
        
        if sum(issrc) == 0:
            print(f"No src found at time {sim.t}")
            return False
        
        idcs_src = cvu.true(issrc)
        #date_inf = people.date_infectious[idcs_src].astype(int)
        date_inf = [0]*len(idcs_src)
        for t, idc in zip(date_inf, idcs_src):
            self.test_results[t].append((idc, 1, t))

        return True

    def _do_random_tests(self,sim, n_tests):
        ## Old method for dev purpose
        if not self._warned["rand_tests"]:
            warnings.warn("Doing random tests!!! ONLY USE FOR DEVELOPMENT PURPOSE")
            self._warned["rand_tests"] = True
        
        today = sim.t
        people = sim.people
        probs = np.ones(len(people.age))
        probs /= probs.sum()

        inds_test = utils.choose_w_rng(probs=probs, n=n_tests, unique=True, 
                rng=self.randstate)

        is_E = (people.exposed &(~people.infectious))
        susc_inds = cvu.itruei(people.susceptible, inds_test)

        is_I_idcs = cvu.itruei(people.infectious | is_E, inds_test)

        ids_rec = cvu.itruei((people.recovered| people.dead), inds_test)

        res_tests_t = self.test_results[today]

        res_tests_t.extend((i, 0, today) for i in susc_inds)
        res_tests_t.extend((i, 1, today) for i in is_I_idcs)
        res_tests_t.extend((i, 2, today) for i in ids_rec)

