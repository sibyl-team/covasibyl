from abc import ABCMeta, abstractmethod
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
import sciris as sc

from sklearn.metrics import roc_curve, auc, roc_auc_score

import covasim.interventions as cvi
import covasim.utils as cvu
import covasim.defaults as cvd

from .tester import CovasimTester, random_all_p
from . import utils
from . import test_utils as tuti

def calc_aucs(true_inf, true_EI, rank, idx_exclude=None):
    rank = rank.sort_index().to_numpy()
    if idx_exclude is not None:
        mask = np.ones(true_EI.shape, dtype=np.bool_)
        mask[idx_exclude] = False

        true_inf = true_inf[mask]
        true_EI = true_EI[mask]
        rank = rank[mask]

    if(true_inf.sum() > 0):
        fpr, tpr, _ = roc_curve(true_inf, rank)
        auc_inf = auc(fpr,tpr)  #if real_inf > 0 else np.nan
    else:
        auc_inf = np.nan
    if true_EI.sum() > 0:
        auc_EI = roc_auc_score(true_EI, rank)
    else:
        auc_EI = np.nan
    return auc_inf, auc_EI

def _remove_diagnosed_prob(test_probs, sim):
    ## remove already diagnosed
    already_diagnosed = sim.people.diagnosed
    idx_diagnosed = cvu.true(already_diagnosed)
    test_probs[idx_diagnosed] = 0.

class BaseRankTester(cvi.Intervention, metaclass=ABCMeta):
    """
    Class to test people with information on
    both symptomatic, quarantined and using the ranker

    remember people are quarantined before being tested
    """

    def __init__(self,
                ranker,
                label,
                symp_test_p=0.5,
                quar_test_p=None,
                start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0., test_delay=0,
                swab_delay=None,
                quar_policy=None,
                iso_cts_strength = 0.1,
                quar_factor=1.,
                logger=None,
                debug=False,
                verbose=True,
                mitigate=True,
                only_sympt=False,
                adoption_fraction=1.,
                **kwargs
                ):

        super().__init__(label="Mitigation: "+label)

        self.ranker = ranker
        #self.n_tests = num_tests
        self.symp_test = symp_test_p # Probability of testing symptomatics
        ## symp_test can also be callable, signature: callable(sim)
        self.sensitivity = sensitivity
        self.specificity = specificity
        # probability of losing a test result
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        #self.quar_test_p   = quar_test_p # Probability of testing people in quarantine in the sympt stage
        self.quar_policy = quar_policy if quar_policy else 'start'
        # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self.test_pdf = cvu.get_pdf(**sc.mergedicts(swab_delay))
        # multiply contacts strength by this factor for isolated individuals
        self.iso_cts_strength = iso_cts_strength
        ## same for the quarantine, but with both infector and infected
        self.quar_factor = quar_factor
        self.adopt_frac=adoption_fraction
        self.debug = debug

        self.contacts_day = None
        self.delayed_init = False
        self.N = None
        self.all_observ = None
        self.daily_obs = None
        self.ranker_data = {"logger":logger}
        self.tester = None
        self._tested_idx = None
        self.hist = []
        self.days_cts = None
        self.verbose = verbose
        if "observe_src" in kwargs and kwargs["observe_src"]:
            self._obs_source = True
        else:
            self._obs_source=False
        self.mitigate = mitigate
        self._check_epi_tests = kwargs["check_epi_tests"] if "check_epi_tests" in kwargs else False
        self.only_random = kwargs["only_random_tests"] if "only_random_tests" in kwargs else False
        self.only_symptom = only_sympt
        if self.only_random and self.only_symptom:
            raise ValueError("Cannot give both 'only random' and 'only symptomatic' tests. Decide one of them.")

        self.extra_stats_fn = kwargs["stats_extra_fn"] if "stats_extra_fn" in kwargs else None
        self._warned = defaultdict(lambda: False)

    def _init_for_sim(self, sim):
        """
        initialize the ranker
        """
        
        pars = sim.pars
        self.N = pars["pop_size"]
        T = pars["n_days"] +1
        self.ranker.init(self.N, T)
        seed = pars["rand_seed"]
        rng = np.random.RandomState(np.random.PCG64(seed))
        if self.adopt_frac < 1:
            print(f"Adoption fraction is: {self.adopt_frac}")
            self.has_app = (rng.random(self.N) < self.adopt_frac).astype(float)
            assert self.has_app.sum() < self.N
        else:
            self.has_app = np.ones(self.N)

        self.tester = CovasimTester(sim, contain=self.mitigate)
        self._warned = defaultdict(lambda: False)

        #self.iso_cts_strength
        self.extra_stats = {}

    def set_extra_stats_fn(self,func):
        """
        Set the function used to save extra statistics
        
        signature: func(sim, rank, tests)
        """
        self.extra_stats_fn = func

    def get_app_usage(self):
        if not self.initialized:
            raise ValueError("Not initialized")
        
        return self.has_app.copy()

    @staticmethod
    def _comp_flag():
        return "rksym"

    def to_json(self):
        '''
        Return JSON-compatible representation

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = {"label": self.input_args["label"], "ranker": self.ranker.__class__.__name__}
        # "n_tests":self.n_tests}
        output = dict(which=which, pars=pars)
        return output

    def __repr__(self):
        try:
            json = self.to_json()
            which = f'{self.__module__}.{self.__class__.__name__}'
            pars = json['pars']
            parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
            output = f"{which}({parstr})"
        except Exception as E:
            output = type(self) + f' (error: {str(E)})' # If that fails, print why
        return output

        
    
    def initialize(self, sim=None):
        
        if sim is None:
            self.delayed_init = True
        else:
            self._init_for_sim(sim)
        
        self.all_observ =[]
        self.daily_obs = []
        #self.ranker_data = {}
        self._tested_idx = []

        self.hist = []
        self.days_cts = []
        self.conts_prev_day = None

        super().initialize(sim)
    
    def _warn_once(self, key:str, message:str):
        if not self._warned[key]:
            warnings.warn(message)
            self._warned[key] = True

    def _set_symp_quar_prob(self, test_probs, sim, symp_p=None):
        """
        set symp prob in the test_probs array
        """
        # Symptomatics test
        if symp_p is None:
            symp_p = self.symp_test
        if callable(symp_p):
            symp_p = symp_p(sim)
        symp_probs = tuti.get_symp_probs(sim, symp_p, self.test_pdf)
        symp_inds  = cvu.true(sim.people.symptomatic)
        ## TODO: Add ili symptom prevalence
        # Handle quarantine testing
        ## THESE ARE CURRENTLY IN QUARANTINE (OR STARTING THE QUARANTINE)
        ## WE DON'T CARE
        """if self.quar_test_p is not None:
            quar_test_inds = cvi.get_quar_inds(self.quar_policy, sim)
            test_probs[quar_test_inds] = self.quar_test_p"""
        # setting the prob of symps AFTER the Q -> if someone is sympt won't be affected
        test_probs[symp_inds] = symp_probs

    
    #@staticmethod
    def draw_probs_symptom(self,sim, randgen, sympt_p):
        N = len(sim.people.age)
        test_probs = np.zeros(N)
        # Symptomatics test
        self._set_symp_quar_prob(test_probs, sim, symp_p=sympt_p)

        ## remove already diagnosed
        _remove_diagnosed_prob(test_probs, sim)
        test_inds_symp = cvu.true(
                    randgen.random(len(test_probs)) < test_probs
                )
        test_inds_symp = self.limit_symp_tests(test_inds_symp, randgen)
        
        return test_probs, test_inds_symp
    

    def apply(self, sim):
        day = sim.t
        
        if self.delayed_init:
            self._init_for_sim(sim)
            self.delayed_init = False

        USE_RANKING = (day >= self.start_day)
        """if not ACTIVE:
            ## observations from the previous day
            self.daily_obs = []"""

        day_stats = dict(day=day)
        day_stats["true_I_rk"] = 0
        N = len(sim.people.age)

        contacts_df = utils.get_contacts_day(sim.people)
        self.days_cts.append(day)

        conts_prev_day = self.conts_prev_day
        HAVE_CONTS = conts_prev_day is not None
        ## process old contacts
        if HAVE_CONTS:
            ## remove contacts for people that do not have the app
            if self.has_app.sum() < self.N:
                conts_prev_day = utils.filt_contacts_mult(
                        conts_prev_day, weight=self.has_app, N=N, only_i=False
                )
            ## remove contacts that are isolated (tested)
            tested_iso = cvu.true(sim.people.diagnosed)
            conts_prev_day = utils.filt_contacts_df(conts_prev_day,
                tested_iso, self.iso_cts_strength, N, only_i=True)
            if self.quar_factor < 1:
                ## filter the contacts for quarantined
                self._warn_once("qfact", f"Quarantine factor is {self.quar_factor}")
                quar_idcs = cvu.true(sim.people.quarantined)
                quar_andapp = (sim.people.quarantined & self.has_app.astype(bool))
                msum = conts_prev_day["m"].sum()
                conts_prev_day = utils.filt_contacts_df(conts_prev_day, quar_idcs, self.quar_factor, N)
                if sum(quar_andapp) > 0:
                    assert conts_prev_day["m"].sum() < msum
                elif len(quar_idcs)>0:
                    print(f"{len(quar_idcs)} Quar not with app")
        
        FIND_INFECTED = sim.people.infectious.sum() > 0 or sim.people.exposed.sum()>0
        FIND_INFECTED = FIND_INFECTED and HAVE_CONTS

        obs_ranker = self.tester.get_results(day-1) ## get the correct obs for the ranker
        if self.debug:
            print(f"Day {day}, active: {USE_RANKING}; n_obs: {len(obs_ranker):d}, num days conts: {len(self.days_cts)}", )

        print("Day: ", sim.t)
        ### get rank from the algorithm
        if FIND_INFECTED:
            ## transform contacts
            ### put today's date
            conts_prev_day["day"] = day
            conts_ranker = conts_prev_day[["i","j","day","m"]].to_records(index=False)

            rank_algo = self.ranker.rank(day, conts_ranker, obs_ranker, self.ranker_data)
        elif not HAVE_CONTS:
            print("No contacts to give")
            if USE_RANKING:
                print("Cannot run ranker, we don't have contacts.")
                USE_RANKING = False
        else:
            self._warn_once("epi_end","Epidemy ended, returning random ranking")
            rank_algo = list(zip(np.arange(N),np.random.rand(N)))
        
        if self._check_epi_tests and USE_RANKING:
            ### random test
            self._warn_once("check_epi","Doing random tests for DEV purpose")
            ### Disable normal intervention
            USE_RANKING = False
            self.tester._do_random_tests(sim, n_tests=(self.n_tests_algo_day+self.n_tests_rand))

        tester_rng = self.tester.randstate
        if USE_RANKING:
            
            true_inf = sim.people.infectious
            true_EI = sim.people.exposed # includes I
            idx_diagnosed = cvu.true(sim.people.diagnosed)
            ## transform ranking from algo
            rank_df = pd.DataFrame(rank_algo, columns=["idx","rank"]).set_index("idx")
            rank_proc = rank_df["rank"].sort_index()
            ### compute AUC immediately
            auc_inf, auc_EI = calc_aucs(true_inf, true_EI, rank_proc,)
            auc_i_ni, auc_ei_ni = calc_aucs(true_inf, true_EI, rank_proc,  idx_exclude=self.get_idx_obs_I())
            print("day {}: AUC_rk(I,EI): ({:4.3f},{:4.3f}), ".format(day, auc_inf, auc_EI), end="")
            ## Do rand tests
            

            true_inf_rk = 0
            accu_rk = 0
            if self.only_random:
                self._warn_once("random_tests", "Doing random tests instead of symptomatics+ranker")
                ## get random tests
                test_indcs_all = self.make_random_tests(sim, tester_rng)
                
                ## save true number of infected found
                day_stats["nt_rand"] = true_inf[test_indcs_all].sum()

            else:
                ## NO RANDOM TESTS
                test_probs = np.zeros(N)
                # Symptomatics test
                self._set_symp_quar_prob(test_probs, sim)
                ## remove already diagnosed
                _remove_diagnosed_prob(test_probs, sim)
                test_inds_symp = cvu.true(
                    tester_rng.random(len(test_probs)) < test_probs
                )
                test_inds_symp = self.limit_symp_tests(test_inds_symp, tester_rng)
            
                if self.only_symptom:
                    self._warn_once("only_sympt", "Only symptomatic testing")
                    ## Don't run algo
                    test_indcs_all = test_inds_symp
                else:
                    ### Ranker tests

                    # add rand tests indices to exclude testing
                    #test_probs[test_inds_symp] = 0.
                    ## get from rank
                    valid = set(range(N)).difference(idx_diagnosed).difference(test_inds_symp)
                    rank_good = rank_proc[list(valid)].sort_values(ascending=False)

                    if len(rank_good)==0:
                        warnings.warn("No tests from ranker, test_probs: {}".format(sum(test_probs>0)))

                    test_inds_rk = self.choose_tests_ranker(rank_good, test_inds_symp)
                    
                    ## accuracy
                    ntest_rk = len(test_inds_rk)
                    if ntest_rk > 0:
                        true_inf_rk = true_inf[test_inds_rk].sum()
                        
                        accu_rk = true_inf_rk / min(ntest_rk, true_inf.sum())
                        if self.verbose:
                            print("n_I_rk: {}, accu {:.2%}".format(true_inf_rk, accu_rk) ,
                            end=" ")
                    ### pull together the tests
                    test_indcs_all = np.concatenate((test_inds_symp, test_inds_rk))

                ## concatenate tests
                day_stats["nt_rand"] = len(test_inds_symp)
                ## END NO RANDOM TESTS
            print(f"nt_rand: {day_stats['nt_rand']}", end=" ")
            
            #print("", end=" ")
            day_stats["auc_I"] = auc_inf
            day_stats["auc_EI"] = auc_EI
            day_stats["auc_I_no"] = auc_i_ni
            day_stats["auc_EI_no"] = auc_ei_ni
            day_stats["accu_I"] = accu_rk
            day_stats["true_I_rk"] = true_inf_rk
            
            
            if self.extra_stats_fn:
                    self.extra_stats[day] = self.extra_stats_fn(sim,rank_proc,test_indcs_all)
            ### test actually
            #if not self.only_symptom:
            #    assert len(np.unique(test_indcs_all)) == self.n_tests
            #print("Num unique tests: ",len(np.unique(test_indcs_all)))

            ## stats -> check among those that I have found from symptomatic testing
        else:
            ### RANKER IS PASSIVE, only giving symptomatics
            ## If we aren't asked to give only random tests
            test_indcs_all =[]
            if not self.only_random:
                probs, test_indcs_all = self.draw_probs_symptom(sim, tester_rng, self.symp_test)
                print(f"n tests sympt: {len(test_indcs_all)}")
                day_stats["nt_rand"] = len(test_indcs_all)
            
        ## this tests the individuals today
        self.tester.run_tests(sim, test_indcs_all,
                    test_sensitivity=self.sensitivity,
                    test_specificity=self.specificity,
                    loss_prob=self.loss_prob, test_delay=self.test_delay)

        if self._obs_source:
            if self.tester._observe_sources(sim):
                self._obs_source = False

        ## find people who are tested today
        ## date_tested is the date in which the test has been required
        results_day = self.tester.get_results(day)
        ## set obs for the next day
        self.daily_obs = results_day # (idx, st, day)
        ##
        if len(results_day) > 0:
            stats_tests = np.unique(results_day[:,1], return_counts=True)
            stats = np.zeros(3,dtype=int)
            stats[stats_tests[0]] = stats_tests[1]
            #assert stats_tests[1].sum() == self.n_tests_algo_day + self.n_tests_symp
            if self.verbose: 
                print("res: ", stats, f" I: {sim.people.infectious.sum()}", end=" " )
            for s,k in enumerate(["S","I","R"]):
                day_stats["test_"+k] = stats[s]
            
            if not USE_RANKING:
                print(f"day {sim.t}")
        #elif self.verbose:
        #    print(f"no obs, t {sim.t}")
        free_birds = len(utils.check_free_birds(sim.people))
        inf_quar = (sim.people.infectious & sim.people.quarantined).sum()
        if USE_RANKING and self.verbose:
            print("free {:d}, nQ: {:d}".format(free_birds, sim.people.quarantined.sum()))
        diagnosed_today = (sim.people.date_diagnosed == sim.t) | (self.tester.date_diagnosed == sim.t)
        if callable(self.symp_test):
            day_stats["p_symp_req"] = self.symp_test(sim)    

        day_stats["n_diag"] = (sim.people.diagnosed.sum())
        #self.ranker_data["num_diagnosed"][day] = (sim.people.diagnosed.sum())
        day_stats["free_birds"] = free_birds
        day_stats["n_diag_today"] = (diagnosed_today).sum()
        day_stats["n_infectious"] = sim.people.infectious.sum()
        day_stats["n_quar"] = (sim.people.quarantined).sum()
        day_stats["n_inf_quar"] = inf_quar
        day_stats["n_inf_untested"] = (sim.people.infectious & (~sim.people.diagnosed)).sum()
        for s,k in enumerate(["S","I","R"]):
            ck = "test_"+k
            if ck not in day_stats:
                day_stats[ck] = 0
        

        self.hist.append(day_stats)
        
        ##update contacts
        self.conts_prev_day = contacts_df

    def get_idx_obs_I(self):
        test_stats = self.tester.get_all_stats()
        if len(test_stats) > 0:
            return np.unique(test_stats[test_stats["res_state"]==1]["i"])
        else:
            return []

    def prepare_export(self):
        ### delete the ranker to save memory
        self.ranker = None
        del self.ranker_data["logger"]
    
    @abstractmethod
    def choose_tests_ranker(self, ranking, test_inds_symp):
        raise NotImplementedError()

    @abstractmethod
    def limit_symp_tests(self, test_inds, randgen):
        raise NotImplementedError()

    @abstractmethod
    def make_random_tests(self, sim, randgen):
        raise NotImplementedError()

class RankTester(BaseRankTester):

    def __init__(self,
                ranker,
                label,
                num_tests,
                symp_test_p=0.5,
                quar_test_p=None,
                start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0., test_delay=0,
                iso_cts_strength = 0.1,
                quar_factor=1.,
                logger=None,
                debug=False,
                verbose=True,
                mitigate=True,
                only_sympt=False,
                adoption_fraction=1.,
                **kwargs
                ):
        super().__init__(ranker,label=label,
        symp_test_p=symp_test_p,
        quar_test_p=quar_test_p,
        start_day=start_day,
        end_day=end_day,
        sensitivity=sensitivity, specificity=specificity,
        loss_prob=loss_prob, test_delay=test_delay,
        iso_cts_strength=iso_cts_strength,
        quar_factor=quar_factor,
        logger=logger,
        debug=debug,
        verbose=verbose,
        mitigate=mitigate,
        only_sympt=only_sympt,
        adoption_fraction=adoption_fraction,
        **kwargs
        )
        self.n_tests = num_tests
    
    def limit_symp_tests(self, test_inds, randgen, ):
        if len(test_inds) > self.n_tests:
            self._warn_once("test_sympt",
                f"Symptom tests exceeded capacity: got {len(test_inds)} people to test > {self.n_tests}")
            randgen.shuffle(test_inds)
            test_inds = test_inds[:self.n_tests]
        return test_inds

    def choose_tests_ranker(self, ranking, test_inds_symp):
        n_tests_algo = self.n_tests - len(test_inds_symp)
        if n_tests_algo > 0:
            test_inds_rk = ranking.iloc[:n_tests_algo].index.to_numpy()
        else:
            test_inds_rk = np.empty((0,),dtype=np.int_)
        
        return test_inds_rk
    
    def make_random_tests(self, sim, randgen):
        return tuti.get_random_indcs_test(sim, self.n_tests, randgen)

class ProbRankTester(BaseRankTester):

    def __init__(self,
                ranker,
                label,
                p_contain,
                symp_test_p=0.5,
                quar_test_p=None,
                start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0., test_delay=0,
                iso_cts_strength = 0.1,
                quar_factor=1.,
                logger=None,
                debug=False,
                verbose=True,
                mitigate=True,
                only_sympt=False,
                adoption_fraction=1.,
                **kwargs
                ):
        super().__init__(ranker,label=label,
        symp_test_p=symp_test_p,
        quar_test_p=quar_test_p,
        start_day=start_day,
        end_day=end_day,
        sensitivity=sensitivity, specificity=specificity,
        loss_prob=loss_prob, test_delay=test_delay,
        iso_cts_strength=iso_cts_strength,
        quar_factor=quar_factor,
        logger=logger,
        debug=debug,
        verbose=verbose,
        mitigate=mitigate,
        only_sympt=only_sympt,
        adoption_fraction=adoption_fraction,
        **kwargs
        )
        self.p_contain = p_contain

    def limit_symp_tests(self, test_inds, randgen):
        ## do nothing
        return test_inds
    
    def choose_tests_ranker(self, ranking, test_inds_symp):
        
        tests_corr = ranking[ranking > self.p_contain]
        
        return tests_corr.index.to_numpy()

    def make_random_tests(self, sim, randgen):
        raise ValueError("Cannot do random tests without fixed number of tests")