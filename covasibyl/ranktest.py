from cmath import log
from tabnanny import verbose
import warnings
import numpy as np
import pandas as pd
import sciris as sc

from sklearn.metrics import roc_curve, auc

import covasim.interventions as cvi
import covasim.utils as cvu
import covasim.defaults as cvd

from .tester import CovasimTester, get_default_test_p_sym
from . import utils



class RankTester(cvi.Intervention):
    """
    Class to test people with information on
    both symptomatic, quarantined and using the ranker

    remember people are quarantined before being tested
    """

    def __init__(self,
                ranker,
                label,
                num_tests_algo,
                num_tests_rand=100,
                symp_test=10.,
                quar_test=1.,
                start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0., test_delay=0,
                swab_delay=None,
                quar_policy=None,
                iso_cts_strength = 0.1,
                logger=None,
                debug=False,
                verbose=True,
                mitigate=True,
                **kwargs
                ):

        super().__init__(label="Mitigation: "+label)

        self.ranker = ranker
        self.n_tests_algo_day = num_tests_algo
        self.n_tests_rand = num_tests_rand
        self.symp_test = symp_test # Probability of testing symptomatics
        self.sensitivity = sensitivity
        self.specificity = specificity
        # probability of losing a test result
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.quar_test   = quar_test # Probability of testing people in quarantine
        self.quar_policy = quar_policy if quar_policy else 'start'
        # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self.test_pdf = cvu.get_pdf(**sc.mergedicts(swab_delay))
        # multiply contacts strength by this factor for isolated individuals
        self.iso_cts_strength = iso_cts_strength
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

    def _init_for_sim(self, sim):
        """
        initialize the ranker
        """
        
        pars = sim.pars
        self.N = pars["pop_size"]
        T = pars["n_days"] +1
        self.ranker.init(self.N, T)

        self.tester = CovasimTester(sim, contain=self.mitigate)

    def to_json(self):
        '''
        Return JSON-compatible representation

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = {"label": self.input_args["label"], "ranker": self.ranker.__class__.__name__,
        "n_random": self.n_tests_rand, "n_algo": self.n_tests_algo_day}
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
        super().initialize(sim)
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

    def apply(self, sim):
        day = sim.t
        
        if self.delayed_init:
            self._init_for_sim(sim)
            self.delayed_init = False

        ACTIVE = (day >= self.start_day)
        """if not ACTIVE:
            ## observations from the previous day
            self.daily_obs = []"""

        obs_day = self.daily_obs
        day_stats = dict(day=day)
        day_stats["true_I_rk"] = 0
        N = len(sim.people.age)

        conts_m = utils.get_contacts_day(sim.people)
        ## remove contacts that are isolated (tested)
        tested_iso = cvu.true(sim.people.diagnosed)
        conts_m = utils.filt_contacts_df(conts_m, tested_iso, self.iso_cts_strength, N)

        conts_m["day"] = day
        self.days_cts.append(day)

        if self.debug:
            print(f"Day {day}, active: {ACTIVE}; n_obs: {len(obs_day):d}, num days conts: {len(self.days_cts)}", )
        #contacts_day = [(i,j,day, val) for i,j,val in zip(conts_m.row, conts_m.col, conts_m.data)]
        contacts_day = conts_m[["i","j","day","m"]].to_records(index=False)

        FIND_INFECTED = sim.people.infectious.sum() > 0 or sim.people.exposed.sum()>0

        ### get rank from the algorithm
        if FIND_INFECTED:
            rank_algo = self.ranker.rank(day, contacts_day, obs_day, self.ranker_data)
        else:
            warnings.warn("Epidemy ended, returning random ranking")
            rank_algo = list(zip(np.arange(N),np.random.rand(N)))

        if ACTIVE:
            test_probs = np.ones(N)
            # Symptomatics test
            get_default_test_p_sym(sim, sim.t, self.symp_test,test_pdf=self.test_pdf,
                test_probs=test_probs
            )
            # Handle quarantine testing
            quar_test_inds = cvi.get_quar_inds(self.quar_policy, sim)
            test_probs[quar_test_inds] *= self.quar_test

            ## remove already diagnosed
            already_diagnosed = sim.people.diagnosed #| np.logical_not(np.isnan(self.tester.date_diagnosed))
            true_inf = sim.people.infectious
            idx_diagnosed = cvu.true(already_diagnosed)
            test_probs[idx_diagnosed] = 0.
            
            ## Do rand tests
            if self.n_tests_rand > (test_probs!=0).sum():
                ### This shouldn't happen, but let's check
                warnings.warn("Number of tests higher than probability sum")
            n_tests_rnd = min(self.n_tests_rand, (test_probs!=0).sum())# Don't try to test more people than have nonzero testing probability
            test_inds_rnd = utils.choose_w_rng(probs=test_probs, n=n_tests_rnd, unique=True) # Choose who actually tests

            ### Ranker tests
            rank_df = pd.DataFrame(rank_algo, columns=["idx","rank"]).set_index("idx")
            rank = rank_df["rank"].sort_index()
            ### compute AUC immediately
            fpr, tpr, _ = roc_curve(true_inf, rank.to_numpy())
            auc_inf = auc(fpr,tpr)  #if real_inf > 0 else np.nan
            # add rand tests indices to exclude testing
            test_probs[test_inds_rnd] = 0.
            ## get from rank
            rank_good = rank[(test_probs>0)].sort_values(ascending=False)
 
            if len(rank_good)==0:
                warnings.warn("No tests from ranker, test_probs: {}".format(sum(test_probs>0)))
            test_inds = rank_good[:self.n_tests_algo_day].index.to_numpy()
            ## accuracy
            
            true_inf_rk = true_inf[test_inds].sum()
           
            

            accu = true_inf_rk / min(len(test_inds), true_inf.sum())
            if self.verbose:
                print("day {}: AUC_I_rk: {:4.3f}, n_I_rk: {}, accu {:.2%}".format(
                day,auc_inf, true_inf_rk, accu) ,
                end=" ")
            #print("", end=" ")
            day_stats["auc_I"] = auc_inf
            day_stats["accu_I"] = accu
            day_stats["true_I_rk"] = true_inf_rk

            ### test actually
            test_indcs_all = np.concatenate((test_inds_rnd, test_inds))
            assert len(np.unique(test_indcs_all)) == self.n_tests_algo_day + self.n_tests_rand        
            self.tester.run_tests(sim, test_indcs_all,
                        test_sensitivity=self.sensitivity,
                        test_specificity=self.specificity,
                        loss_prob=self.loss_prob, test_delay=self.test_delay)

            ## stats -> check among those that I have found from symptomatic testing

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
        #elif self.verbose:
        #    print(f"no obs, t {sim.t}")
        if not ACTIVE:
            print(f"day {sim.t}")

        free_birds = len(utils.check_free_birds(sim.people))
        inf_quar = (sim.people.infectious & sim.people.quarantined).sum()
        if ACTIVE and self.verbose:
            print("free {:d}".format(free_birds))
        diagnosed_today = (sim.people.date_diagnosed == sim.t) | (self.tester.date_diagnosed == sim.t)

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

    def prepare_export(self):
        ### delete the ranker to save memory
        self.ranker = None
        del self.ranker_data["logger"]