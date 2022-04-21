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
                debug=False):

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
        self.debug = debug

        self.contacts_day = None
        self.delayed_init = False
        self.N = None
        self.all_observ = None
        self.daily_obs = None
        self.ranker_data = None
        self.tester = None
        self._tested_idx = None
        self.hist = []
        self.days_cts = None

    def _init_for_sim(self, sim):
        """
        initialize the ranker
        """
        
        pars = sim.pars
        self.N = pars["pop_size"]
        T = pars["n_days"] +1
        self.ranker.init(self.N, T)

        self.tester = CovasimTester(sim)

        
    
    def initialize(self, sim=None):
        super().initialize(sim)
        if sim is None:
            self.delayed_init = True
        else:
            self._init_for_sim(sim)
        
        self.all_observ =[]
        self.daily_obs = []
        self.ranker_data = {}
        self._tested_idx = []

        self.hist = []
        self.days_cts = []

    def apply(self, sim):
        day = sim.t
        
        if self.delayed_init:
            self._init_for_sim(sim)
            self.delayed_init = False

        ACTIVE = (day >= self.start_day)
        if not ACTIVE:
            ## observations from the previous day
            self.daily_obs = []

        obs_day = self.daily_obs
        day_stats = dict(day=day)
        N = len(sim.people.age)

        conts_m = utils.get_contacts_day(sim.people)
        conts_m["day"] = day
        self.days_cts.append(day)

        if self.debug:
            print(f"Day {day}, active: {ACTIVE}; n_obs: {len(obs_day):d}, num days conts: {len(self.days_conts)}", )
        #contacts_day = [(i,j,day, val) for i,j,val in zip(conts_m.row, conts_m.col, conts_m.data)]
        contacts_day = conts_m[["i","j","day","m"]].to_records(index=False)

        ### get rank from the algorithm
        rank_algo = self.ranker.rank(day, contacts_day, obs_day, self.ranker_data)

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
            already_diagnosed = sim.people.diagnosed | np.logical_not(np.isnan(self.tester.date_diagnosed))
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
            # add rand tests indices to exclude testing
            test_probs[test_inds_rnd] = 0.
            ## get from rank
            rank_good = rank[(test_probs>0)].sort_values(ascending=False)
            if len(rank_good==0):
                warnings.warn("No tests from ranker")
            test_inds = rank_good[:self.n_tests_algo_day].index.to_numpy()
            ## accuracy
            true_inf = sim.people.infectious
            real_inf = true_inf[test_inds].sum()
            if real_inf > 0:
                fpr, tpr, _ = roc_curve(true_inf[test_inds], rank[test_inds].to_numpy())
                auc_inf = auc(fpr,tpr)  #if real_inf > 0 else np.nan
            else:
                auc_inf = np.nan
            print("day {}: AUC_I: {:4.3f}, accu {:.2%}".format(
                day,auc_inf, real_inf/self.n_tests_algo_day) ,
                end=" ")

            ### test actually
            test_indcs_all = np.concatenate((test_inds_rnd, test_inds))
            assert len(np.unique(test_indcs_all)) == self.n_tests_algo_day + self.n_tests_rand        
            self.tester.apply_tests(sim, test_indcs_all,
                        test_sensitivity=self.sensitivity,
                        test_specificity=self.specificity,
                        loss_prob=self.loss_prob, test_delay=self.test_delay)

            ## find people who are tested today
            ## date_tested is the date in which the test has been required
            results_day = self.tester.get_results(day)
            ## set obs for the next day
            self.daily_obs = [(idx, st, day) for idx, st in results_day]

            stats_tests = np.unique(results_day[:,1], return_counts=True)
            stats = np.zeros(3,dtype=int)
            stats[stats_tests[0]] = stats_tests[1]
            assert stats_tests[1].sum() == self.n_tests_algo_day + self.n_tests_symp
            print("res: ", stats, f" I: {sim.people.infectious.sum()}", end=" " )
            for s,k in enumerate(["S","I","R"]):
                day_stats["test_"+k] = stats[s]

        free_birds = len(utils.check_free_birds(sim.people))
        if ACTIVE:
            print("free {:d}".format(free_birds))
        diagnosed_today = (sim.people.date_diagnosed == sim.t) | (self.tester.date_diagnosed == sim.t)

        day_stats["num_diagnosed"] = (sim.people.diagnosed.sum())
        #self.ranker_data["num_diagnosed"][day] = (sim.people.diagnosed.sum())
        day_stats["free_birds"] = free_birds
        day_stats["num_diagnosed_day"] = (diagnosed_today).sum()
        day_stats["n_infectious"] = sim.people.infectious.sum()