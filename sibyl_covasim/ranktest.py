import numpy as np
import pandas as pd
import sciris as sc

from covasim.interventions import Intervention
import covasim.utils as cvu
import covasim.defaults as cvd

from .test import CovasimTester
from . import utils


class RankTester(Intervention):
    """
    Class to test people with information on
    both symptomatic, quarantined and using the ranker

    remember people are quarantined before being tested
    """

    def __init__(self,
                ranker,
                label,
                num_tests_algo,
                num_tests_symp=100,
                symp_test=10.,
                start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0., test_delay=0,
                swab_delay=None,
                debug=False):

        super().__init__(label="Mitigation: "+label)

        self.ranker = ranker
        self.n_tests_algo_day = num_tests_algo
        self.n_tests_symp = num_tests_symp
        self.symp_test = symp_test
        self.sensitivity = sensitivity
        self.specificity = specificity
        # probability of losing a test result
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
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
            rank_df = pd.DataFrame(rank_algo, columns=["idx","rank"]).set_index("idx")
            rank = rank_df["rank"].sort_index()