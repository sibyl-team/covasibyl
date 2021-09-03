import numpy as np
import scipy.sparse as sp
import covasim.interventions as cvi
import covasim.utils as cvu
from .test import CovasimTester


class Mitigation(cvi.Intervention):
    """
    Class to run epidemic mitigation
    Give a ranker from epidemic_mitigation, and it will run the ranker and
    quarantine people
    """

    def __init__(self,
                ranker,
                label,
                n_tests, start_day=0, end_day=None,
                sensitivity=1.0, specificity:float=1.0, 
                loss_prob=0, test_delay=0,
                quar_period=None,
                notif_delay=0,
                debug=False):

        super().__init__(label="Mitigation: "+label)

        self.ranker = ranker
        self.n_daily_tests = n_tests
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.quar_period = quar_period
        self.notif_delay = notif_delay
        self.debug = debug

        self.contacts_day = None
        self.delayed_init = False
        self.N = None
        self.all_observ = None
        self.daily_obs = None
        self.ranker_data = None
        self.tester = None

    def _init_for_sim(self, sim):
        """
        initialize the ranker
        """
        if self.quar_period is None:
            self.quar_period = sim['quar_period']
        
        pars = sim.pars
        self.N = pars["pop_size"]
        T = pars["n_days"]
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


    def apply(self, sim):
        
        day = sim.t
        
        if self.delayed_init:
            self._init_for_sim(sim)
            self.delayed_init = False
        if(day < self.start_day):
            ## do nothing
            self.daily_obs = []
        if self.debug:
            print(f"Day {day} observations: ", len(self.daily_obs))
        # get contacts of the day
        conts_m = sum(
            sp.csr_matrix(
            (layer["beta"],(layer["p1"],layer["p2"])), shape=(self.N, self.N))
            for k,layer in sim.people.contacts.items()
            )
        
        conts_m = conts_m.tolil()
        conts_m.setdiag(0.)
        conts_m = conts_m.tocoo()

        contacts_day = [(i,j,day, val) for i,j,val in zip(conts_m.row, conts_m.col, conts_m.data)]
        
        ### get rank from the algorithm
        rank_algo = self.ranker.rank(day, contacts_day, self.daily_obs, self.ranker_data)
        rank = np.array(sorted(rank_algo, key= lambda tup: tup[1], reverse=True))
        rank = rank[:,0].astype(int)

        test_inds = rank[:self.n_daily_tests]
        ## test people
        ## not using sim.people.test because doesn't record the susceptible tests
        #sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)
        
        self.tester.test(sim, test_inds,
                    test_sensitivity=self.sensitivity,
                    test_specificity=self.specificity,
                    loss_prob=self.loss_prob, test_delay=self.test_delay)

        ## find people who are tested today
        ## date_tested is the date in which the test has been required
        results_day = self.tester.get_results(day)
        
            #print("test results ",results_day)
        ## change obs
        self.daily_obs = [(idx, st, day) for idx, st in results_day]

        ## quarantine individuals
        inds_quar = cvu.true(sim.people.date_diagnosed == sim.t)
        sim.people.schedule_quarantine(inds_quar,
                    start_date=sim.t + self.notif_delay,
                    period=self.quar_period - self.notif_delay)