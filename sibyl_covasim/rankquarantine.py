import numpy as np
import sciris as sc
from sklearn.metrics import roc_curve, auc

import covasim.interventions as cvi
import covasim.utils as cvu
import covasim.defaults as cvd
from .tester import CovasimTester
from .utils import get_contacts_day

import pandas as pd


def check_free_birds(people):
    free_idx = (people.infectious & np.logical_not(people.quarantined)).nonzero()[0]
    return free_idx

class Mitigation(cvi.Intervention):
    """
    Class to run epidemic mitigation
    Give a ranker from epidemic_mitigation, and it will run the ranker and
    quarantine people

    ranker:         the intervention ranker
    label: 
    num_tests:      Number of tests performed daily
    symp_test   (float) : odds ratio of a symptomatic person testing (default: 100x more likely) 
    start_day:      Starting day of the tracing intervention
    end_day:        End day of tracing intervention
    sensitivity (float) : of tests
    specificity (float) : of tests
    loss_prob:      Probability of losing a test result
    test_delay:     Number of days until availability of test result
    quar_period:    Number of days to quarantine infected people
    notif_delay:    Delay in relaying the test result (affecting quarantine)
    swab_delay  (dict) : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
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
                quar_period=None,
                notif_delay=0,
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
        self.quar_period = quar_period
        self.notif_delay = notif_delay
        # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self.swab_pdf = cvu.get_pdf(**sc.mergedicts(swab_delay)) 
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

    def _init_for_sim(self, sim):
        """
        initialize the ranker
        """
        if self.quar_period is None:
            self.quar_period = sim['quar_period']
            print("Quarantine period: ", self.quar_period)
        
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
        self.ranker_data = {"num_diagnosed":{},
        "num_diagnosed_day":{}}
        self._tested_idx = []

        self.hist = []
        self.days_conts = []

    def _prob_test_sympt(self, sim, t, test_probs=None):
        """
        Default way of testing symptomatic individuals
        """
        # Calculate test probabilities for people with symptoms
        N = len(sim.people.age)
        symp_inds = cvu.true(sim.people.symptomatic)
        if test_probs is None:
            test_probs = np.ones(N)
        symp_test = self.symp_test
        if self.swab_pdf: # Handle the onset to swab delay
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) # Initialize the count
            count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
            symp_test *= self.swab_pdf.pdf(symp_time) * count[symp_time] # Put it all together
            ### comment on above: basically we use the probability from pdf, but
            ### then divide by the empirical frequency (number of people with delay =d / tot number of symptomatics)
            ### This way, the less frequent delays are more likely to be chosen

        test_probs[symp_inds] *= symp_test # Update the test probabilities
        return test_probs

    def _apply_quarantine(self, sim):
        """
        Put the people tested positive on this day in quarantine
        """
        diagnosed_today = (sim.people.date_diagnosed == sim.t) | (self.tester.date_diagnosed == sim.t)
        inds_quar = cvu.true(diagnosed_today)
        ### WRONG: people will be in quarantine BEFORE being tested
        sim.people.schedule_quarantine(inds_quar,
                        start_date=sim.t + self.notif_delay,
                        period=self.quar_period - self.notif_delay)
        return inds_quar

    def apply(self, sim):
        
        day = sim.t
        
        if self.delayed_init:
            self._init_for_sim(sim)
            self.delayed_init = False
        if(day < self.start_day):
            ## do nothing
            self.daily_obs = []
        
        # get contacts of the day
        conts_m = get_contacts_day(sim.people)
        conts_m["day"] = day
        self.days_conts.append(day)

        if self.debug:
            print(f"Day {day}, active: {day >= self.start_day}; n_obs: {len(self.daily_obs):d}, num days conts: {len(self.days_conts)}", )
        #contacts_day = [(i,j,day, val) for i,j,val in zip(conts_m.row, conts_m.col, conts_m.data)]
        contacts_day = conts_m[["i","j","day","m"]].to_records(index=False)
        
        ### get rank from the algorithm
        rank_algo = self.ranker.rank(day, contacts_day, self.daily_obs, self.ranker_data)
        #rank = np.array(sorted(rank_algo, key= lambda tup: tup[1], reverse=True))
        #rank_idx = rank[:,0].astype(int)
        
        day_stats = dict(day=day)
        if day >= self.start_day:
            rank_df = pd.DataFrame(rank_algo, columns=["idx","rank"]).set_index("idx")
            rank = rank_df["rank"].sort_index()
            ## remove already diagnosed
            already_diagnosed = (sim.people.date_diagnosed < day) | (self.tester.date_diagnosed < day)
            
            true_inf = sim.people.infectious
            idx_diagnosed = cvu.true(already_diagnosed)
            ### test first for symptomatics
            test_symp_probs = self._prob_test_sympt(sim=sim, t=day,)
            test_symp_probs[idx_diagnosed] = 0.
            inds_symps = cvu.choose_w(probs=test_symp_probs, n=self.n_tests_symp, unique=True) 

            already_diagnosed[inds_symps] =  True
            ## test people with rankers
            n_test_algo = self.n_tests_algo_day
            rank_good = rank[np.logical_not(already_diagnosed)].sort_values(ascending=False)
            test_inds = rank_good[:n_test_algo].index.to_numpy()

            num_already_t = len(set(idx_diagnosed).intersection(test_inds))
            num_chosen_both = len(set(inds_symps).intersection(
                rank.sort_values(ascending=False)[:n_test_algo].index.to_numpy() ) )

            ## accuracy
        
            real_inf = true_inf[test_inds].sum()
            if real_inf > 0:
                fpr, tpr, _ = roc_curve(true_inf[test_inds], rank[test_inds].to_numpy())
                auc_inf = auc(fpr,tpr)  #if real_inf > 0 else np.nan
            else:
                auc_inf = np.nan
            print("day {}: AUC_I: {:4.3f}, accu {:.2%}, n_retest {}".format(
                day,auc_inf, real_inf/self.n_tests_algo_day, num_already_t) ,
                end=" ")

            
            ## test people
            ## not using sim.people.test because doesn't record the susceptible tests

            test_inds = np.concatenate((test_inds,inds_symps))
            assert len(np.unique(test_inds)) == self.n_tests_algo_day + self.n_tests_symp        
            self.tester.apply_tests(sim, test_inds,
                        test_sensitivity=self.sensitivity,
                        test_specificity=self.specificity,
                        loss_prob=self.loss_prob, test_delay=self.test_delay)

            ## find people who are tested today
            ## date_tested is the date in which the test has been required
            results_day = self.tester.get_results(day)
            
            #print("test results ",results_day)
            ## change obs
            self.daily_obs = [(idx, st, day) for idx, st in results_day]
            
            stats_tests = np.unique(results_day[:,1], return_counts=True)
            stats = np.zeros(3,dtype=int)
            stats[stats_tests[0]] = stats_tests[1]
            assert stats_tests[1].sum() == self.n_tests_algo_day + self.n_tests_symp
            print("res: ", stats, f" I: {sim.people.infectious.sum()}", end=" " )
            for s,k in enumerate(["S","I","R"]):
                day_stats["test_"+k] = stats[s]
            #print(self.daily_obs)

            ## quarantine individuals
            inds_quar = self._apply_quarantine(sim)
            print("Q TOT: {:d}, today: {:d}".format(sum(sim.people.quarantined), len(inds_quar)),end=" ")

        
        free_birds = len(check_free_birds(sim.people))
        if day >= self.start_day:
            print("free {:d}".format(free_birds))
        diagnosed_today = (sim.people.date_diagnosed == sim.t) | (self.tester.date_diagnosed == sim.t)

        day_stats["num_diagnosed"] = (sim.people.diagnosed.sum())
        #self.ranker_data["num_diagnosed"][day] = (sim.people.diagnosed.sum())
        day_stats["free_birds"] = free_birds
        day_stats["num_diagnosed_day"] = (diagnosed_today).sum()
        day_stats["n_infectious"] = sim.people.infectious.sum()


        self.hist.append(day_stats)