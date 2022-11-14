from collections import defaultdict
import warnings
import numpy as np

import sciris as sc

import covasim.defaults as cvd
import covasim.utils as cvu

from covasim.interventions import (Intervention, preprocess_day, process_daily_data, get_day,
                get_quar_inds)

from .tester import CovasimTester
from .utils import choose_w_rng, randround_rng
from .test_utils  import find_ili_inds, get_symp_probs, make_symp_probs_covasim_def, possible_find_infected

def choose_probs(test_probs, randgen):
    return cvu.true(
                    randgen.random(len(test_probs)) < test_probs
                )

class TestProbNum(Intervention):
    '''
    Test the specified number of people per day. Useful for including historical
    testing data. The probability of a given person getting a test is dependent
    on the total number of tests, population size, and odds ratios. Compare this
    intervention with cv.test_prob().

    IMPORTANT: people who are quarantined, doesn't matter if they are symptomatic or not

    Args:
        daily_tests (arr)   : number of tests per day, can be int, array, or dataframe/series; if integer, use that number every day; if 'data' or another string, use loaded data
        symp_test   (float) : odds ratio of a symptomatic person testing (default: 80x more likely)
        quar_test   (float) : probability of a person in quarantine testing (default: 100x more likely)
        quar_policy (str)   : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget   (dict)  : subtarget intervention to people with particular indices (format: {'ind': array of indices, or function to return indices from the sim, 'vals': value(s) to apply}
        ili_prev    (arr)   : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity (float) : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob   (float) : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay  (int)   : days for test result to be known (default 0, i.e. results available instantly)
        start_day   (int)   : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day     (int)   : day the intervention ends
        swab_delay  (dict)  : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
        kwargs      (dict)  : passed to Intervention()

    **Examples**::

        interv = cv.test_num(daily_tests=[0.10*n_people]*npts)
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests=[0.10*n_people]*npts, subtarget={'inds': lambda sim: cv.true(sim.people.age>50), 'vals': 1.2}) # People over 50 are 20% more likely to test
        interv = cv.test_num(daily_tests='data') # Take number of tests from loaded data using default column name (new_tests)
        interv = cv.test_num(daily_tests='swabs_per_day') # Take number of tests from loaded data using a custom column name
    '''

    def __init__(self, daily_tests, symp_test_p=0.5, quar_test=100.0, quar_policy=None, subtarget=None,
                 ili_prev=None, sensitivity=1.0, specificity=1.0, loss_prob=0, test_delay=0, contain=True,
                 start_day=0, end_day=None, swab_delay=None, init_sympt=False, 
                 nfixed_rand_tests=0, save_test_probs=False, no_rnd_tests=False,no_testing=False,**kwargs):
        
        
        super().__init__(**kwargs) # Initialize the Intervention object
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test_p   = symp_test_p   # Set probability of testing symptomatics
        self.quar_test   = quar_test # Probability of testing people in quarantine
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget  # Set any other testing criteria
        self.ili_prev    = ili_prev     # Should be a list of length matching time or a float or a dataframe
        self.sensitivity = sensitivity
        self.specifity = specificity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.init_sympt  = init_sympt
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        
        self.mtester = None
        self.mitigate = contain
        self.nfixed_rand_tests = nfixed_rand_tests
        self.save_test_probs = save_test_probs
        self.no_rnd_tests = no_rnd_tests
        self.no_testing = no_testing
        self.tested_idcs_rnd = None
        self.hist = None
        self._warned = None

    def _warn_once(self, key:str, message:str):
        if not self._warned[key]:
            warnings.warn(message)
            self._warned[key] = True

    def initialize(self, sim):
        ''' Fix the dates and number of tests '''

        # Handle days
        super().initialize()

        self.start_day   = preprocess_day(self.start_day, sim)
        self.end_day     = preprocess_day(self.end_day,   sim)
        self.days        = [self.start_day, self.end_day]

        # Process daily data
        self.daily_tests = process_daily_data(self.daily_tests, sim, self.start_day)
        self.ili_prev    = process_daily_data(self.ili_prev,    sim, self.start_day)

        self.mtester = CovasimTester(sim, contain=self.mitigate)
        self.extra_rng = np.random.RandomState(np.random.PCG64(3))

        self.hist =[]
        self.tested_idcs_rnd = {}
        self.test_probs_rand = {}

        self._warned = defaultdict(lambda: False)
        
        return
    def _make_symp_probs_all(self,sim, start_day):
        return make_symp_probs_covasim_def(sim, start_day, self.symp_test_p,
        pdf=self.pdf, ili_prev=self.ili_prev, rng=self.extra_rng)

    def _run_tests_def(self, sim, test_inds):
        ### Helper function to shorten the testing
        t = sim.t
        self.mtester.run_tests(sim, test_inds, test_sensitivity=self.sensitivity,
                test_specificity=self.specifity, loss_prob=self.loss_prob, test_delay=self.test_delay)


    def apply(self, sim):

        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)

        tester_rng = self.mtester.randstate
        day_stats = dict(day=t)
        day_stats["true_I_rk"] = 0
        day_stats["n_infectious"] = sim.people.infectious.sum()

        if end_day is not None and t > end_day:
            return

        # Check that there are still tests
        rel_t = t - start_day
        if rel_t < len(self.daily_tests):
            if sim.rescale_vec[t] != 1:
                self._warn_once("rescale_tests", f"Rescaling number of tests because of rescale_vec: {sim.rescale_vec[t:]}")
                # Correct for scaling that may be applied by rounding to the nearest number of tests
                n_tests_all = randround_rng(self.daily_tests[rel_t]/sim.rescale_vec[t], self.extra_rng)
            else:
                n_tests_all = self.daily_tests[rel_t]
            if not (n_tests_all and np.isfinite(n_tests_all)): # If there are no tests today, abort early
                return
            #else:
            #    sim.results['new_tests'][t] += n_tests_all
        else:
            return
        # With dynamic rescaling, we have to correct for uninfected people outside of the population who would test
        if sim.rescale_vec[t]/sim['pop_scale'] < 1: # We still have rescaling to do
            self._warn_once("rescale_probs", "Rescaling test probability with randround")
            in_pop_tot_prob = test_probs.sum()*sim.rescale_vec[t] # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = sim.scaled_pop_size - sim.rescale_vec[t]*sim['pop_size'] # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests_all = randround_rng(n_tests_all*in_frac, self.extra_rng) # Recompute the number of tests

        if t < start_day:
            if self.init_sympt:
                ## No intervention yet, observe only symptomatics
                test_probs = self._make_symp_probs_all(sim, start_day)
                ##DEBUGGING
                #print(f"{t} pn0: {test_probs[test_probs>0]} in0: {cvu.true(test_probs>0)}")

                #print(f"{t}: Randstate: {tester_rng._bit_generator.state['state']}")

                test_inds_sym = choose_probs(test_probs,tester_rng)
                ## Limit symptomatic tests
                if len(test_inds_sym) > n_tests_all:
                    tester_rng.shuffle(test_inds_sym)
                    test_inds_sym = test_inds_sym[:n_tests_all]

                ### Test
                if not self.no_testing:
                    self._run_tests_def(sim, test_inds_sym)

                ntrue_I = len(cvu.itruei(sim.people.symptomatic, test_inds_sym))
                day_stats['nI_symp'] = ntrue_I
                self.hist.append(day_stats)
            return

        FIND_INFECT = possible_find_infected(sim)
        ## test for symptomatics first
        # Calculate test probabilities for people with symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        
        # Define symptomatics, accounting for ILI prevalence
        ili_inds = find_ili_inds(sim, self.ili_prev, symp_inds, start_day)

        test_probs = self._make_symp_probs_all(sim, start_day)
        #print(f"Symp probs: {symp_prob}, nposstest={(test_probs > 0).sum()}")

        #test_inds_sym = cvu.true(cvu.binomial_arr(test_probs))
        if FIND_INFECT:
            test_inds_sym = choose_probs(test_probs,tester_rng)
            if len(test_inds_sym) > n_tests_all:
                tester_rng.shuffle(test_inds_sym)
                test_inds_sym = test_inds_sym[:n_tests_all]
        else:
            print("No infect to find")
            test_inds_sym = np.empty((0,), dtype=np.int_)

        
        ntests_rand = n_tests_all - len(test_inds_sym)
        ntrue_I = len(cvu.itruei(sim.people.symptomatic, test_inds_sym))
        inf_diag = (sim.people.symptomatic & sim.people.diagnosed)
        #print(f"day {sim.t}, test inf: {ntrue_I} tot inf free: {sim.people.symptomatic.sum() - inf_diag.sum()}")
        test_inds = test_inds_sym
        day_stats["nI_symp"] = ntrue_I

        if ntests_rand > 0 and FIND_INFECT:
            ## do random tests
            test_probs_rnd = np.ones(sim.n) # Begin by assigning equal testing weight (converted to a probability) to everyone
            diag_inds = cvu.true(sim.people.diagnosed)
            symp_prob = get_symp_probs(sim, self.symp_test_p, self.pdf)
            # Handle quarantine testing
            quar_test_inds = get_quar_inds(self.quar_policy, sim)
            test_probs_rnd[quar_test_inds] *= self.quar_test

            test_probs_rnd[symp_inds]       = (symp_prob/self.symp_test_p)*80*(self.quar_test/144)          # People with symptoms (true positive)
            test_probs_rnd[ili_inds]        = 80*(self.quar_test/144)
            # Don't rediagnose people
            test_probs_rnd[diag_inds] = 0.
            test_probs_rnd[test_inds_sym] = 0.
            
            ntp = (test_probs_rnd!=0).sum()
            if ntp < ntests_rand:
               print(f"Can only test {ntp} people, asking for {ntests_rand}") 
               ntests_rand = ntp
            
            ntrueI_rand = 0 ## we'll overwrite this later
            if self.nfixed_rand_tests>0:
                self._warn_once("test_save",f"Saving {self.nfixed_rand_tests} tests per day")
                ## fake extraction
                inds_test_rnd = choose_w_rng(probs=test_probs_rnd, n=self.nfixed_rand_tests, unique=True,
                                        rng=tester_rng)
            else:
                inds_test_rnd = choose_w_rng(probs=test_probs_rnd, n=ntests_rand, unique=True,
                                        rng=tester_rng)
            
            #cvu.choose_w(probs=test_probs_rnd, n=ntests_rand, unique=True)
            if not self.no_rnd_tests:
                test_inds = np.concatenate((test_inds, inds_test_rnd))
                ntrueI_rand = len(cvu.itruei(sim.people.infectious, inds_test_rnd))
            else:
                self._warn_once("test_norand", "No random testing performed")

            if self.save_test_probs:
                self.test_probs_rand[sim.t] = test_probs_rnd

            day_stats["nI_ttq"] = ntrueI_rand

            self.tested_idcs_rnd[t] = inds_test_rnd

        
        # Run tests
        if not self.no_testing:
            self._run_tests_def(sim, test_inds)
    
    
        self.hist.append(day_stats)

        return test_inds