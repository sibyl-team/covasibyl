
import numpy as np
import sciris as sc
import covasim.utils as cvu
import covasim.defaults as cvd

from covasim.interventions import (Intervention, preprocess_day, process_daily_data, get_day,
                get_quar_inds, get_subtargets)

class TestProb(Intervention):
    '''
    Assign each person a probability of being tested for COVID based on their
    symptom state, quarantine state, and other states. Unlike test_num, the
    total number of tests not specified, but rather is an output.

    Args:
        symp_prob        (float)     : probability of testing a symptomatic (unquarantined) person
        asymp_prob       (float)     : probability of testing an asymptomatic (unquarantined) person (default: 0)
        symp_quar_prob   (float)     : probability of testing a symptomatic quarantined person (default: same as symp_prob)
        asymp_quar_prob  (float)     : probability of testing an asymptomatic quarantined person (default: same as asymp_prob)
        quar_policy      (str)       : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget        (dict)      : subtarget intervention to people with particular indices  (see test_num() for details)
        ili_prev         (float/arr) : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        sensitivity      (float)     : test sensitivity (default 100%, i.e. no false negatives)
        loss_prob        (float)     : probability of the person being lost-to-follow-up (default 0%, i.e. no one lost to follow-up)
        test_delay       (int)       : days for test result to be known (default 0, i.e. results available instantly)
        start_day        (int)       : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day          (int)       : day the intervention ends (default: no end)
        swab_delay       (dict)      : distribution for the delay from onset to swab; if this is present, it is used instead of test_delay
        kwargs           (dict)      : passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, ili_prev=None,
                 sensitivity=1.0, loss_prob=0.0, test_delay=0, start_day=0, end_day=None, swab_delay=None, exp_as_I=False,
                  **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.quar_policy      = quar_policy if quar_policy else 'start'
        self.subtarget        = subtarget
        self.ili_prev         = ili_prev
        self.sensitivity      = sensitivity
        self.loss_prob        = loss_prob
        self.test_delay       = test_delay
        self.start_day        = start_day
        self.end_day          = end_day
        self.pdf              = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self.exp_as_I = exp_as_I
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        super().initialize()
        self.start_day = preprocess_day(self.start_day, sim)
        self.end_day   = preprocess_day(self.end_day,   sim)
        self.days      = [self.start_day, self.end_day]
        self.ili_prev  = process_daily_data(self.ili_prev, sim, self.start_day)
        return


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def apply(self, sim):
        ''' Perform testing '''

        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        # Find probablity for symptomatics to be tested
        symp_inds  = cvu.true(sim.people.symptomatic)
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape)
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = self.pdf.pdf(symp_time) * symp_prob * count[symp_time]

        # Define symptomatics, accounting for ILI prevalence
        pop_size = sim['pop_size']
        ili_inds = []
        if self.ili_prev is not None:
            rel_t = t - start_day
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * pop_size)  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(pop_size, n_ili) # Give some people some symptoms, assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.setdiff1d(np.arange(pop_size), symp_inds), ili_inds)

        # Handle quarantine and other testing criteria
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        symp_quar_inds  = np.intersect1d(quar_test_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_test_inds, asymp_inds)
        diag_inds       = cvu.true(sim.people.diagnosed)

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim['pop_size']) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = symp_prob            # People with symptoms (true positive)
        test_probs[ili_inds]        = self.symp_prob       # People with symptoms (false positive) -- can't use swab delay since no date symptomatic
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.true(cvu.binomial_arr(test_probs)) # Finally, calculate who actually tests

        # Actually test people
        test_people(sim.people, test_inds, test_sensitivity=self.sensitivity,
            loss_prob=self.loss_prob, test_delay=self.test_delay,
            exp_as_I=self.exp_as_I) # Actually test people
        sim.results['new_tests'][t] += len(test_inds)*sim['pop_scale']/sim.rescale_vec[t] # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return test_inds

def test_people(people, inds, test_sensitivity=1.0, loss_prob=0.0, test_delay=0, exp_as_I=False):
    '''
    Method to test people. Typically not to be called by the user directly;
    see the test_num() and test_prob() interventions.

    Args:
        inds: indices of who to test
        test_sensitivity (float): probability of a true positive
        loss_prob (float): probability of loss to follow-up
        test_delay (int): number of days before test results are ready
    '''

    inds = np.unique(inds)
    people.tested[inds] = True
    people.date_tested[inds] = people.t # Only keep the last time they tested

    is_infectious = cvu.itruei(people.infectious, inds)
    if exp_as_I:
        is_exp_inf=cvu.itruei((people.exposed &(~people.infectious)), inds)
        is_infectious = np.concatenate((is_infectious,is_exp_inf))
        assert len(np.unique(is_infectious)) == len(is_infectious)
    pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
    is_inf_pos    = is_infectious[pos_test]

    not_diagnosed = is_inf_pos[np.isnan(people.date_diagnosed[is_inf_pos])]
    not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
    final_inds    = not_diagnosed[not_lost]

    # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
    people.date_diagnosed[final_inds] = people.t + test_delay
    people.date_pos_test[final_inds] = people.t

    return final_inds
