import numpy as np

import covasim.utils as cvu
import covasim.defaults as cvd

from . import utils

# TODO: make class to do tests, with delay

def random_all_p(p, n, rand=np.random):
    '''
    Perform multiple binomial (Bernolli) trials

    Args:
        p (float): probability of each trial succeeding
        n (int): number of trials (size of array)

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

class CovasimTester:

    def __init__(self, sim, seed=None):
        
        pars = sim.pars
        self.N = pars["pop_size"]
        self.date_diagn_state = np.full((3,self.N), np.nan)
        self.date_diagnosed = np.full(self.N, np.nan)
        self.diagnosed = np.zeros(self.N,dtype=bool)
        self.date_posit_test = np.full_like(self.date_diagnosed, np.nan)
    
    def _not_diagnosed(self):
        return np.isnan(self.date_diagnosed)

    def apply_tests(self, sim, inds, test_sensitivity=1.0, test_specificity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people, accounting for recovered 
        and susceptible individuals (SIR model)


        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            test_specificity (float): probability of a true negative
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''
        people = sim.people
        inds = np.unique(inds)
        num_tests = len(inds)
        ## check that there are no diagnosed (eg positive-tested) people
        diagnosed = people.diagnosed[inds] | self.diagnosed[inds]
        assert diagnosed.sum() == 0

        people.tested[inds] = True
        people.date_tested[inds] = sim.t # Only keep the last time they tested

        
        
        ## lost tests
        not_lost = utils.n_binomial(1.0-loss_prob, len(inds))
        inds_test = inds[not_lost]
        num_lost = np.logical_not(not_lost).sum()
        #if(num_lost > 1):
        #    print(f"Lost {num_lost} tests")
        ## exposed individuals remain exposed when infectious
        is_E = (people.exposed &(~people.infectious))

        ## check susceptibles
        susc_inds = cvu.itruei((people.susceptible | is_E), inds_test)
        neg_test = random_all_p(test_specificity, len(susc_inds))
        res_susc = susc_inds[neg_test]
        
        res_infected = susc_inds[np.logical_not(neg_test)] # false positive
        if(test_specificity >= 1.):
            assert len(res_infected) == 0

        ### find infectious indices
        #inf_or_exp = people.infectious[inds_test] | people.exposed[inds_test]
        is_I_idcs = cvu.itruei(people.infectious, inds_test) #return indices
        ## find the ones which test positive
        pos_test      = np.random.rand(len(is_I_idcs)) < test_sensitivity #random_all_p(test_sensitivity, len(is_I_idcs))
        is_inf_pos    = is_I_idcs[pos_test]
        ### concatenate results
        res_susc = np.concatenate((res_susc, is_I_idcs[np.logical_not(pos_test)]))
        res_infected = np.concatenate((res_infected, is_inf_pos))
        
        """## infectious, positive, not diagnosed yet
        inf_not_diagnosed = res_infected[]
        #not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        #final_inds    = not_diagnosed[not_lost]
        """
        final_inf_idcs = res_infected
        # check for recovered
        is_recov = people.recovered[inds_test] | people.dead[inds_test]
        rec_inds = cvu.true(is_recov)
        # TODO: use specificity here?
        res_recov = inds_test[rec_inds]
        
        #assert len(res_recov) + len(res_infected) + len(res_susc) == num_tests
        ## Keep this for compatibility
        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[final_inf_idcs] = sim.t + test_delay
        self.date_posit_test[final_inf_idcs] = sim.t
        ## COPY ON COVASIM: important!! -> diagnosed individuals are isolated
        sim.people.date_diagnosed[final_inf_idcs] = sim.t +test_delay
        sim.people.date_pos_test[final_inf_idcs] = sim.t

        self.date_diagn_state[0, res_susc] = sim.t + test_delay
        self.date_diagn_state[1, final_inf_idcs] = sim.t + test_delay
        self.date_diagn_state[2, res_recov] = sim.t + test_delay

        ## handle diagnosed
        diag_inds  = people.check_inds(self.diagnosed, self.date_diagnosed, filter_inds=None) # Find who was actually diagnosed on this timestep
        self.diagnosed[diag_inds] = True

        return num_lost

    def get_results(self, day):

        return np.stack(np.where(self.date_diagn_state == day),1)[:,::-1]