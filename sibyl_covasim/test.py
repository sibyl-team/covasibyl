import numpy as np
import covasim.interventions as cvi
import covasim.utils as cvu

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

class CovasimTester:

    def __init__(self, sim, maxtest=None):
        
        pars = sim.pars
        self.N = pars["pop_size"]
        self.date_diagn_state = np.full((3,self.N), np.nan)
        self.date_diagnosed = np.full(self.N, np.nan)
        self.date_pos_test = np.full_like(self.date_diagnosed, np.nan)

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

        people.tested[inds] = True
        people.date_tested[inds] = sim.t # Only keep the last time they tested

        ## lost tests
        not_lost = cvu.n_binomial(1.0-loss_prob, len(inds))
        inds_test = inds[not_lost]
        num_lost = np.logical_not(not_lost).sum()
        #if(num_lost > 1):
        #    print(f"Lost {num_lost} tests")

        ## check susceptibles
        susc_inds = cvu.itruei(people.susceptible, inds_test)
        neg_test = random_all_p(test_specificity, len(susc_inds))
        res_susc = susc_inds[neg_test]
        res_infected = susc_inds[np.logical_not(neg_test)]

        ### find infectious indices
        inf_or_exp = people.infectious[inds_test] | people.exposed[inds_test]
        is_infectious = cvu.true(inf_or_exp)
        ## find the ones which test positive
        pos_test      = random_all_p(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]
        ### concatenate results
        res_susc = np.concatenate((res_susc, is_infectious[np.logical_not(pos_test)]))
        res_infected = np.concatenate((res_infected, is_inf_pos))

        ## infectious, positive, not diagnosed yet
        inf_not_diagnosed = res_infected[np.isnan(people.date_diagnosed[res_infected]) & np.isnan(self.date_diagnosed[res_infected])]
        #not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        #final_inds    = not_diagnosed[not_lost]
        inf_not_diagnosed = res_infected
        # check for recovered
        is_recov = people.recovered[inds_test] | people.dead[inds_test]
        rec_inds = cvu.true(is_recov)
        # TODO: use specificity here?
        ## check they haven't been diagnosed before?
        res_recov = rec_inds#rec_inds[np.isnan(people.date_diagnosed[rec_inds])]
        
        assert len(res_recov) + len(res_infected) + len(res_susc) == num_tests
        ## Keep this for compatibility
        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        self.date_diagnosed[inf_not_diagnosed] = sim.t + test_delay
        self.date_pos_test[inf_not_diagnosed] = sim.t

        self.date_diagn_state[0, res_susc] = sim.t + test_delay
        self.date_diagn_state[1, inf_not_diagnosed] = sim.t + test_delay
        self.date_diagn_state[2, res_recov] = sim.t + test_delay

        return

    def get_results(self, day):

        return np.stack(np.where(self.date_diagn_state == day),1)[:,::-1]