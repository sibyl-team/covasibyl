import numpy as np
import covasim.interventions as cvi
import covasim.utils as cvu

# TODO: make class to do tests, with delay

class CovasimTester:

    def __init__(self, sim, maxtest=None):
        
        pars = sim.pars
        self.N = pars["pop_size"]
        self.date_diagn_state = np.full((3,self.N), np.nan)

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
        
        people.tested[inds] = True
        people.date_tested[inds] = sim.t # Only keep the last time they tested

        ## lost tests
        not_lost = cvu.n_binomial(1.0-loss_prob, len(inds))
        inds_test = inds[not_lost]

        ## check susceptibles
        susc_inds = cvu.itruei(people.susceptible, inds_test)
        neg_test = cvu.n_binomial(test_specificity, len(susc_inds))
        test_susc = susc_inds[neg_test]
        test_inf = susc_inds[np.logical_not(neg_test)]

        ### find infectious indices
        is_infectious = cvu.itruei(people.infectious, inds_test)
        ## find the ones which test positive
        pos_test      = cvu.n_binomial(test_sensitivity, len(is_infectious))
        is_inf_pos    = is_infectious[pos_test]
        ### concatenate results
        test_susc = np.concatenate((test_susc, is_infectious[np.logical_not(pos_test)]))
        test_inf = np.concatenate((test_inf, is_inf_pos))

        ## infectious, positive, not diagnosed yet
        inf_not_diagnosed = test_inf[np.isnan(people.date_diagnosed[test_inf])]
        #not_lost      = cvu.n_binomial(1.0-loss_prob, len(not_diagnosed))
        #final_inds    = not_diagnosed[not_lost]

        # check for recovered
        rec_inds = cvu.itruei(people.recovered, inds_test)
        # TODO: use specificity here?
        ## check they haven't been diagnosed before?
        test_rec = rec_inds#rec_inds[np.isnan(people.date_diagnosed[rec_inds])]
        
        ## Keep this for compatibility
        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        people.date_diagnosed[inf_not_diagnosed] = sim.t + test_delay
        people.date_pos_test[inf_not_diagnosed] = sim.t

        self.date_diagn_state[0, test_susc] = sim.t + test_delay
        self.date_diagn_state[1, inf_not_diagnosed] = sim.t + test_delay
        self.date_diagn_state[2, test_rec] = sim.t + test_delay

        return

    def get_results(self, day):

        return np.stack(np.where(self.date_diagn_state == day),1)[:,::-1]