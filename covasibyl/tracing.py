from collections import defaultdict
import numpy as np

import covasim.defaults as cvd
import covasim.utils as cvu
from covasim.interventions import contact_tracing

def binomial_filter_rng(prob, arr, rng): # No speed gain from Numba
    '''
    Binomial "filter" -- the same as n_binomial, except return
    the elements of arr that succeeded.

    Args:
        prob (float): probability of each trial succeeding
        arr (array): the array to be filtered

    Returns:
        Subset of array for which trials succeeded

    **Example**::

        inds = cv.binomial_filter(0.5, np.arange(20)**2) # Return which values out of the (arbitrary) array passed the coin flip
    '''
    return arr[(rng.random(len(arr)) < prob).nonzero()[0]]

class FlexibleTracing(contact_tracing):

    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None, presumptive=False, quar_period=None, capacity=None, **kwargs):
        super().__init__(trace_probs=trace_probs, trace_time=trace_time, start_day=start_day, 
            end_day=end_day, presumptive=presumptive, quar_period=quar_period, capacity=capacity, **kwargs) # Initialize the Intervention object
        self.rng = None
        self.has_app = None
        
        return

    def initialize(self, sim):

        self.rng = np.random.RandomState(np.random.PCG64(2))
        return super().initialize(sim)

    def select_cases(self, sim):
        '''
        Return people to be traced at this time step
        '''
        if not self.presumptive:
            inds = cvu.true(sim.people.date_diagnosed == sim.t) # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == sim.t) # Tested this time step, time to trace
            inds = cvu.itruei(sim.people.exposed, just_tested) # This is necessary to avoid infinite chains of asymptomatic testing

        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if self.capacity is not None:
            capacity = int(self.capacity / sim.rescale_vec[sim.t])  # Convert capacity into a number of agents
            if len(inds) > capacity:
                inds = self.rng.choice(inds, capacity, replace=False)

        return inds

    def identify_contacts(self, sim, trace_inds):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''

        if not len(trace_inds):
            return {}

        contacts = defaultdict(list)

        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_inds = sim.people.contacts[lkey].find_contacts(trace_inds)
            if len(traceable_inds):
                contacts[self.trace_time[lkey]].extend(binomial_filter_rng(this_trace_prob, traceable_inds, self.rng)) # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, inds in contacts.items():
            array_contacts[trace_time] = np.fromiter(inds, dtype=cvd.default_int)

        return array_contacts