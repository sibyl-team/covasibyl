import scipy.sparse as sp
import covasim.interventions as cvi


class Mitigation(cvi.Intervention):
    """
    Class to run epidemic mitigation
    Give a ranker from epidemic_mitigation, and it will run the ranker and 
    quarantine people
    """

    def __init__(self, ranker, label):

        super().__init__(label="Mitigation: "+label)

        self.ranker = ranker
        self.contacts_day = None
        self.delayed_init = False

    def _init_ranker(self, sim):
        """
        initialize the ranker
        """
        pars = sim.pars
        N = pars["pop_size"]
        T = pars["n_days"]
        self.ranker.init(N, T)
    
    def initialize(self, sim=None):
        super().initialize(sim)
        if sim is None:
            self.delayed_init = True
        else:
            self._init_ranker(sim)

    def apply(self, sim):
        pass