import scipy.sparse as sp
import covasim.interventions as cvi

class ContactsSaver(cvi.Intervention):

    def __init__(self):

        super().__init__(label="Contacts counter")
        self.contacts = None
        self.conts_csr = None
        self.n = None
        #self.num_changed = None

    def initialize(self, sim=None):
        super().initialize(sim)
        self.contacts = {}
        self.conts_csr = {}
        if sim is not None:
            self.n = sim.n
        #self.num_changed = None


    def apply(self, sim):
        day = sim.t
        if self.n is None:
            self.n = sim.n

        mcontacts = self.contacts

        #if len(mcontacts) == 0:
        """
        mcontacts[day]=sum( sp.csr_matrix(
            (layer["beta"],(layer["p1"],layer["p2"])
            ), shape=(sim.n, sim.n)) for k, layer in sim.people.contacts.items())
        #else:
        """
        mcontacts[day] = {k:layer.to_df() for k,layer in sim.people.contacts.items()}

    def _build_csr(self, day):
    
        self.conts_csr[day] = sum(sp.csr_matrix(
            (layer["beta"],(layer["p1"],layer["p2"])
            ), shape=(self.n, self.n)) for k,layer in self.contacts[day].items())



    def to_csr(self, day=None):
        """
        Return the contacts as sparse matrices
        """
        if day is None:
            for t in self.contacts:
                if t not in self.conts_csr:
                    self._build_csr(t)
            return self.conts_csr

        if day not in self.conts_csr:
            self._build_csr(day)

        return self.conts_csr[day]
        
            



