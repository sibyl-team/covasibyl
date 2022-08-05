from collections import defaultdict
import warnings

import numpy as np
import matplotlib.pyplot as pl

import sciris as sc
from covasim.analysis import Analyzer
from covasibyl import utils
import covasim.utils as cvu


class store_seir(Analyzer):

    def __init__(self, printout=False, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.t = []
        self.S = []
        self.E = []
        self.I = []
        self.R = []
        self.Q = []
        self.IND = []
        self.Efree = []
        self.printout=printout
        return

    def apply(self, sim):
        ppl = sim.people # Shorthand
        self.t.append(sim.t)
        self.S.append(ppl.susceptible.sum())
        self.E.append(ppl.exposed.sum() - ppl.infectious.sum())
        self.I.append(ppl.infectious.sum())
        self.R.append(ppl.recovered.sum() + ppl.dead.sum())
        self.Q.append(ppl.quarantined.sum())
        self.IND.append((ppl.infectious & (~ppl.diagnosed)).sum())  
        self.Efree.append((ppl.exposed & (~ppl.infectious) & (~ppl.diagnosed)).sum() )
        EIfree = self.IND[-1]+self.Efree[-1]
        if (self.printout):
            print(f"day {sim.t} -> I (free): {self.I[-1]} ({self.IND[-1]}),"+\
                f" E+I (free): {self.I[-1]+self.E[-1]} ({EIfree}) R: {self.R[-1]}")      
        return

    def plot(self, **args):
        pl.figure()
        pl.plot(self.t, self.S, label='S')
        pl.plot(self.t, self.E, label='E')
        pl.plot(self.t, self.I, label='I')
        pl.plot(self.t, self.R, label='R')
        pl.legend()
        pl.xlabel('Day')
        pl.ylabel('People')
        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        return

    def out_save(self):
        
        return np.array(list(zip(self.t, self.S, self.E, self.I, self.R, self.Q, self.IND, self.Efree)),
            dtype=[(i,np.int_) for i in ["t","S","E","I","R", "Q", "nondiag","Enondiag"]])


class QuarantineSaver(Analyzer):
    def __init__(self, printout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.quar_day = dict()

        self.dequar_day = dict()

        self.inquar = None

    def apply(self, sim):

        day = sim.t

        quar_idcs = np.where(sim.people.quarantined)[0]

        if self.inquar is None:
            self.quar_day[day] = quar_idcs
            ## no going out of quarantine
        else:
            new_quar = np.setdiff1d(quar_idcs, self.inquar)
            self.quar_day[day] = new_quar
            new_out = np.setdiff1d(self.inquar, quar_idcs)
            self.dequar_day[day] = new_out

        self.inquar = quar_idcs

        pending = sim.people._pending_quarantine
        if len(pending[day]) > 0:
            pd_q = np.array(pending[day])
            idcs_pending = pd_q[:,0]
            notq = np.setdiff1d(idcs_pending, self.quar_day[day])
            iquar = cvu.itruei(sim.people.quarantined | sim.people.diagnosed | sim.people.dead | sim.people.recovered, notq)
            notq = np.setdiff1d(notq, iquar)
            #iddiag = cvu.itruei(sim.people.diagnosed, notq)
            #notq = np.setdiff1d(notq, iddiag)
            if len(notq) > 0:
                print(f"Day {day}, Not quarantining idcs {notq}")
        else:
            assert len(self.quar_day[day]) == 0

class ContactsSaver(Analyzer):
    def __init__(self,quar_factor=1., printout=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.contacts_saved = None
        self.quar_factor=quar_factor
    
        self._warned = defaultdict(lambda: False)

    def _warn_once(self, key:str, message:str):
        if not self._warned[key]:
            warnings.warn(message)
            self._warned[key] = True

    def apply(self, sim):

        N = len(sim.people.sex)
        contacts_df = utils.get_contacts_day(sim.people)

        tested_iso = cvu.true(sim.people.diagnosed)
        contacts_df = utils.filt_contacts_df(contacts_df,
                tested_iso, self.iso_cts_strength, N, only_i=True)

        if self.quar_factor < 1:
            ## filter the contacts for quarantined
            self._warn_once("qfact", f"Quarantine factor is {self.quar_factor}")
            quar_idcs = cvu.true(sim.people.quarantined)

            if len(quar_idcs) > 0:
                msum = contacts_df["m"].sum()
                contacts_df = utils.filt_contacts_df(contacts_df, quar_idcs, self.quar_factor, N)
                assert conts_prev_day["m"].sum() < msum

        self.contacts_saved = contacts_df