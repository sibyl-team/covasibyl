from collections import defaultdict
from time import time
import warnings

import numpy as np
import matplotlib.pyplot as pl

import sciris as sc
from covasim.analysis import Analyzer
from covasim.interventions import get_day
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
    def __init__(self,quar_factor=1., iso_factor=0.1, start_day=0, end_day=None, save_only="",printout=False, every_day=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.contacts_saved = None
        self.quar_factor=quar_factor
        self.iso_factor=iso_factor
        self.save_only = save_only
        self.everyday = every_day
        self.start_day=start_day
        self.end_day=end_day
    
        self._warned = defaultdict(lambda: False)

    def _warn_once(self, key:str, message:str):
        if not self._warned[key]:
            warnings.warn(message)
            self._warned[key] = True
    
    def initialize(self, sim=None):

        found_rktest = False
        if sim is not None:
            intervs = sim["interventions"]
            for i in intervs:
                try:
                    self.app_usage = i.get_app_usage()
                    found_rktest = True
                    break
                except AttributeError:
                    pass
        if not found_rktest:
            self._warn_once("app_usage","Could not get app usage (cannot find intervention?)\n\tSaving all contacts.")
            self.app_usage = np.ones(sim.pars["pop_size"])
        return super().initialize(sim)

    def apply(self, sim):
        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return
        
        t0 = time()
        N = len(sim.people.sex)
        cts_df = utils.get_contacts_day(sim.people)
        if self.app_usage.sum() < N:
            ## remove contacts that do not have app
            cts_df = utils.filt_contacts_mult(cts_df, self.app_usage,
            N, only_i=False) 
        

        tested_iso = cvu.true(sim.people.diagnosed)
        cts_df = utils.filt_contacts_df(cts_df,
                tested_iso, self.iso_factor, N, only_i=True)

        if self.quar_factor < 1:
            ## filter the contacts for quarantined
            self._warn_once("qfact", f"Quarantine factor is {self.quar_factor}")
            quar_idcs = cvu.true(sim.people.quarantined)

            if len(quar_idcs) > 0:
                msum = cts_df["m"].sum()
                cts_df = utils.filt_contacts_df(cts_df, quar_idcs, self.quar_factor, N)
                assert cts_df["m"].sum() < msum

        ## filter by state
        if self.save_only=="EI":
            exp_i_idc=cvu.true(sim.people.exposed)
            cts_df = utils.sel_contacts_idx_df(cts_df, exp_i_idc, N,which="or")
        elif self.save_only!="":
            self._warn_once("stat_save",f"Saving string {self.save_only} not recognized, saving all contacts by state")

        if self.everyday:
            day = sim.t
            if self.contacts_saved is None:
                self.contacts_saved = {}
            cts_df["day"]= day
            self.contacts_saved[day] = cts_df.to_records(index=False)
        else:
            self.contacts_saved = cts_df
        
        t_take=time() - t0
        print(f"Saved contacts in {t_take:4.3f} s")


class StateSaver(Analyzer):
    def __init__(self, label=None):
        super().__init__(label)

        self.states = None
    
    def initialize(self, sim=None):

        self.states = dict()
        super().initialize(sim)
        
    def apply(self, sim):
        
        N = int(sim["pop_size"])
        status = np.empty(N, dtype=np.int8)
        people = sim.people
        status[people.susceptible] = 0
        is_E = (people.exposed &(~people.infectious))
        status[is_E] = 1
        

        nonsympt = (people.infectious &(~people.symptomatic))
        asympt = nonsympt & np.isnan(people.date_symptomatic)
        presympt = nonsympt & np.logical_not(asympt)
        #stats_df.loc[inds_test[asympt], "t_state"] = "AS"
        status[asympt] = 2
        status[presympt] = 3
        #stats_df.loc[inds_test[presympt],"t_state"] = "PS"

        symptomat = people.symptomatic
        severe = people.severe
        critical = people.critical
        mild = symptomat & np.logical_not(severe | critical)
        status[mild] = 4
        status[critical] = 5
        status[severe]=6

        status[people.recovered] = 7
        status[people.dead] = 8

        self.states[sim.t] = status

