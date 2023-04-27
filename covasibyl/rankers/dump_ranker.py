import sys
import time
import numpy as np
import pandas as pd
from .template_rank import AbstractRanker

TAU_INF = 10000000

class DumpRanker(AbstractRanker):

    def __init__(self,t_dump, save_name="rankdump",
                lamb = 0.99):
        self.description = "class for dumping contacts and observations"
        self.t_dump=t_dump
        self.lamb = lamb
        self.rng = np.random.RandomState(1)
        self.name_save = save_name

    def init(self, N, T):
        self.contacts = None
        #dummy obs, needed if the first time you add only one element
        self.obs = [(0,-1,0)] 
        self.T = T
        self.N = N
        self.rank_not_zero = np.zeros(T)

        return True

    '''def _save_contacts(self, daily_contacts):
        """
        Save contacts in a pandas dataframe
        This is slower than numpy but easier to handle
        """
        if isinstance(daily_contacts, np.recarray):
            daily_contacts.dtype.names = "i", "j", "t", "lambda"
            cts_d = pd.DataFrame(daily_contacts)
        else:
            cts_d = pd.DataFrame(np.array(daily_contacts), columns=["i", "j", "t", "lambda"])

        assert len(cts_d) == len(daily_contacts)
        print(f"{len(cts_d)} new contacts,", end=" ")
        if self.contacts is None:
            self.contacts = cts_d
        else:
            self.contacts = pd.concat((self.contacts, cts_d), ignore_index=True)
    '''

    def _save_contacts(self, daily_contacts):
        """
        Save contacts in a pandas dataframe
        This is slower than numpy but easier to handle
        """
        conts_dtype = np.dtype([(k, "int") for k in ["i","j","t"]]+[("lambda", "float")])
        if len(daily_contacts) > 0:
            if isinstance(daily_contacts, np.recarray):
                cts_d = daily_contacts.copy()
                cts_d.dtype.names = "i", "j", "t", "lambda"
                
            else:
                cts_d = np.array(daily_contacts,dtype=conts_dtype)
        else:
            cts_d = np.empty((0,), dtype=conts_dtype)
            #print("New contacts: ", cts_d)

        assert len(cts_d) == len(daily_contacts)
        print(f"{len(cts_d)} new contacts,", end=" ")
        if self.contacts is None or len(self.contacts)==0:
            self.contacts = cts_d
        else:
            self.contacts = np.concatenate((self.contacts, cts_d))
    

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        
        t0=time.time()
        for obs in daily_obs:
            self.obs.append(obs)
        to = time.time()-t0

        t0=time.time()
        self._save_contacts(daily_contacts)
        tc=time.time()-t0
        
        print(f"t_saveobs: {to:5.4e}, t_savec: {tc:6.5f} s", end="\n")

        if t_day == self.t_dump:
            obs_df = pd.DataFrame(self.obs, columns=["i", "s", "t_test"])
            mcontacts = self.contacts

            save_d = {"contacts": mcontacts,
                "observ": obs_df.to_records(index=False)
            }
            
            np.savez_compressed(self.name_save+f"_t{t_day}.npz", **save_d)
            print("Saved contacts and observations")

        

        rankv = self.rng.rand(self.N)

        return list(zip( range(rankv.shape[0]), rankv ))

