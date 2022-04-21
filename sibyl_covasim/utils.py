from covasim import immunity
import numpy as np
#import numba as nb
import scipy.sparse as sp
import pandas as pd


def filter_contacts(p1, p2, beta):
    """
    Remove invalid contacts, then duplicate them
    """
    valid = (p1!=p2)
    i = p1[valid]
    j = p2[valid]
    m = beta[valid]
    return np.concatenate((i,j)), np.concatenate((j,i)), np.concatenate((m,m))

def get_contacts_day(people):
    N = len(people.sex)
    c = 0
    for n,lay in people.contacts.items():
        #print(n,lay)
        w = people.pars["beta_layer"][n]
        u = filter_contacts(**lay)
        mat = sp.csr_matrix((u[2]*w,(u[0],u[1])), shape=(N,N))
        c += mat
    cend = c.tocoo()
    return pd.DataFrame(dict(zip(["i","j","m"],(cend.row, cend.col, cend.data))) )