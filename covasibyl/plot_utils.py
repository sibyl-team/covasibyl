import numpy as np
import matplotlib.pyplot as plt

def plot_vals(res_dict, keys_avoid=["Free"], quants=None, apply_dat=None, ax=None, tlim=None, colors=None):
    
    if colors == None:
        colors = dict()
    p = plt if ax is None else ax
    fn = (lambda x: x )if apply_dat is None else apply_dat
    for k, dat in res_dict.items():
        #if k=="TestCT": continue
        if k in keys_avoid: continue
        dat=fn(dat)
        if tlim is not None:
            dat = dat[:,:tlim]
        if quants is not None:
            qs = np.quantile(dat,quants,axis=0)
        s=np.std(dat, axis=0)/np.sqrt(dat.shape[0]-1)
        m = dat.mean(0)
        clr = colors[k] if k in colors else None
        
        l, = p.plot(dat.mean(0),linewidth=2, label=k,color=clr)
        #plt.fill_between(np.arange(dat.shape[1]),qs[0],qs[1], color=l.get_color(), alpha=0.24)
        if quants is None:
            p.fill_between(np.arange(dat.shape[1]),m-s,m+s, color=l.get_color(), alpha=0.22)
        else:
            p.fill_between(np.arange(dat.shape[1]),qs[0],qs[1], color=l.get_color(), alpha=0.22)
    