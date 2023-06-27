import warnings

import numpy as np
from numba import njit

def get_people_dates(peop):
    """
    Get array of people dates from covasim people
    """
    dates_save=np.rec.fromarrays((peop.date_exposed, peop.date_infectious, 
                      peop.date_symptomatic, peop.date_severe, peop.date_critical,
                      peop.date_diagnosed, peop.date_recovered, peop.date_dead),
          names=("date_exposed", "date_infectious", "date_symptomatic","date_severe", 
                "date_critical", "date_diagnosed", "date_recovered", "date_dead")
     )
    return dates_save

def filter_resu_arr(resu):
    if hasattr(resu,"values"):
        return resu.values
    else: 
        return resu

def idcs_notnan(arr):
    ''' Return indices in arr which are not-nan '''
    return (~np.isnan(arr)).nonzero()[0]

@njit()
def rolling_average(arr,window, debug=False):
    """
    Perform rolling average, assuming constant values when window is at the edge
    """
    if len(arr) <= window:
        raise ValueError("Array too large for the window size")
    ur =np.empty(len(arr))
    nw=(window-1)//2
    if (window-1)%2 !=0:
        raise ValueError("Need odd window size")
    if debug: print(nw)
    for t in range(len(arr)):
        if t-nw < 0:
            mint=0
        else:
            mint = t-nw
        if t+nw>=len(arr):
            maxt=len(arr)
        else:
            maxt=t+nw+1
        nx_0=mint-(t-nw)
        nx_las= t+nw+1-maxt
        if debug: print(t,mint,nx_0 , maxt, nx_las)

        ur[t] = sum(arr[mint:maxt])+nx_0*arr[0]+nx_las*arr[len(arr)-1]

    return ur/window

def compute_r_eff_resu(res_dict, method='daily', smoothing=2, window=7):
    '''
    Effective reproduction number based on number of people each person infected.

    Args:
        method (str): 'daily' uses daily infections, 'infectious' counts from the date infectious, 'outcome' counts from the date recovered/dead
        smoothing (int): the number of steps to smooth over for the 'daily' method
        window (int): the size of the window used for 'infectious' and 'outcome' calculations (larger values are more accurate but less precise)

    Returns:
        r_eff (array): the r_eff results array
    '''

    # Initialize arrays to hold sources and targets infected each day
    
    
    window = int(window)
    dates=res_dict["people_dates"]
    results=res_dict["sim_res"]
    npts=len(results["n_infectious"])

    sources = np.zeros(npts)
    targets = np.zeros(npts)
    filt=filter_resu_arr
    # Default method -- calculate the daily infections
    if method == 'daily':

        # Find the dates that everyone became infectious and recovered, and hence calculate infectious duration
        recov_inds   = idcs_notnan(dates['date_recovered'])
        dead_inds    = idcs_notnan(dates['date_dead'])
        date_recov   = dates['date_recovered'][recov_inds]
        date_dead    = dates['date_dead'][dead_inds]
        date_outcome = np.concatenate((date_recov, date_dead))
        inds         = np.concatenate((recov_inds, dead_inds))
        date_inf     = dates["date_infectious"][inds]
        if len(date_outcome):
            mean_inf     = date_outcome.mean() - date_inf.mean()
        else:
            warnmsg ='There were no infections during the simulation'
            warnings.warn(warnmsg)
            mean_inf = 0 # Doesn't matter since r_eff is 0

        # Calculate R_eff as the mean infectious duration times the number of new infectious divided by the number of infectious people on a given day
        new_infections = filt(results['new_infections']) - filt(results['n_imports'])
        n_infectious = filt(results['n_infectious'])
        raw_values = mean_inf*np.divide(new_infections, n_infectious, out=np.zeros(npts), where=n_infectious>0)

        # Handle smoothing, including with too-short arrays
        len_raw = len(raw_values) # Calculate the number of raw values
        
        values = raw_values

    # Method not recognized
    else: # pragma: no cover
        errormsg = f'Method must be "daily", "infectious", or "outcome", not "{method}"'
        raise ValueError(errormsg)

    return values


def compute_r_eff_expo(res_dict, method='daily'):
    '''
    Effective reproduction number calculation from daily change of infected individuals

    Args:
        method (str): 'daily' uses daily infections,

    Returns:
        r_eff (array): the r_eff results array
    '''

    # Initialize arrays to hold sources and targets infected each day
    
    
    dates=res_dict["people_dates"]
    results=res_dict["sim_res"]
    npts=len(results["n_infectious"])

    filt=filter_resu_arr
    # Default method -- calculate the daily infections
    if method == 'daily':

        # Find the dates that everyone became infectious and recovered, and hence calculate infectious duration
        recov_inds   = idcs_notnan(dates['date_recovered'])
        dead_inds    = idcs_notnan(dates['date_dead'])
        date_recov   = dates['date_recovered'][recov_inds]
        date_dead    = dates['date_dead'][dead_inds]
        date_outcome = np.concatenate((date_recov, date_dead))
        inds         = np.concatenate((recov_inds, dead_inds))
        date_inf     = dates["date_exposed"][inds]
        if len(date_outcome):
            mean_inf     = date_outcome.mean() - date_inf.mean()
        else:
            warnmsg ='There were no infections during the simulation'
            warnings.warn(warnmsg)
            mean_inf = 0 # Doesn't matter since r_eff is 0

        # Calculate R_eff as the mean infectious duration times the number of new infectious divided by the number of infectious people on a given day
        new_infections = filt(results['new_infections']) - filt(results['n_imports'])
        n_infectious = filt(results['n_exposed']) #filt(results['n_infectious'])
        raw_values = mean_inf*np.divide(new_infections, n_infectious, out=np.zeros(npts), where=n_infectious>0)

        # Handle smoothing, including with too-short arrays
        len_raw = len(raw_values) # Calculate the number of raw values
        
        values = raw_values

    # Method not recognized
    else: # pragma: no cover
        errormsg = f'Method must be "daily", "infectious", or "outcome", not "{method}"'
        raise ValueError(errormsg)

    return values