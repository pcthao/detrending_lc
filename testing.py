
import numpy as np 
import lightkurve as lk 
from scipy.sparse import diags
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader, TensorDataset


def get_quarter (target_name, mission_name, cadence_name, quarter_num):
    lc = lk.search_lightcurve(target_name, mission=mission_name, cadence=cadence_name,quarter=quarter_num).download(quality_bitmask=0, flux_column="pdcsap_flux")
    
    ##################################################################
    # pre-processing the data 
    x0 = np.ascontiguousarray(lc.time.value, dtype=np.float64)
    y = np.ascontiguousarray(lc.flux, dtype=np.float64)
    ivar = np.ascontiguousarray(1 / lc.flux_err ** 2, dtype=np.float64)

    np.count_nonzero(~np.isnan(x0))

    m = np.isfinite(y)
    ivar[~m] = 0.0

    x = np.arange(x0.min(), x0.max(), np.median(np.diff(x0)))
    y = np.interp(x, x0[m], y[m])
    ivar = np.interp(x, x0[m], ivar[m])

    # Mask "interpolated" points - HACK
    dx = np.abs(x - x0[m][np.searchsorted(x0[m], x)])
    ivar[dx > 2 * np.median(np.diff(x0))] = 0.0

    mu = np.median(y) # counts
    # y = counts, divide by the median [parts per thousand]
    y = 1e3 * (y / mu - 1)
    ivar *= 1e-6 * mu ** 2
    
    return x, y, ivar

def starting_i (mask_length):
    if mask_length % 2 != 0:
        pass 
    else: 
        raise ValueError('mask_length needs to be an odd value')
    y = 1/2*mask_length-1/2 
    return int(y) 


def rolling_window_masks(flux_array, mask_length, ii, plot = False):
    # get the length of the flux 
    length = len(flux_array)

    # this is a list of [n+2, n+1, 0, n+1, n+2] 
    n_ls = np.abs(np.arange(-ii, ii+1))
    
    # create a list of what to input into the final array  
    k_ls = []
    for i in range(0,len(n_ls)):
        k_ls.append(np.ones(length-n_ls[i]))

    # indices of the offset 
    offset = np.arange(-ii, ii+1)
    
    A = diags(k_ls,offset).toarray()
    
    if plot == True:
        fig, ax = plt.subplots(1, figsize=(15,5))
        plt.imshow(mask_window)
        plt.show()

    # invert the final array so that 0s are along the diagonal 
    return 1.-A 

def unravel_lc (flux_array, mask_window, model, null_val):
    import time 
    start = time.time()
        
    # get the length of the flux array
    N = len(flux_array)
    # create an array that is the same as the flux array N times 
    yy = np.vstack([flux_array]*N)

    # lets mask the flux array 
    yy_masked = yy * mask_window
    # set the 0 values/mask values in the mask_window to be null_val 
    yy_masked[yy_masked==0.] = null_val

    # a boolean mask in which the 1s are in the diagonal, shape = [N,N]
    final_mask = np.zeros((N,N))
    # fill 1s in the diagonal 
    np.fill_diagonal(final_mask, val = 1)

    # convert the arrays[N,N] to be a tensor [N,N,-1]
    yy_masked = torch.FloatTensor(yy_masked).reshape(N,-1,1)
    final_mask = torch.FloatTensor(final_mask).reshape(N,-1,1)

    dataset = TensorDataset(yy_masked, final_mask)
    loader = DataLoader(dataset, shuffle=False, batch_size = 1) 

    final_ls = []
    with torch.no_grad():
        for batch_idx, (yy_masked, final_mask) in enumerate(loader): 
            # inputting the yy_masked into the model/ forward propogation  
            pred_flux, hidden = model(yy_masked)
            pred = (pred_flux * final_mask) #.detach().numpy() [1,NUM,1]

            # lets sum each 'row' because it will then be the middle value that we want :) 
            # lets append that to a list
            final_ls.append(torch.sum(pred).item())
    import time 
    time_elapsed = time.time() - start
    print('Unraveling complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return final_ls

def expected_transit_time (period, t0_bkjd, x):
    '''
    period [days]
    t0: t0 value [bkjd]
    x: time [bkjd]
    '''
    min_x, max_x = x[0], x[-1]
    min_n, max_n = (min_x - t0_bkjd) / period, (max_x - t0_bkjd) / period
    tr_ls = [] 
    
    for n in range(int(min_n),int(max_n)+1):
        tr_ls.append(t0_bkjd + (period*n))
    return tr_ls

def plot_comparision (x, y, pred_flux):
    fig, (ax0, ax1) = plt.subplots(nrows=2,figsize=(15,4), gridspec_kw={'height_ratios':[9,4]}, sharex=True) 

    ax1 = plt.gca()
    ax1.yaxis.set_ticks_position('both') # ticks appear on both side 
    ax1.xaxis.set_ticks_position('both') # ticks appear on both side 
    ax1.tick_params(which='both', direction="in", width=2) # moves the major and minor ticks inside the plot 
    ax1.minorticks_on() # turn on minor ticks 
    ax1.tick_params(axis='x', which='minor', bottom=True)  # turn off minor ticks in x-axis
    ax1.tick_params(axis='x', which='minor', top=True)     # turn off minor ticks in x-axis

    ax0.yaxis.set_ticks_position('both') # ticks appear on both side 
    ax0.xaxis.set_ticks_position('both') # ticks appear on both side 
    ax0.tick_params(which='both', direction="in", width=2) # moves the major and minor ticks inside the plot 
    ax0.minorticks_on() # turn on minor ticks 
    ax0.tick_params(axis='x', which='minor', bottom=True)  # turn off minor ticks in x-axis
    ax0.tick_params(axis='x', which='minor', top=True)     # turn off minor ticks in x-axis
    
    ax1.set_xlabel ("Time [BKJD]")
    ax0.set_ylabel ("Norm. Flux")
    ax1.set_ylabel ("Res")

    ax0.plot(x, y, label = "Truth", linewidth = 2, color = 'k')
    ax0.scatter(x, pred_flux, label = "Prediction", s=6, color ='red')
    res = y-pred_flux
    ax1.axhline(y=0.0, color='k', linestyle='-')
    ax1.scatter(x,res, s=3, color = 'red')
    ax0.legend()
    plt.show()
    return 