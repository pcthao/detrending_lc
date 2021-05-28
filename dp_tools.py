
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import torch 


def get_data (target_name, mission_name, cadence_name, quarter_ls, plot = False, linear = False):
    x_quarter_ls = []
    y_quarter_ls = []
    # list for weights 
    w_quarter_ls = []

    n_quarter = len(quarter_ls)

    for i in range(0,len(quarter_ls)):
        #print (quarter_ls[i])
        lc = lk.search_lightcurve(target_name, mission=mission_name, cadence=cadence_name, quarter=quarter_ls[i]).download(quality_bitmask=0, flux_column="pdcsap_flux")
        ##################################################################
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
        ##################################################################
        x_quarter_ls.append(x.tolist())
        y_quarter_ls.append(y.tolist())
        w_quarter_ls.append(ivar.tolist())
        if plot == True:
            lc.plot()
    return x_quarter_ls, y_quarter_ls, w_quarter_ls, n_quarter


def remove_linear_trend(x_ls, y_ls):
    if len(np.shape(x_ls)) == 1:
        x_ls = np.reshape(x_ls, (1,len(x_ls)))
        y_ls = np.reshape(y_ls, (1,len(y_ls)))

    norm_yfit_ls = []
    
    for i in range(0,len(y_ls)):
       ##### lets fit a linear trend on x and y
        m, b = np.polyfit(x_ls[i], y_ls[i], 1)
        for j in range(0,len(x_ls[i])):
            norm_yy = m * x_ls[i][j] + b
            norm_yfit_ls.append(norm_yy)
    norm_y_ls = np.reshape(norm_yfit_ls, (np.shape(x_ls)))
    ls = y_ls-norm_y_ls
    return ls 


def minimum_cut (quarter_ls):
    minlen = len(min(quarter_ls,key=len))
    ls = []
    for i in range(0,len(quarter_ls)):
        ls.append(quarter_ls[i][:minlen])
    return ls



def split_matrices (flux, time, weights, train_percent=0.80, val_percent=0.10, seed=None, plot=True):
    # conver to numpy array 
    flux = np.array(flux)
    time = np.array(time)
    weights = np.array(weights)
    
    
    nflux = np.shape(flux)[1]
    ntime = np.shape(time)[1]
    
    flux_train, time_train = flux[:,: int(nflux*train_percent)], time[:,:int(ntime*train_percent)]
    flux_val, time_val = flux[:,int(nflux*train_percent): int(nflux*(train_percent+val_percent))], time[:,int(ntime*train_percent): int(ntime*(train_percent+val_percent))]
    flux_test, time_test = flux[:,int(nflux * (train_percent+val_percent)):], time[:,int(ntime* (train_percent+val_percent)):]
    train_weights = weights[:,: int(nflux*train_percent)]
    print ("total length:         ", np.shape(flux))
    print ("length of training:   ", np.shape(flux_train))
    print ("length of validation: ", np.shape(flux_val))
    print ("length of testing:    ", np.shape(flux_test))
    #print ("length of traing :    ", np.shape(train_weights))

    if plot==True:
        fig, ax = plt.subplots(1, figsize=(20,5))
        ax.yaxis.set_ticks_position('both') # ticks appear on both side 
        ax.xaxis.set_ticks_position('both') # ticks appear on both side 
        ax.tick_params(which='both', direction="in", width=2) # moves the major and minor ticks inside the plot 
        ax.minorticks_on() # turn on minor ticks 
        ax.tick_params(axis='x', which='minor', bottom=True)  # turn off minor ticks in x-axis
        ax.tick_params(axis='x', which='minor', top=True)     # turn off minor ticks in x-axis
        plt.scatter(time_train, flux_train, label = "train", s=1)
        plt.scatter(time_val, flux_val, label = "val ",s=1)
        plt.scatter(time_test, flux_test, label = "test",s=1)
        plt.xlabel ("Time [days]")
        plt.ylabel ("Norm Flux")
        plt.legend()
        plt.show()
    return flux_train, time_train, flux_val, time_val, flux_test, time_test, train_weights


def convert_list2tensor (ls):
    # if it's a 2-d array 
    if len(np.shape(ls)) > 1:
        ls_tensor = torch.FloatTensor(ls).reshape(np.shape(ls)[0],np.shape(ls)[1],1)
    else: 
        ls_tensor = torch.FloatTensor(ls).reshape(1, -1, 1)
    return ls_tensor
