
import torch
import time 
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
from random import randint
import random 
import copy 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence




def weighted_mae_loss (output, target, weight):
    loss = torch.mean(weight*torch.abs(target-output))
    return loss 

def weighted_mse_loss(output, target, weight):
    #source: https://discuss.pytorch.org/t/custom-loss-functions/29387/3
    loss = torch.mean(weight*(output - target)**2)
    return loss

def weighted_huber_loss(output, target, weight, delta=5):
	loss = torch.where(torch.abs(target-output) < delta, 0.5*(target-output)**2),  delta*torch.abs(target - output) - 0.5*(delta**2)
	return torch.sum(loss * weight)

def create_boolean_mask_sequence(input, mask_percent, seq_size, output_size =1):
    '''
    @param input: flux  
    @param mask_percent: ideally how many points to mask of the flatten light curve 
    @param seq_size: tuple of the size of sequence to be masked 
    @param output_size: number of features, in this case is set to 1 
    
    @output array: a boolean mask that's made of 0's and 1's that is the same shape as the input array 
    '''
    # get the shape of the input array 
    x = np.shape(input)[0]
    y = np.shape(input)[1]
    # we want to know the total length of the training data 
    length = len(input.flatten())
    # the number of zeros we want to add in the input array 
    num_zeros= int (length*mask_percent)
    array = np.ones((length))
    # inital sum_steps is step to 0
    sum_steps = 0 
    # list of a randomize number of steps 
    random_step_ls =[]
    
    while sum_steps <= num_zeros:
        # get a value that ranges from 
        random_step = randint(seq_size[0],seq_size[1])
        # update the sum_steps parameter
        sum_steps +=random_step
        # save this value for later 
        random_step_ls.append(random_step)
    # lets get random starting values 
    startindex = random.sample(range(0, len(array)), len(random_step_ls))
    
    # update the array 
    for i in range(0,len(startindex)):
        array[startindex[i]:startindex[i] + random_step_ls[i]] = 0
    array = torch.FloatTensor(array).reshape(x,y, output_size)
    return array 


train_colors = cm.Dark2(np.linspace(0,1,10))

def train_model (model, x_train, y_train, w_train, optimizer, n_quarter, null, mask_percent, seq_size, n_epochs = 1000, criterion='mse', save_file = False):
    
    start = time.time()
    tr_loss =[]

    for epoch in range(n_epochs): 

        # creates a random mask each epoch 
        m = create_boolean_mask_sequence(y_train, mask_percent, seq_size)
        # lets apply this mask to the y_train values 
        masked_y_train = y_train* m 
        masked_y_train[masked_y_train==0] = null
                
        train_dataset = TensorDataset(x_train, masked_y_train, y_train, w_train, m)
        loader = DataLoader(train_dataset, shuffle=True, batch_size = n_quarter) 

        # the inital sum of the mean square error is set to 0 
        sum_loss_train = 0

        # Clears existing gradients from previous epoch
        model.zero_grad()

        # trains the model 
        model.train()

        for batch_idx, (x_train, masked_y_train, y_train, w_train, m) in enumerate(loader):

            # inputting the masked_y_train into the model/ forward propogation  
            out_lstm, hidden = model(masked_y_train.float())

            # Creates a criterion that measures the mean squared error between each element in the input 
            # this is the loss function that we are using 
            if criterion == 'mse':
                # need to conver the boolean mask array to boolean array :) 
                m = m.type(dtype=torch.bool)
                loss = weighted_mse_loss(out_lstm, y_train, w_train) #*(~m)
            elif criterion == 'mae':
                loss = weighted_mae_loss(out_lstm, y_train, w_train)
            elif criterion == 'huber':
                loss = weighted_huber_loss(out_lstm, y_train, w_train)
            else:
                loss = criterion (out_lstm, y_train)

            # apend the lost 
            tr_loss.append(loss.item())
            
            sum_loss_train += loss

            loss.backward()

            # Updates the model's weights :) 
            optimizer.step() 
            
            # update the scheduler's learning rate 
            #scheduler.step(loss)

            # lets keep the best weights for the lowest training loss :) 
            if (sum_loss_train) == np.min(tr_loss):
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'uni_weights_only.pth')

            if epoch % 100 ==0:
                with torch.no_grad():
                    # does not learn new weights -- evalulates the model 
                    model.eval()
                    out_lstm, hidden = model(y_train)
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
                    ax0.scatter(x_train, y_train, s=5, label="Output", color = train_colors[0])
                    ax0.scatter(x_train, masked_y_train, s=5, label="Input", color = train_colors[2])
                    ax0.scatter(x_train, out_lstm, s=5, alpha = 0.6, label='Pred', color = train_colors[3])
                    res = (torch.reshape(y_train, (-1,))).cpu().detach().numpy()-(torch.reshape(out_lstm, (-1,))).cpu().detach().numpy()
                    ax1.scatter(x_train, res, s=6, color ='k')
                    ymax = np.max(res)
                    ymin = -1*ymax
                    #xmin = 475
                    #xmax = 500
                    ax1.axhline(y=0, color = 'red', linestyle = '--')

                    leg = ax0.legend(loc='lower left', bbox_to_anchor= (0.02, 1.01), markerscale=4., ncol=3, borderaxespad=0, handletextpad=0.8, frameon=False)
                    ax0.set_title("Epoch : "+ str(epoch) + '/' + str(n_epochs))
                    ax1.set_ylim([ymin, ymax])
                    #ax1.set_xlim([xmin,xmax])
                    #ax1.set_xlim([270,285])
                    plt.tight_layout()
                    ax1.set_title('LOSS =' + str(np.round(loss.item(),3)), loc='right', fontsize=12)
                    # lets plot the MSE on the residuals plot
                    #fig.text(0.2, 0.3,'MSE =' + str(np.round(mse.item(),3)), ha='center', va='center', color = 'k')
                    if save_file == True:
                        plot_filename = "lstm_training_epoch"+epoch+".png"
                        plt.savefig(plot_filename, bbox_inches='tight', dpi=500)
                        print (plot_filename, "has been saved")
                    plt.show()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, tr_loss


def train_model_lag (model, loader, lag, optimizer, n_epochs = 1000, criterion='mse', save_file = False):
    
    start = time.time()
    tr_loss = [] 
    
    for epoch in range(n_epochs): 

        # the inital sum of the mean square error is set to 0 
        sum_loss_train = 0

        # Clears existing gradients from previous epoch
        model.zero_grad()

        # trains the model 
        model.train()

        for batch_idx, (x_input_train, x_output_train, w_output_train, y_input_train, y_output_train) in enumerate(loader):
            
            #print (np.shape(x_input_train))
            
            # puts the training data into the model, outputting the val+hidden state
            out_lstm, hidden = model(y_input_train)

            # Creates a criterion that measures the mean squared error between each element in the input 
            # this is the loss function that we are using 
            if criterion == 'mse':
                loss = weighted_mse_loss(out_lstm, y_output_train, w_output_train)
            elif criterion == 'mae':
                loss = weighted_mae_loss(out_lstm, y_output_train, w_output_train)
            elif criterion == 'huber':
                loss = weighted_huber_loss(out_lstm, y_output_train, w_output_train)
            else:
                loss = criterion (out_lstm, y_output_train)

            # to get the mse so we can see how that changes over time 
            tr_loss.append(loss.item())

            sum_loss_train += loss

            loss.backward()
            
            # Updates the model's weights :) 
            optimizer.step() 

        # lets keep the best weights for the lowest training loss :) 
        if (sum_loss_train) == np.min(tr_loss):
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'uni_weights_only.pth')

        if epoch % 100 ==0:
            with torch.no_grad():
                # does not learn new weights -- evalulates the model 
                model.eval()
                out_lstm, hidden = model(y_input_train)
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

                ax0.scatter(x_input_train, y_input_train, s=8, label="input", color = train_colors[0])
                ax0.scatter(x_output_train, y_output_train, s=8, label='truth', color = train_colors[2])
                ax0.scatter(x_output_train, out_lstm, s=8, alpha = 0.6, label='pred', color = train_colors[3])
                res = (torch.reshape(y_output_train, (-1,))).cpu().detach().numpy()-(torch.reshape(out_lstm, (-1,))).cpu().detach().numpy()
                ax1.scatter(x_output_train, res, s=6, color ='k')
                ymax = np.max(res)
                ymin = -1*ymax
                ax1.axhline(y=0, color = 'red', linestyle = '--')

                leg = ax0.legend(loc='lower left', bbox_to_anchor= (0.02, 1.01), markerscale=4., ncol=3, borderaxespad=0, handletextpad=0.8, frameon=False)
                ax0.set_title("Epoch : "+ str(epoch) + '/' + str(n_epochs) +  " ; Lag : "+ str(lag), loc='right')
                ax1.set_ylim([ymin, ymax])
                plt.tight_layout()
                ax1.set_title('Loss =' + str(np.round(loss.item(),3)), loc='right', fontsize=12)
                if save_file == True:
                    plot_filename = "lstm_training_lag"+str(lag)+"_epoch"+epoch+".png"
                    plt.savefig(plot_filename, bbox_inches='tight', dpi=500)
                    print (plot_filename, "has been saved")
                plt.show()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, tr_loss