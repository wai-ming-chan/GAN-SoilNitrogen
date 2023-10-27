import time
from sklearn.preprocessing import MinMaxScaler
import random
import os

from scipy.special import comb
import torch
torch.backends.cudnn.benchmark = True

# import self-defined functions
# from generator import Generator_simple, genNoise, genNoise_reduce, plot_GAN_training, trainGAN, trainGAN_RNN, trainGPGAN_RNN
# from predictor import train_prediction, plot_prediction_error, plot_prediction_sites
import GAN
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse

import sys

# Check if CUDA is available, otherwise use CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)  # Set the random seed to 42
torch.manual_seed(42) # Set random seed for reproducibility

start_time = time.time()
script_filename = os.path.basename(__file__) # Get the file name of the current script
script_name = os.path.splitext(script_filename)[0]
# print("Current script name:", sys.argv[0])
print(f'start running : {sys.argv[0]}')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
GAN_BATCH_SIZE = 16
GAN_NUM_EPOCH = 1000
GAN_NUM_DATA = 100
SEQ_SIZE = 5
DAY_STEP = 5 
GAN_K_FOLDS = 5
GAN_TRAIN_RATE = 0.8
K_STEP = 1

GAN_RNN = True
GAN_CNN = ~GAN_RNN
PRED_RNN = True
PRED_CNN = ~PRED_RNN
GAN_PLOT = False
# NUM_FEATURE = 3 # number of training features (1-8)
COLS_FEATURES = ['initial_N_rate', 'other_N', 'day_cos', 'day_sin' ,'SRAD','DTTD','N_uptake',  'TMAX', 'TMIN']

dataset_filename = "data/data_AGAI_DSSAT_scale.csv"

#########################################################################################
#  Generator Hyper-parameters
#########################################################################################
GEN_input_size = 4 #6
GEN_hidden_size = 256 #64
GEN_num_layer = 1
GEN_dropout_prob = 0.1
GEN_lr = 1e-3

GEN_cond_dim = 4; # first 4 features are used as conditioning features
GAN_clip_value = 0.01 # Lower and upper clip value for disc. weights.
GAN_n_critic = 5; # Number of training steps for discriminator per iter. (Default: 5).
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
DIS_hidden_size = 256 #64
DIS_num_layer = 1
DIS_dropout_prob = 0.2
DIS_lr = 1e-3

#########################################################################################
#  Prediction  Hyper-parameters
#########################################################################################
PRED_lr = 1e-2
PRED_EPOCH = 300

#########################################################################################
#  GAN training
#########################################################################################

all_error_noGAN = np.zeros(PRED_EPOCH) 
all_error_GAN = np.zeros(PRED_EPOCH)
all_error_GAN_vanilla = np.zeros(PRED_EPOCH)
# all_error_GAN_cond = np.zeros(PRED_EPOCH)
all_error_GAN_WGAN = np.zeros(PRED_EPOCH)

all_loss_noGAN = np.zeros(PRED_EPOCH) 
all_loss_GAN = np.zeros(PRED_EPOCH)
all_loss_GAN_vanilla = np.zeros(PRED_EPOCH)
# all_error_GAN_cond = np.zeros(PRED_EPOCH)
all_loss_GAN_WGAN = np.zeros(PRED_EPOCH)

all_R2_noGAN = np.zeros(PRED_EPOCH) 
all_R2_GAN = np.zeros(PRED_EPOCH)
all_R2_GAN_vanilla = np.zeros(PRED_EPOCH)
# all_R2_GAN_cond = np.zeros(PRED_EPOCH)
all_R2_GAN_WGAN = np.zeros(PRED_EPOCH)


NUM_MC = 5
for m in range(NUM_MC):
    # print(f'running pass:{m:3d}')

    error_noGAN, r2_noGAN = GAN.train_noGAN(GAN_BATCH_SIZE, SEQ_SIZE,DAY_STEP, GAN_NUM_EPOCH, GAN_NUM_DATA, GAN_TRAIN_RATE,GEN_input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr,DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr,PRED_lr,PRED_EPOCH, K_STEP,COLS_FEATURES)
    error_GPGAN, r2_GPGAN = GAN.trainGPGAN(GAN_BATCH_SIZE, SEQ_SIZE,DAY_STEP, GAN_NUM_EPOCH, GAN_NUM_DATA, GAN_TRAIN_RATE,GEN_input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr,DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, PRED_lr,PRED_EPOCH, K_STEP,COLS_FEATURES)
    loss_d_GAN_vanilla, loss_g_GAN_vanilla, error_GAN_vanilla, r2_GAN_vanilla = GAN.trainGAN_vanilla(GAN_BATCH_SIZE,DAY_STEP, SEQ_SIZE, GAN_NUM_EPOCH, GAN_NUM_DATA, GAN_TRAIN_RATE,GEN_input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr,DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, PRED_lr,PRED_EPOCH, K_STEP,COLS_FEATURES)
    loss_d_GAN_WGAN, loss_g_GAN_WGAN, error_GAN_WGAN, r2_GAN_WGAN = GAN.trainGAN_WGAN(GAN_BATCH_SIZE, SEQ_SIZE,DAY_STEP, GAN_NUM_EPOCH, GAN_NUM_DATA, GAN_TRAIN_RATE,GEN_input_size, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr,GAN_clip_value, GAN_n_critic, DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, PRED_lr,PRED_EPOCH, K_STEP,COLS_FEATURES)
    # error_GAN_cond, r2_GAN_cond = GAN.trainGAN_Cond(GAN_BATCH_SIZE, SEQ_SIZE,DAY_STEP, GAN_NUM_EPOCH, GAN_NUM_DATA, GAN_TRAIN_RATE,GEN_input_size,GEN_cond_dim, GEN_hidden_size, GEN_num_layer, GEN_dropout_prob, GEN_lr,DIS_hidden_size, DIS_num_layer, DIS_dropout_prob, DIS_lr, PRED_lr,PRED_EPOCH, K_STEP,COLS_FEATURES)

    # print(f'[exp-{m}] rmse (noGAN): {error_noGAN[-1]:.4f} | (GAN vanilla): {error_GAN_vanilla[-1]:.4f} | (GPGAN): {error_GPGAN[-1]:.4f} | (WGAN): {error_GAN_WGAN[-1]:.4f}') 
    # print(f'[exp-{m}] R2 (noGAN): {r2_noGAN[-1]:.4f} | (GAN vanilla): {r2_GAN_vanilla[-1]:.4f} | (GPGAN): {r2_GPGAN[-1]:.4f} | (WGAN): {r2_GAN_WGAN[-1]:.4f}') 
    
    all_error_noGAN += (error_noGAN) 
    all_error_GAN += (error_GPGAN[-PRED_EPOCH:])
    all_error_GAN_vanilla += error_GAN_vanilla
    # all_error_GAN_cond += error_GAN_cond
    all_error_GAN_WGAN += error_GAN_WGAN

    all_R2_noGAN    += r2_noGAN
    all_R2_GAN      +=  r2_GPGAN[-PRED_EPOCH:]
    all_R2_GAN_vanilla += r2_GAN_vanilla
    # all_R2_GAN_cond += r2_GAN_cond
    all_R2_GAN_WGAN += r2_GAN_WGAN


    fig, ax = plt.subplots()
    ax.semilogy(error_noGAN, linestyle='-', color='b', label=f'noGAN : {error_noGAN[-1]:.4f}')
    ax.semilogy(error_GPGAN[-PRED_EPOCH:], linestyle='-', color='r', label=f'GPGAN : {error_GPGAN[-1]:.4f}')
    ax.semilogy(error_GAN_vanilla, linestyle='-', color='g', label=f'GAN vanilla : {error_GAN_vanilla[-1]:.4f}')
    # ax.semilogy(error_GAN_cond, linestyle='-', color='m', label=f'CGAN : {error_GAN_cond[-1]:.4f}')
    ax.semilogy(error_GAN_WGAN, linestyle='-', color='y', label=f'WGAN : {error_GAN_WGAN[-1]:.4f}')
    # ax.set_ylim(0,1)
    ax.set_ylim(1e-2, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test RMSE')
    ax.set_title(f'Error Plot [Seq:{SEQ_SIZE}] [Train_rate:{GAN_TRAIN_RATE}] D_lr:{DIS_lr}, G_lr:{GEN_lr}')
    ax.legend()
    ax.grid(True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figs/{script_name}_WGAN_rmse_{current_time}.png')
    plt.show()
    plt.close()

    if GAN_PLOT:
        fig, ax = plt.subplots()
        ax.plot(error_GAN_vanilla[0], linestyle='--', color='r', label=f'D loss')
        ax.plot(error_GAN_vanilla[1], linestyle='-', color='b', label=f'G loss')
        ax.set_ylim(0,5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'GAN training Plot D_lr:{DIS_lr}, G_lr:{GEN_lr}')
        ax.legend()
        ax.grid(True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'figs/{script_name}_GANoriginal_loss_{current_time}.png')
        plt.show()
        plt.close()


avg_error_noGAN = (all_error_noGAN) / NUM_MC
avg_error_GAN = (all_error_GAN) / NUM_MC
avg_error_GAN_vanilla = (all_error_GAN_vanilla) / NUM_MC
# avg_error_GAN_cond = (all_error_GAN_cond) / NUM_MC
avg_error_GAN_WGAN = (all_error_GAN_WGAN) / NUM_MC

all_R2_noGAN    = all_R2_noGAN / NUM_MC
all_R2_GAN      =  all_R2_GAN / NUM_MC
all_R2_GAN_vanilla = all_R2_GAN_vanilla / NUM_MC
# all_R2_GAN_cond = all_R2_GAN_cond / NUM_MC
all_R2_GAN_WGAN = all_R2_GAN_WGAN / NUM_MC

print(f'[AVG] rmse (noGAN): {avg_error_noGAN[-1]:.4f} | (GAN vanilla): {avg_error_GAN_vanilla[-1]:.4f} | (GPGAN): {avg_error_GAN[-1]:.4f} | (WGAN): {avg_error_GAN_WGAN[-1]:.4f}') 
print(f'[AVG] R2 (noGAN): {all_R2_noGAN[-1]:.4f} | (GAN vanilla): {all_R2_GAN_vanilla[-1]:.4f} | (GPGAN): {all_R2_GAN[-1]:.4f} | (WGAN): {all_R2_GAN_WGAN[-1]:.4f}') 
    

output_filename = f"{script_filename}_output.csv"
data = np.column_stack((all_R2_noGAN, all_R2_GAN, all_R2_GAN_vanilla, all_R2_GAN_WGAN))
# Save the data to the CSV file
np.savetxt(output_filename, data, delimiter=",", header="R2_noGAN, R2_GAN, R2_GAN_vanilla, R2_GAN_WGAN", comments="")
print(f"Data saved to {output_filename}")








fig, ax = plt.subplots()
ax.semilogy(avg_error_noGAN, linestyle='-', color='b', label=f'noGAN : {avg_error_noGAN[-1]:.4f}')
ax.semilogy(avg_error_GAN, linestyle='-', color='r', label=f'GPGAN : {avg_error_GAN[-1]:.4f}')
ax.semilogy(avg_error_GAN_vanilla, linestyle='-', color='g', label=f'GAN vanilla: {avg_error_GAN_vanilla[-1]:.4f}')
# ax.semilogy(avg_error_GAN_cond, linestyle='-', color='m', label=f'CGAN : {avg_error_GAN_cond[-1]:.4f}')
ax.semilogy(avg_error_GAN_WGAN, linestyle='-', color='y', label=f'WGAN: {avg_error_GAN_WGAN[-1]:.4f}')

# ax.set_ylim(0,1)
ax.set_ylim(1e-2, 1)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test RMSE')
ax.set_title(f'[Average] Error Plot [Seq:{SEQ_SIZE}] [Train_rate:{GAN_TRAIN_RATE}] D_lr:{DIS_lr}, G_lr:{GEN_lr}')
ax.legend()
ax.grid(True)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'figs/{script_name}_AvgGPGAN_rmse_{current_time}.png')
plt.show()
plt.close()