import torch.nn as nn
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch.utils.data as data
import predictor as P_models
# import Prediction_GRU
import dataloader_v2 as DSSAT
import generator as G_models
import discriminator as D_models
# from torcheval.metrics import R2Score
from torchmetrics.functional import r2_score

# from torchsummary import summary
from torchinfo import summary
import sys
import numpy as np
# ######################################################################################################################
# def trainGPGAN_noGAN(BATCH_SIZE, SEQ_LENGTH,DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
#                 GEN_INPUT_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, \
#                 DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
#                 K_STEP):
#     print(f'start of trainGPGAN')
    

#     dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
#     cols_feature = ['SRAD','DTTD','N_uptake']
#     cols_label = ['inorganic_N']
#     num_features = len(cols_feature)
#     num_labels = len(cols_label)

#     myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
#     X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.empty_cache()
#     X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

#     dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
#     # Generator & Predictor (or Discriminator)
#     generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

#     predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

#     LOAD_PRED_MODEL = False
#     if LOAD_PRED_MODEL:
#         predictor.load_state_dict(torch.load('prediction_gru_model.pth'))
#         print('Model predictor is loaded.')
#     # sys.exit(0)

#     # Define the loss functions and optimizers
#     criterion = nn.MSELoss()
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
#     optimizer_P = torch.optim.Adam(predictor.parameters(), lr=DIS_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    

#     error_output, error_output_noGAN, error_output_GAN = [], [], []
#     # 1) training without GAN, prediction only
#     # Training loop
#     for epoch in range(NUM_EPOCH):
#         with torch.no_grad():
#             label_test_true = y_test[:,-1,:].to(device)
#             label_test_pred = predictor(X_test.to(device))
#             p_loss_test = criterion(label_test_pred, label_test_true)
#             error_test = np.sqrt(p_loss_test.item())

#         error_train =0 
#         for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
#             batch_len,seq_len,feature_dim = real_data[0].shape
#             #----- Train the Predictor (equiv Discriminator) --------------------------------------
#             predictor.zero_grad()
#             optimizer_P.zero_grad()

#             label_true = real_data[1][:,-1,:].to(device)
#             label_pred = predictor(real_data[0].to(device))
#             p_loss = criterion(label_pred, label_true)
#             p_loss.backward()     
#             optimizer_P.step()       
#             error_train_P = np.sqrt(p_loss.item())
            
        

#         if True:
#             print(f'[noGAN] epoch:{epoch:4d} | train mse:{error_train:.4f} | test mse:{error_test:.4f}') 

#         error_output_noGAN.append(error_test)

#     for _ in range(10):
#         with torch.no_grad():
#             label_test_true = y_test[:,-1,:].to(device)
#             label_test_pred = predictor(X_test.to(device))
#             p_loss_test = criterion(label_test_pred, label_test_true)
#             error_test = np.sqrt(p_loss_test.item())
#             print(f'AFTER P-training, error_test:{error_test:.4f}')

#     # 2) training with GAN
#     # Training loop
#     for epoch in range(NUM_EPOCH):
#         with torch.no_grad():
#             label_test_true = y_test[:,-1,:].to(device)
#             label_test_pred = predictor(X_test.to(device))
#             p_loss_test = criterion(label_test_pred, label_test_true)
#             error_test = np.sqrt(p_loss_test.item())

#         error_train =0 
#         for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
#             batch_len,seq_len,feature_dim = real_data[0].shape
#             #----- Train the Predictor (equiv Discriminator) --------------------------------------
#             predictor.zero_grad()
#             optimizer_P.zero_grad()

#             label_true = real_data[1][:,-1,:].to(device)
#             label_pred = predictor(real_data[0].to(device))
#             p_loss = criterion(label_pred, label_true)
#             p_loss.backward()     
#             optimizer_P.step()       
#             error_train_P = np.sqrt(p_loss.item())

#             #----- Train the Generator --------------------------------------
#             generator.zero_grad()
#             optimizer_G.zero_grad()

#             noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
#             fake_data = generator(noise)
#             fake_features = fake_data[:,:,:num_features]
#             fake_labels = fake_data[:,-1,num_features].unsqueeze(1)
#             fake_labels_pred = predictor(fake_features)
            
#             g_loss = criterion(fake_labels_pred, fake_labels)
#             g_loss.backward()  
#             optimizer_G.step()
#             error_train_G = np.sqrt(g_loss.item())

#         # with torch.no_grad():
#         #     label_test_true = y_test[:,-1,:].to(device)
#         #     label_test_pred = predictor(X_test.to(device))
            
#         #     p_loss_test = criterion(label_test_pred, label_test_true)
#         #     error_test = np.sqrt(p_loss_test.item())
        
#         if True:
#             print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}') 

#         error_output_GAN.append(error_test)



#     return error_output_noGAN, error_output_GAN


######################################################################################################################
def trainGAN_vanilla(BATCH_SIZE, SEQ_LENGTH,DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
                GEN_INPUT_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, \
                DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
                PRED_lr,PRED_EPOCH, \
                K_STEP, COLS_FEATURE ):
    if False:
        print(f'start of trainGPGAN')
    

    dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
    # cols_feature = ['SRAD','DTTD','N_uptake']
    cols_feature = COLS_FEATURE
    cols_label = ['inorganic_N']
    num_features = len(cols_feature)
    num_labels = len(cols_label)

    myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
    # X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence(seq_len=SEQ_LENGTH, train_rate=TRAIN_RATE, flag_fakedata=False)
    X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

    dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    # Generator & Predictor (or Discriminator)
    generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    discriminator = D_models.Discriminator_GRU(input_size=num_features+num_labels, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    # sys.exit(0)

    # Define the loss functions and optimizers
    criterionBCE = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    
    
    rmse_output = []
    R2_output     = []
    loss_d_output = []
    loss_g_output = []
    # Training loop
    for epoch in range(NUM_EPOCH * 2):

        error_GAN_d =0
        error_GAN_g =0
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            # print(f'GAN-line202: {batch_len} / {seq_len} / {feature_dim}')

            real_labels     = torch.ones(batch_len, 1, device=device)
            # real_labels_gen = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)

            #----- Train the Discriminator --------------------------------------
            discriminator.zero_grad()
            optimizer_D.zero_grad()

            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
            fake_data = generator(noise)
            output_D_real = discriminator( torch.cat((real_data[0], real_data[1]), dim=2).to(device) )
            output_D_fake = discriminator( fake_data.detach() )

            d_loss_real = criterionBCE( output_D_real, real_labels)
            d_loss_fake = criterionBCE( output_D_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_D.step()

            # # print(f'label_true:{np.shape(label_true)} | label_pred:{np.shape(label_pred)}') 
            # p_loss = criterionMSE(label_pred, label_true)
            # p_loss.backward()     
            # optimizer_P.step()       
            error_GAN_d = (d_loss.item())


            #----- Train the Generator --------------------------------------
            generator.zero_grad()
            optimizer_G.zero_grad()
            
            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
            fake_data = generator(noise)
            if False:
                fake_features = fake_data[:,:,:num_features]
                fake_labels = fake_data[:,-1,num_features].unsqueeze(1)
                fake_labels_pred = predictor(fake_features)
            output_D_fake = discriminator(fake_data)
            g_loss = criterionBCE( output_D_fake, real_labels)

            # g_loss = criterionMSE(fake_labels_pred, fake_labels)
            g_loss.backward()  
            optimizer_G.step()
            error_GAN_g = (g_loss.item())

        if False:
            with torch.no_grad():
                label_test_true = y_test[:,-1,:].to(device)
                label_test_pred = predictor(X_test.to(device))
                
                p_loss_test = criterionMSE(label_test_pred, label_test_true)
                error_test = np.sqrt(p_loss_test.item())
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}') 

        loss_d_output.append(error_GAN_d)
        loss_g_output.append(error_GAN_g)

    ######################## Start Prediction Training ############################
    ## Generate Fake data
    noise = torch.randn([NUM_FAKE_DATA, SEQ_LENGTH, GEN_INPUT_SIZE])
    with torch.no_grad():
        fake_data = generator(noise.to(device))
    fake_data_feature, fake_data_label = fake_data[:,:,:num_features], fake_data[:,:,num_features]
    X_train_GAN= torch.cat((X_train, fake_data_feature.cpu()), dim=0)
    y_train_GAN  = torch.cat((y_train, fake_data_label.unsqueeze(-1).cpu()), dim=0)
    dataloaderGAN = data.DataLoader(data.TensorDataset(X_train_GAN, y_train_GAN), shuffle=True, batch_size=BATCH_SIZE)

    criterionMSE = nn.MSELoss()
    predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=PRED_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    for epoch in range(PRED_EPOCH):
        error_train =0 
        error_test =0
        for batch_idx, real_data in enumerate(dataloaderGAN): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            #----- Train the Predictor --------------------------------------
            predictor.zero_grad()
            optimizer_P.zero_grad()
            label_true = real_data[1][:,-1,:].to(device)
            label_pred = predictor(real_data[0].to(device))
            p_loss = criterionMSE(label_pred, label_true)
            p_loss.backward()     
            optimizer_P.step()       
            error_train = np.sqrt(p_loss.item())

            
        with torch.no_grad():
            label_test_true = y_test[:,-1,:].to(device)
            label_test_pred = predictor(X_test.to(device))
            p_loss_test = criterionMSE(label_test_pred, label_test_true)
            error_test = np.sqrt(p_loss_test.item()) 
            rmse_output.append(error_test)

            r2_test = r2_score(label_test_pred, label_test_true).cpu()
            R2_output.append(r2_test)
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}')
    return loss_d_output, loss_g_output, rmse_output, R2_output


######################################################################################################################

def trainGPGAN(BATCH_SIZE, SEQ_LENGTH,DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
                GEN_INPUT_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, \
                DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
                PRED_lr,PRED_EPOCH, \
                K_STEP, COLS_FEATURE):
    if False:
        print(f'start of trainGPGAN')
    

    dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
    # cols_feature = ['SRAD','DTTD','N_uptake']
    cols_feature = COLS_FEATURE
    cols_label = ['inorganic_N']
    num_features = len(cols_feature)
    num_labels = len(cols_label)

    myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
    # X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence(seq_len=SEQ_LENGTH, train_rate=TRAIN_RATE, flag_fakedata=False)
    X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

    dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    # Generator & Predictor (or Discriminator)
    generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    LOAD_PRED_MODEL = False
    if LOAD_PRED_MODEL:
        predictor.load_state_dict(torch.load('prediction_gru_model.pth'))
        print('Model predictor is loaded.')
    # sys.exit(0)

    # Define the loss functions and optimizers
    criterion = nn.MSELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=DIS_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    
    R2_output    = []
    error_output = []
    # Training loop
    for epoch in range(NUM_EPOCH):

        error_train =0 
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            #----- Train the Predictor (equiv Discriminator) --------------------------------------
            predictor.zero_grad()
            optimizer_P.zero_grad()

            label_true = real_data[1][:,-1,:].to(device)
            label_pred = predictor(real_data[0].to(device))
            # print(f'label_true:{np.shape(label_true)} | label_pred:{np.shape(label_pred)}') 
            p_loss = criterion(label_pred, label_true)
            p_loss.backward()     
            optimizer_P.step()       
            error_train_P = np.sqrt(p_loss.item())


            #----- Train the Generator --------------------------------------
            generator.zero_grad()
            optimizer_G.zero_grad()
            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
            fake_data = generator(noise)
            fake_features = fake_data[:,:,:num_features]
            fake_labels = fake_data[:,-1,num_features].unsqueeze(1)
            fake_labels_pred = predictor(fake_features)

            g_loss = criterion(fake_labels_pred, fake_labels)
            g_loss.backward()  
            optimizer_G.step()
            error_train_G = np.sqrt(g_loss.item())

            # xxx= torch.cat((real_data[0], real_data[1]), dim=2)
            # print(f'real_data:{np.shape(real_data[0])} + {np.shape(real_data[1])}=>{np.shape(xxx)} | fake_data:{np.shape(fake_data)}') 
        with torch.no_grad():
            label_test_true = y_test[:,-1,:].to(device)
            label_test_pred = predictor(X_test.to(device))
            
            p_loss_test = criterion(label_test_pred, label_test_true)
            error_test = np.sqrt(p_loss_test.item())
            
            r2_test = r2_score(label_test_pred, label_test_true).cpu()
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}') 

        error_output.append(error_test)
        R2_output.append(r2_test)
    return error_output, R2_output


######################################################################################################################
def train_noGAN(BATCH_SIZE, SEQ_LENGTH, DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
                GEN_INPUT_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, \
                DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
                PRED_lr,PRED_EPOCH, \
                K_STEP, COLS_FEATURE):
    # print(f'start of trainGPGAN')

    dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
    # cols_feature = ['SRAD','DTTD','N_uptake']
    cols_feature = COLS_FEATURE
    cols_label = ['inorganic_N']
    num_features = len(cols_feature)
    num_labels = len(cols_label)

    myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
    # X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence(seq_len=SEQ_LENGTH, train_rate=TRAIN_RATE, flag_fakedata=False)
    X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

    dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    # Generator & Predictor (or Discriminator)
    # generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    # sys.exit(0)

    # Define the loss functions and optimizers
    criterion = nn.MSELoss()
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=PRED_lr, weight_decay=1e-5, betas = (0.5, 0.9))
    # R2metric = R2Score()

    error_output    = []
    R2_output       = []

    # Training loop
    for epoch in range(PRED_EPOCH):

        error_train =0 
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            #----- Train the Predictor (equiv Discriminator) --------------------------------------
            predictor.zero_grad()
            optimizer_P.zero_grad()
            label_true = real_data[1][:,-1,:].to(device)
            label_pred = predictor(real_data[0].to(device))
            # print(f'label_true:{np.shape(label_true)} | label_pred:{np.shape(label_pred)}') 
            p_loss = criterion(label_pred, label_true)
            p_loss.backward()     
            optimizer_P.step()       
            error_train = np.sqrt(p_loss.item())

            
        with torch.no_grad():
            label_test_true = y_test[:,-1,:].to(device)
            label_test_pred = predictor(X_test.to(device))
            p_loss_test = criterion(label_test_pred, label_test_true)
            error_test = np.sqrt(p_loss_test.item())

            r2_test = r2_score(label_test_pred, label_test_true).cpu()
        if False:
            print(f'[noGAN] epoch:{epoch:4d} | train mse:{error_train:.4f} | test mse:{error_test:.4f}') 

        error_output.append(error_test)
        R2_output.append(r2_test)

    # torch.save(predictor.state_dict(), 'prediction_gru_model.pth')
    # print('Model predictor is saved.')
    return error_output, R2_output 



######################################################################################################################
def trainGAN_Cond(BATCH_SIZE, SEQ_LENGTH,DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
                GEN_INPUT_SIZE, GEN_COND_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, \
                DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
                PRED_lr,PRED_EPOCH, \
                K_STEP, COLS_FEATURE ):
    if False:
        print(f'start of trainGPGAN')
    

    dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
    # cols_feature = ['SRAD','DTTD','N_uptake']
    cols_feature = COLS_FEATURE
    cols_label = ['inorganic_N']
    num_features = len(cols_feature)
    num_labels = len(cols_label)

    myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
    # X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence(seq_len=SEQ_LENGTH, train_rate=TRAIN_RATE, flag_fakedata=False)
    X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

    dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    # Generator & Predictor (or Discriminator)
    # generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    discriminator = D_models.Discriminator_GRU(input_size=num_features+num_labels, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    generator_cond = G_models.Generator_GRU_COND(input_size=GEN_INPUT_SIZE, cond_input_size=GEN_COND_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    # sys.exit(0)

    # Define the loss functions and optimizers
    criterionBCE = nn.BCELoss()
    
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_G_cond = torch.optim.Adam(generator_cond.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    
   

    rmse_output = []
    loss_d_output = []
    loss_g_output = []
    R2_output = []
    # Training loop
    for epoch in range(NUM_EPOCH * 2):

        error_GAN_d =0
        error_GAN_g =0
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            feature_cond = real_data[0][:, :, 0:GEN_COND_SIZE].to(device)
            real_feature_label = torch.cat((real_data[0], real_data[1]), dim=2).to(device)
            real_labels     = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)

            #----- Train the Discriminator --------------------------------------
            discriminator.zero_grad()
            optimizer_D.zero_grad()

            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
            fake_data = generator_cond(noise, feature_cond)


            output_D_real = discriminator( real_feature_label )
            output_D_fake = discriminator( fake_data.detach() )

            d_loss_real = criterionBCE( output_D_real, real_labels)
            d_loss_fake = criterionBCE( output_D_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizer_D.step()

            # # print(f'label_true:{np.shape(label_true)} | label_pred:{np.shape(label_pred)}') 
            # p_loss = criterionMSE(label_pred, label_true)
            # p_loss.backward()     
            # optimizer_P.step()       
            error_GAN_d = (d_loss.item())


            #----- Train the Generator --------------------------------------
            generator_cond.zero_grad()
            optimizer_G_cond.zero_grad()
            
            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)
            fake_data = generator_cond(noise, feature_cond)
            if False:
                fake_features = fake_data[:,:,:num_features]
                fake_labels = fake_data[:,-1,num_features].unsqueeze(1)
                fake_labels_pred = predictor(fake_features)
            output_D_fake = discriminator(fake_data)
            g_loss = criterionBCE( output_D_fake, real_labels)

            # g_loss = criterionMSE(fake_labels_pred, fake_labels)
            g_loss.backward()  
            optimizer_G_cond.step()
            error_GAN_g = (g_loss.item())

        if False:
            with torch.no_grad():
                label_test_true = y_test[:,-1,:].to(device)
                label_test_pred = predictor(X_test.to(device))
                
                p_loss_test = criterionMSE(label_test_pred, label_test_true)
                error_test = np.sqrt(p_loss_test.item())
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}') 

        loss_d_output.append(error_GAN_d)
        loss_g_output.append(error_GAN_g)

    ######################## Start Prediction Training ############################
    ## Generate Fake data
    X_train_GAN, y_train_GAN = X_train, y_train 
    dup_factor = int(np.ceil(NUM_FAKE_DATA / np.shape(X_train_GAN)[0] ))
    # print(f'X_train_GAN: {np.shape(X_train_GAN)} | y_train_GAN: {np.shape(y_train_GAN)}')
    for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
        batch_len,seq_len,feature_dim = real_data[0].shape
        feature_cond = real_data[0][:, :, 0:GEN_COND_SIZE].to(device)
        feature_cond = feature_cond.repeat(dup_factor, 1, 1)
        noise = torch.randn([batch_len * dup_factor, SEQ_LENGTH, GEN_INPUT_SIZE]).to(device)
        with torch.no_grad():
            # print(f'noise: {np.shape(noise)} | feature_cond: {np.shape(feature_cond)}')
            fake_data = generator_cond(noise, feature_cond) 
            fake_data_feature, fake_data_label = fake_data[:,:,:num_features], fake_data[:,:,num_features:]
            # print(f'fake_data_feature: {np.shape(fake_data_feature)} | fake_data_label: {np.shape(fake_data_label)}')
            X_train_GAN= torch.cat((X_train_GAN, fake_data_feature.cpu()), dim=0)
            y_train_GAN  = torch.cat((y_train_GAN, fake_data_label.cpu()), dim=0)
    # print(f'[After] X_train_GAN: {np.shape(X_train_GAN)} | y_train_GAN: {np.shape(y_train_GAN)}')
     
    dataloaderGAN = data.DataLoader(data.TensorDataset(X_train_GAN, y_train_GAN), shuffle=True, batch_size=BATCH_SIZE)

    criterionMSE = nn.MSELoss()

    predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=PRED_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    for epoch in range(PRED_EPOCH):
        error_train =0 
        error_test =0
        for batch_idx, real_data in enumerate(dataloaderGAN): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            #----- Train the Predictor --------------------------------------
            predictor.zero_grad()
            optimizer_P.zero_grad()
            label_true = real_data[1][:,-1,:].to(device)
            label_pred = predictor(real_data[0].to(device))
            p_loss = criterionMSE(label_pred, label_true)
            p_loss.backward()     
            optimizer_P.step()       
            error_train = np.sqrt(p_loss.item())

            
        with torch.no_grad():
            label_test_true = y_test[:,-1,:].to(device)
            label_test_pred = predictor(X_test.to(device))
            p_loss_test = criterionMSE(label_test_pred, label_test_true)
            error_test = np.sqrt(p_loss_test.item()) 
            rmse_output.append(error_test)

            r2_score(label_test_pred, label_test_true).cpu()
            R2_output.append(r2_score)
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}')
    return loss_d_output, loss_g_output, rmse_output, R2_output


######################################################################################################################
def trainGAN_WGAN(BATCH_SIZE, SEQ_LENGTH,DAY_STEP, NUM_EPOCH, NUM_FAKE_DATA, TRAIN_RATE, \
                GEN_INPUT_SIZE, GEN_HIDDEN_SIZE, GEN_NUM_LAYERS, GEN_DROP_PROB, GEN_LR, GAN__CLIP, GAN_n_critic,\
                DIS_HIDDEN_SIZE, DIS_NUM_LAYERS, DIS_DROP_PROB, DIS_LR, \
                PRED_lr,PRED_EPOCH, \
                K_STEP, COLS_FEATURE ):
   

    dataset_filename = "data/data_AGAI_DSSAT_scale.csv"
    # cols_feature = ['SRAD','DTTD','N_uptake']
    cols_feature = COLS_FEATURE
    cols_label = ['inorganic_N']
    num_features = len(cols_feature)
    num_labels = len(cols_label)

    myDSSAT = DSSAT.CSVDatasetLoader(dataset_filename, cols_feature,cols_label)
    
    # X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence(seq_len=SEQ_LENGTH, train_rate=TRAIN_RATE, flag_fakedata=False)
    X_train, y_train, X_test, y_test = myDSSAT.get_shortSequence_jump(seq_len=SEQ_LENGTH, day_step=DAY_STEP, train_rate=TRAIN_RATE, flag_fakedata=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

    dataloader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    # Generator & Predictor (or Discriminator)
    generator = G_models.Generator_GRU(input_size=GEN_INPUT_SIZE, hidden_size=GEN_HIDDEN_SIZE, num_layers=GEN_NUM_LAYERS, dropout_prob=GEN_DROP_PROB, bidirectional=True, output_size=num_features+1, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    discriminator = D_models.Discriminator_GRU(input_size=num_features+num_labels, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)

    # sys.exit(0)

    # Define the loss functions and optimizers
    criterionBCE = nn.BCELoss()

    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=GEN_LR, weight_decay=1e-5, betas = (0.5, 0.9))
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=GEN_LR)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=DIS_LR)
    
    
    rmse_output = []
    loss_d_output = []
    loss_g_output = []
    R2_output       = []
    # Training loop
    for epoch in range(NUM_EPOCH * 2):

        error_GAN_d =0
        error_GAN_g =0
        for batch_idx, real_data in enumerate(dataloader): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape

            real_labels     = torch.ones(batch_len, 1, device=device)
            fake_labels     = -torch.zeros(batch_len, 1, device=device)

            ##############################################
            # (1) Update D network: maximize log(D(x)) - log(D(G(z)))
            ##############################################
            # Set discriminator gradients to zero.
            discriminator.zero_grad()
            # optimizer_D.zero_grad()

            noise = torch.randn([batch_len, seq_len, GEN_INPUT_SIZE]).to(device)

            # Train with real
            real_output = discriminator(torch.cat((real_data[0], real_data[1]), dim=2).to(device))
            errD_real = torch.mean(real_output)
            D_x = real_output.mean().item()
            errD_real.backward()

            # Generate fake image batch with G
            fake_data = generator(noise)

            # Train with fake
            fake_output = discriminator(fake_data.detach())
            errD_fake = -torch.mean(fake_output)
            D_G_z1 = fake_output.mean().item()
            errD_fake.backward()

            errD = errD_real + errD_fake # Add gradients from all-real and all-fake batches
            optimizer_d.step() # Update D
            for p in discriminator.parameters(): # Clip weights of discriminator
                p.data.clamp_(-GAN__CLIP, GAN__CLIP)

            # Train the generator every n_critic iterations.
            if (batch_idx + 1) % GAN_n_critic == 0:
                ##############################################
                # (2) Update G network: maximize -log(D(G(z)))
                ##############################################
                # Set generator gradients to zero
                generator.zero_grad()
                # Generate fake image batch with G
                fake_data = generator(noise)
                fake_output = discriminator(fake_data)
                errG = torch.mean(fake_output)
                D_G_z2 = fake_output.mean().item()
                errG.backward()
                optimizer_g.step()

                # progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{batch_idx + 1}/{len(self.dataloader)}] "
                #                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                #                                 f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                error_GAN_d = (errD.item())
                error_GAN_g = (errG.item())

        if False:
            with torch.no_grad():
                label_test_true = y_test[:,-1,:].to(device)
                label_test_pred = predictor(X_test.to(device))
                
                p_loss_test = criterionMSE(label_test_pred, label_test_true)
                error_test = np.sqrt(p_loss_test.item())
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}') 

        loss_d_output.append(error_GAN_d)
        loss_g_output.append(error_GAN_g)

    ######################## Start Prediction Training ############################
    ## Generate Fake data
    noise = torch.randn([NUM_FAKE_DATA, SEQ_LENGTH, GEN_INPUT_SIZE])
    with torch.no_grad():
        fake_data = generator(noise.to(device))
    fake_data_feature, fake_data_label = fake_data[:,:,:num_features], fake_data[:,:,num_features]
    X_train_GAN= torch.cat((X_train, fake_data_feature.cpu()), dim=0)
    y_train_GAN  = torch.cat((y_train, fake_data_label.unsqueeze(-1).cpu()), dim=0)
    dataloaderGAN = data.DataLoader(data.TensorDataset(X_train_GAN, y_train_GAN), shuffle=True, batch_size=BATCH_SIZE)

    criterionMSE = nn.MSELoss()
    predictor = P_models.Prediction_GRU(input_size=num_features, hidden_size=DIS_HIDDEN_SIZE, num_layers=DIS_NUM_LAYERS, dropout_prob=DIS_DROP_PROB, bidirectional=True, output_size=num_labels, batch_size=BATCH_SIZE, seq_len=SEQ_LENGTH).to(device)
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=PRED_lr, weight_decay=1e-5, betas = (0.5, 0.9))

    for epoch in range(PRED_EPOCH):
        error_train =0 
        error_test =0
        for batch_idx, real_data in enumerate(dataloaderGAN): # (real_data, _) to discard the target labels
            batch_len,seq_len,feature_dim = real_data[0].shape
            #----- Train the Predictor --------------------------------------
            predictor.zero_grad()
            optimizer_P.zero_grad()
            label_true = real_data[1][:,-1,:].to(device)
            label_pred = predictor(real_data[0].to(device))
            p_loss = criterionMSE(label_pred, label_true)
            p_loss.backward()     
            optimizer_P.step()       
            error_train = np.sqrt(p_loss.item())

            
        with torch.no_grad():
            label_test_true = y_test[:,-1,:].to(device)
            label_test_pred = predictor(X_test.to(device))
            p_loss_test = criterionMSE(label_test_pred, label_test_true)
            error_test = np.sqrt(p_loss_test.item()) 
            rmse_output.append(error_test)

            r2_test = r2_score(label_test_pred, label_test_true).cpu()
            R2_output.append(r2_test)
        
        if False:
            print(f'[GAN] epoch:{epoch:4d} | P/G train rmse:{error_train_P:.4f}/{error_train_G:.4f} | test rmse:{error_test:.4f}')
    return loss_d_output, loss_g_output, rmse_output, R2_output


######################################################################################################################
