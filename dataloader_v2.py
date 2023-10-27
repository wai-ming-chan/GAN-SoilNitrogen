###############################################################
import pandas as pd
import numpy as np
import torch
import pickle 
import sys
###############################################################

cols_feature = ['SRAD','DTTD','N_uptake']
cols_label = ['inorganic_N']

###############################################################

class CSVDatasetLoader:
    def __init__(self, file_path, cols_feature, cols_label):
        """
        Initialize the CSVDatasetLoader.

        Parameters:
        - file_path (str): The path to the CSV dataset file.
        """
        self.file_path = file_path
        self.data = None
        self.cols_feature = cols_feature
        self.cols_label = cols_label

        self.load_data()

    def load_data(self):
        """
        Load the CSV dataset into a Pandas DataFrame.

        Returns:
        - pd.DataFrame: The loaded dataset.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except FileNotFoundError:
            print(f"CSV file not found at '{self.file_path}'. Please check the file path.")
            return None

    def get_features(self):
        """
        Get the feature columns from the dataset.

        Returns:
        - pd.DataFrame: The feature columns.
        """
        if self.data is not None:
            # You may adjust this based on your dataset's structure
            # For example, if your features are columns 1 to N-1 and the last column is the target,
            # you can use self.data.iloc[:, :-1]
            features = self.data.iloc[:, :-1]  # columns 1 to N-1
            # features = self.data.iloc[:, 1:]  # Assuming the first column is an identifier or index

            # return features
            return self.data[self.cols_feature].values.astype('float32')
        else:
            print("No data loaded. Call 'load_data' first.")
            return None

    def get_target(self):
        """
        Get the target column from the dataset.

        Returns:
        - pd.Series: The target column.
        """
        if self.data is not None:
            # You may adjust this based on your dataset's structure
            # For example, if your target is the last column, you can use self.data.iloc[:, -1]
            target = self.data.iloc[:, -1]  # Assuming the last column is the target
            # target = self.data.iloc[:, 0]  # Assuming the first column is the target
            # return target
        
            return self.data[self.cols_label].values.astype('float32')
        else:
            print("No data loaded. Call 'load_data' first.")
            return None
    
    def get_shortSequence_jump(self, seq_len, day_step, train_rate, flag_fakedata=False):
        df_siteID = self.data[['site_id']].values.astype('int')
        df_DAP = self.data[['plantation_day']].values.astype('int')
        df_features = self.get_features()
        df_targets = self.get_target()

        # day_step = 7
        # day_step = 1
        X_train, y_train, X_test, y_test = [], [], [], []

        for idx in np.unique(df_siteID):
            indices  = [i for i, site_id in enumerate(df_siteID) if site_id == idx]
            # print(f"site-{idx} = len(indices):{len(indices)}\n")
            # print(f"site-{idx} = indices:{indices}\n")
            X = [df_features[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
            y = [df_targets[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
            X, y = np.array(X), np.array(y)
            
            site_features = df_features[indices]
            site_targets = df_targets[indices]
            # min_index = np.min(indices)
            max_index = len(indices) # # np.max(indices) 
            
            subarrays2, subarrays3 = [], []
            for i in range(len(indices) - (seq_len-1) * day_step):
                end_index = (i + (seq_len-1) * day_step)
                subarray_f, subarray_t = site_features[i:end_index+1:day_step], site_targets[i:end_index+1:day_step]
                subarrays2.append(subarray_f)
                subarrays3.append(subarray_t)
                # print(f'[i:{i}] subarrays2: {np.shape(subarrays2)}')

                subarray = site_targets                
            X = np.array(subarrays2)
            y = np.array(subarrays3)
            
            # subarrays2 = [
            #     site_features[i:end_index+1:day_step] for i in range(len(indices) - seq_len + 1)
            #     if (end_index := min(i + (seq_len-1) * day_step, max_index + 1)) < max_index + 1
            # ]

            # subarrays3 = [
            #     site_targets[i:end_index+1:day_step] for i in range(len(indices) - seq_len + 1)
            #     if (end_index := min(i + (seq_len-1) * day_step, max_index + 1)) < max_index + 1
            # ]
             
            shuffled_indices = np.random.permutation(X.shape[0])  # Shuffle the indices of the first axis
            split_point = int(X.shape[0] * train_rate) # Calculate the split point based on the desired percentage (e.g., 75%)
            X_train_site = X[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            X_test_site = X[shuffled_indices[split_point:]]
            y_train_site = y[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            y_test_site = y[shuffled_indices[split_point:]]
            
            X_train = X_train_site if len(X_train) == 0 else np.concatenate((X_train, X_train_site), axis=0)
            X_test = X_test_site if len(X_test) == 0 else np.concatenate((X_test, X_test_site), axis=0)
            y_train = y_train_site if len(y_train) == 0 else np.concatenate((y_train, y_train_site), axis=0)
            y_test = y_test_site if len(y_test) == 0 else np.concatenate((y_test, y_test_site), axis=0)
            
            # print(f'[site:{idx}] X: {np.shape(X)} | X_train: {np.shape(X_train)} | X_test: {np.shape(X_test)} | y_train: {np.shape(y_train)} | y_test: {np.shape(y_test)}')
            # sys.exit("line112") 

        PRINT_SIZE = False
        if PRINT_SIZE:
            print(f'[jump:{day_step}] X_train: {np.shape(X_train)} | X_test: {np.shape(X_test)} | y_train: {np.shape(y_train)} | y_test: {np.shape(y_test)}')
        return X_train, y_train, X_test, y_test
    
    def get_shortSequence(self,seq_len, train_rate, flag_fakedata=False):
        df_siteID = self.data[['site_id']].values.astype('int')
        df_DAP = self.data[['plantation_day']].values.astype('int')
        df_features = self.get_features()
        df_targets = self.get_target()

        X_train, y_train, X_test, y_test = [], [], [], []

        for idx in np.unique(df_siteID):
            indices  = [i for i, site_id in enumerate(df_siteID) if site_id == idx]
            if False:
                print(f'site= {idx} | samples= {len(indices)}')
            
            # X = [df_features[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
            # y = [df_targets[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
            site_features = df_features[indices]
            site_targets = df_targets[indices]
            X = [site_features[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
            y = [site_targets[i:i+seq_len] for i in range(len(indices) - seq_len + 1)]
             
            X, y = np.array(X), np.array(y)

            shuffled_indices = np.random.permutation(X.shape[0])  # Shuffle the indices of the first axis
            split_point = int(X.shape[0] * train_rate) # Calculate the split point based on the desired percentage (e.g., 75%)
            X_train_site = X[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            X_test_site = X[shuffled_indices[split_point:]]
            y_train_site = y[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            y_test_site = y[shuffled_indices[split_point:]]
            
            X_train = X_train_site if len(X_train) == 0 else np.concatenate((X_train, X_train_site), axis=0)
            X_test = X_test_site if len(X_test) == 0 else np.concatenate((X_test, X_test_site), axis=0)
            y_train = y_train_site if len(y_train) == 0 else np.concatenate((y_train, y_train_site), axis=0)
            y_test = y_test_site if len(y_test) == 0 else np.concatenate((y_test, y_test_site), axis=0)
            
        PRINT_SIZE = True
        if PRINT_SIZE:
            print(f'X_train: {np.shape(X_train)} | X_test: {np.shape(X_test)} | y_train: {np.shape(y_train)} | y_test: {np.shape(y_test)}')
        return X_train, y_train, X_test, y_test

# Example usage:
if __name__ == "__main__":
    data_loader = CSVDatasetLoader("data/data_AGAI_DSSAT_scale.csv", cols_feature,cols_label)
    # data = data_loader.load_data()
    # if data is not None:
    #     # Now you can use 'features' and 'target' in your machine learning pipeline.
    #     ss = data_loader.get_shortSequence(seq_len=10, train_rate=0.8, flag_fakedata=False)

        



class GAN_dataloader():
    def __init__(self):
        # self.batch_size = batch_size
        self.token_stream = []
        # self.cols_feature = ['day_cos','day_sin','initial_N_rate','other_N','DTTD','SRAD','TMAX','TMIN','N_uptake']
        self.cols_feature = ['SRAD','DTTD','N_uptake']
        
        with open('pt_DSSAT_transformer.pkl', 'rb') as file:
            self.pt_DSSAT = pickle.load(file)
        with open('pt_KSU_transformer.pkl', 'rb') as file:
            self.pt_KSU = pickle.load(file)

    def get_cols_feature(self):
        return self.cols_feature

    def soilIN_inv_transform(self,labels, flag_tensor=True):
        if flag_tensor == True:
            labels_inv = self.pt_DSSAT.inverse_transform( labels.cpu() )
        else:
            labels_inv = self.pt_DSSAT.inverse_transform( labels )
        return labels_inv
    
    def soilIN_inv_transform_array(self,labels):
        labels_inv = self.pt_DSSAT.inverse_transform( labels )
        return labels_inv
    
    #
    # dataset for GAN training 
    #  +) sliding window sized with lookback
    #  +) all possible sliding windows
    def create_dataset_GAN_slidingWindow_all(self, lookback, train_rate):
    
        df = pd.read_csv('data/data_AGAI_DSSAT_scale_all_filled.csv')
        df_label_all = df[["inorganic_N"]].values.astype('float32')
        df_feature_all = df[self.cols_feature].values.astype('float32')
        df_day_all = df[["plantation_day"]].values.astype('int')
        df_site_all = df[["site_id"]].values.astype('int')
        del df

        X_train, y_train, X_test, y_test = [], [], [], []
        lookback = lookback + 1
        for idx in np.unique(df_site_all):
            indices  = [i for i, site_id in enumerate(df_site_all) if site_id == idx]
            features = df_feature_all[indices]
            labels = df_label_all[indices]
            days = df_day_all[indices]
            
            nz_indices = np.where(labels != 0)[0] # Find all nonzero elements
            nz_days =  np.transpose(days[nz_indices])
            
            array = nz_days[0]
            sub_arrays = [array[i:i+lookback] for i in range(len(array) - lookback + 1)]

            X, y = [], []
            for select_indices in sub_arrays:
                if False:
                    print(f'[site-{idx}] train pattern:{select_indices}')
                if len(X) == 0: # X == []:
                    X.append(features[select_indices, :])
                else:
                    array2 = features[select_indices, :]
                    array2 = array2.reshape(1,array2.shape[0],array2.shape[1])
                    array1 = np.array(X)
                    if array1.shape[1] > array2.shape[1]:  # Check the dimension before padding
                        array2 = np.pad(array2, ((0, 0), (array1.shape[1] - array2.shape[1], 0), (0, 0)), mode='constant')
                    elif  array1.shape[1] < array2.shape[1]:  # Check the dimension before padding:
                        array1 = np.pad(array1, ((0, 0), (array2.shape[1] - array1.shape[1], 0), (0, 0)), mode='constant')
                    # Combine the arrays
                    X = np.concatenate((array1, array2), axis=0)
                if len(y) == 0: #y == []:
                    y.append(labels[select_indices, :])
                else:
                    array2 = labels[select_indices, :]
                    array2 = array2.reshape(1,array2.shape[0],array2.shape[1])
                    array1 = np.array(y)
                    if array1.shape[1] > array2.shape[1]:  # Check the dimension before padding
                        array2 = np.pad(array2, ((0, 0), (array1.shape[1] - array2.shape[1], 0), (0, 0)), mode='constant')
                    elif  array1.shape[1] < array2.shape[1]:  # Check the dimension before padding:
                        array1 = np.pad(array1, ((0, 0), (array2.shape[1] - array1.shape[1], 0), (0, 0)), mode='constant')
                    # Combine the arrays
                    y = np.concatenate((array1, array2), axis=0)
                # y.append(labels[select_indices[-1]]) # target_label
            #----------------------------------------------------
            y=np.array(y).reshape(np.shape(y)[0], np.shape(y)[1], 1)
            shuffled_indices = np.random.permutation(X.shape[0])  # Shuffle the indices of the first axis
            split_point = int(X.shape[0] * train_rate) # Calculate the split point based on the desired percentage (e.g., 75%)
            X_train_site = X[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            X_test_site = X[shuffled_indices[split_point:]]

            X_train = X_train_site if len(X_train) == 0 else np.concatenate((X_train, X_train_site), axis=0)
            X_test = X_test_site if len(X_test) == 0 else np.concatenate((X_test, X_test_site), axis=0)

            y_train_site = y[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            y_test_site = y[shuffled_indices[split_point:]]
            
            y_train = y_train_site if len(y_train) == 0 else np.concatenate((y_train, y_train_site), axis=0)
            y_test = y_test_site if len(y_test) == 0 else np.concatenate((y_test, y_test_site), axis=0)

            
            if False:
                print(f'[dataloader] site-{idx} X_train:{np.shape(X_train)} y_train:{np.shape(y_train)} X_test:{np.shape(X_test)} y_test:{np.shape(y_test)}')
        print(f'GAN data create | X_train:{np.shape(X_train)} y_train:{np.shape(y_train)} X_test:{np.shape(X_test)} y_test:{np.shape(y_test)}')
        return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)
    # END of def create_dataset_GAN_slidingWindow_all
    #
    #
    #---------------------------------------------------------------------------------------------------------

    def create_dataset_slidingWindow_all(self,lookback, train_rate, flag_fakedata=False):
    
        if flag_fakedata:
            df = pd.read_csv('data/fake_data.csv')
        else:
            df = pd.read_csv('data/data_AGAI_DSSAT_scale_all_filled.csv')
        df_label = df[["inorganic_N"]].values.astype('float32')
        df_feature = df[self.cols_feature].values.astype('float32')
        df_day = df[["plantation_day"]].values.astype('int')
        df_site = df[["site_id"]].values.astype('int')
        del df 

        X_train, y_train, X_test, y_test = [], [], [], []
        
        for idx in np.unique(df_site):
            indices  = [i for i, site_id in enumerate(df_site) if site_id == idx]
            features = df_feature[indices]
            labels = df_label[indices]
            days = df_day[indices]
            
            nz_indices = np.where(labels != 0)[0] # Find all nonzero elements
            nz_days =  np.transpose(days[nz_indices])
            
            array = nz_days[0]
            sub_arrays = [array[i:i+lookback] for i in range(len(array) - lookback + 1)]
            X, y = [], []
            for select_indices in sub_arrays:
                # print(f'[site-{idx}] train pattern:{select_indices}')
                if len(X) == 0:
                    X.append(features[select_indices, :])
                else:
                    array2 = features[select_indices, :]
                    array2 = array2.reshape(1,array2.shape[0],array2.shape[1])
                    array1 = np.array(X)
                    if array1.shape[1] > array2.shape[1]:  # Check the dimension before padding
                        array2 = np.pad(array2, ((0, 0), (array1.shape[1] - array2.shape[1], 0), (0, 0)), mode='constant')
                    elif  array1.shape[1] < array2.shape[1]:  # Check the dimension before padding:
                        array1 = np.pad(array1, ((0, 0), (array2.shape[1] - array1.shape[1], 0), (0, 0)), mode='constant')
                    # Combine the arrays
                    X = np.concatenate((array1, array2), axis=0)
                y.append(labels[select_indices[-1]]) # target_label
            #----------------------------------------------------
            y=np.array(y).reshape(np.shape(y)[0], 1, np.shape(y)[1])
            shuffled_indices = np.random.permutation(X.shape[0])  # Shuffle the indices of the first axis
            split_point = int(X.shape[0] * train_rate) # Calculate the split point based on the desired percentage (e.g., 75%)
            X_train_site = X[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            X_test_site = X[shuffled_indices[split_point:]]
            
            X_train = X_train_site if len(X_train) == 0 else np.concatenate((X_train, X_train_site), axis=0)
            X_test = X_test_site if len(X_test) == 0 else np.concatenate((X_test, X_test_site), axis=0)

            y_train_site = y[shuffled_indices[:split_point]] # Split the matrix into two smaller matrices
            y_test_site = y[shuffled_indices[split_point:]]
            
            y_train = y_train_site if len(y_train) == 0 else np.concatenate((y_train, y_train_site), axis=0)
            y_test = y_test_site if len(y_test) == 0 else np.concatenate((y_test, y_test_site), axis=0)

            
            if False:
                print(f'site-{idx} X_train:{np.shape(X_train)} y_train:{np.shape(y_train)} X_test:{np.shape(X_test)} y_test:{np.shape(y_test)}')
        if flag_fakedata == False:
            print(f'Ture Data loaded | X_train:{np.shape(X_train)} y_train:{np.shape(y_train)} X_test:{np.shape(X_test)} y_test:{np.shape(y_test)}')
        else:
            print(f'Fake Data loaded | X_train:{np.shape(X_train)} y_train:{np.shape(y_train)} X_test:{np.shape(X_test)} y_test:{np.shape(y_test)}')
        return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)
    # END of def create_dataset_slidingWindow_all
    #
    #
    # ---------------------------------------------------------------------------------------------------------