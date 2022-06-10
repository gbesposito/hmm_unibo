import numpy as np
    

def rand_mat(nrow,ncol):
    while True:
        M = np.zeros((nrow,ncol))
        for i in range(nrow):
            for j in range(ncol-1): M[i,j] = (1/ncol)+np.random.normal(0,0.2)
            M[i,ncol-1] = 1-sum(M[i,:])
        if (M>0).all(): break
    return M


def delta(M1, M2):
    if M1.shape != M2.shape: raise ValueError("shape mismatch")
    if len(M1.shape) == 1:
        M3 = np.zeros(M1.shape[0])
        for i in range(M3.shape[0]): M3[i] = (M1[i] - M2[i])**2
    else:              
        M3 = np.zeros(M1.shape)
        for i in range(M1.shape[0]): M3[i] = np.array([(M1[i,j] - M2[i,j])**2 for j in range(M1.shape[1])])
    return M3.sum()

def WalkForward(df_train_start, 
                inputVariables, 
                step=1, 
                start_interval = 32):
    """
    Description:
    Create forward walk backtest dataframe list

    Input:

    df -- dataframe with input data
    model_parameter -- list of model input variables
    incremental -- create dataframe adding incrementally block
    step -- number of time step in the past used to train the model

    Output:
    list of dictionary with all dataframes for training and testing

    """
    temporal_path = df_train_start.index
    range_date=(range(start_interval, len(temporal_path)))
    experiment = []
    
    for t in range_date:
        
        
        df_train = df_train_start[(df_train_start.index >= pd.to_datetime(temporal_path[t-start_interval]))&(df_train_start.index < pd.to_datetime(temporal_path[t-step]))]
        
        #df_test = df_train_start.loc[(df_train_start.index == pd.to_datetime(temporal_path[t]))]
        


        # from dataframe to input matrix & target vector
        np_train_input = df_train[inputVariables].values
        #np_test_input = df_test[inputVariables].values
        
        
        
        experiment.append({'date_key': df_train.index.max(),
                    'universe_train': df_train,
                    "train_input": np_train_input})
    return experiment