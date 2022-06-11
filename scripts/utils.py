import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from termcolor import colored
import empyrical as emp
#

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
    
    temporal_path = df_train_start.index
    range_date=(range(start_interval, len(temporal_path)))
    experiment = []
    
    for t in range_date:
        df_train = df_train_start[(df_train_start.index >= pd.to_datetime(temporal_path[t-start_interval]))&
                                  (df_train_start.index < pd.to_datetime(temporal_path[t-step]))]
        
        # from dataframe to input matrix & target vector
        np_train_input = df_train[inputVariables].values
        #np_test_input = df_test[inputVariables].values
        
        
        
        experiment.append({'date_key': df_train.index.max(),
                    'universe_train': df_train,
                    "train_input": np_train_input})
    return experiment

def performance_extd(returns,
                annualization=252,
                color='red', 
                plot=True,
                style_line='-',
                label='LONG',
                verbose=True,
                round_=True):
    """
    Calculates basic performance measures on series of returns(Return, Volatility, Sharpe)
    
   
    Args:
        returns: Daily(period) noncumulative returns of the strategy
        color: the color for the print
        plot: True if you want plot cagr
        verbose: if you want to print the output
        round_: if you want rount the number
        annualization : int, factor used to annualize values (252 if data is daily,
                                                                            52 if data is weekly,
                                                                            12 if data is monthly,
                                                                            4 of data is quarterly)
    Returns: 
        pandas DataFrame
    """

    Return = emp.annual_return(returns, annualization=annualization) * 100
    Volatility = emp.annual_volatility(returns,
                                       annualization=annualization) * 100

    Sharpe2 = emp.sharpe_ratio(returns, annualization=annualization)
    Sortino = emp.sortino_ratio(returns, annualization=annualization)
    Calmar = emp.calmar_ratio(returns, annualization=annualization)
    

    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    Max_DD = dd.min() * 100

    if round_:
        Return = np.round(Return, 2)
        Volatility = np.round(Volatility, 2)
        Sharpe = np.round(Sharpe2, 2)
        Max_DD = np.round(Max_DD, 2)

        end = dd.idxmin()
        start = r.loc[:end].idxmax()

    stat = {
        'Return': [Return],
        'Volatility': [Volatility],
        'Sharpe': [Sharpe],
        'Max Draw-Down': [Max_DD]
        
    }

    if verbose:
        print(colored('Return:', color), Return)
        print(colored('Volatility:', color), Volatility)
        print(colored('Sharpe Ratio:', color), round(Sharpe2, 2))
        print(colored('Sortino Ratio:', color), round(Sortino, 2))
        print(colored('Calmar Ratio:', color), round(Calmar, 2))
        print(colored('Max Draw-Down:', color), Max_DD,
              colored('Start:', color), start.strftime("%m/%d/%Y"),
              colored('End:', color), end.strftime("%m/%d/%Y"))
    if plot:
        start = returns.index.min()
        ptf_cum = pd.Series()
        ptf_cum[start] = 1
        ptf_cum = ptf_cum.append(np.cumprod(returns[returns.index > start] +
                                            1))
        ax=ptf_cum.plot(figsize=(20, 10), color=color, label=label, style=style_line)
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    return pd.DataFrame(stat)