import pandas as pd
import numpy as np
global param
param = dict()
def function_0():
    ####################################################################
    ##                                              Import Input Data ##
    ####################################################################

    df_pipe = pd.read_excel('example_3.xlsx', sheet_name = 'Linepipe')

    df_pipe = df_pipe.transpose()
    df_pipe.columns = df_pipe.loc['Property']
    df_pipe = df_pipe.iloc[-1].squeeze()  

    param['sensitivity'] = 0
    param['od']          = df_pipe['od']
    param['wt']          = df_pipe['wt']
    param['young']       = df_pipe['young']
    param['poisson']     = df_pipe['poisson']
    param['alpha']       = df_pipe['alpha']

    yield '*PARAMETER\n'
    for key in param.keys():
        yield f'{key} = {str(param[key])}\n'
