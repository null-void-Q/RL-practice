import pandas as pd
import numpy as np
from services import make_scaler

   
def loadAndPreprocess(file_path, means_path='./f_mean.npy'):
    print('Loading Data...')

    data = pd.read_csv(file_path,dtype='float32')

    # data reduction
   
    data = data.query('weight > 0').reset_index(drop=True)
    data = data.query('date > 85').reset_index(drop = True)
    

    #feature preprocessing
    features = data.columns[data.columns.str.contains('feature')]
    means = pd.Series(np.load(means_path),index=features[1:],dtype='float32')
    data = data.fillna(means)

    weights = data['weight'].to_numpy()
    features = data[features].to_numpy()
    response = data['resp'].to_numpy()
    dates = data['date'].astype(int).to_numpy()

    print('Scalling Data...')
    features = make_scaler(features)

    print('Data Ready.')
    return features, response, dates, weights


def calculate_mean(file_path):
    data = pd.read_csv(file_path,dtype='float32')
    features = data.iloc[:,data.columns.str.contains('feature')]
    f_mean = features.iloc[:,1:].mean()
    f_mean = f_mean.values
    np.save('f_mean.npy', f_mean)
   