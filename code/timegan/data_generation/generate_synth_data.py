from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading
import numpy as np
import tensorflow as tf
import sys
import pickle
sys.path.insert(0, '../../data_process/')
from my_data_pre_intdash import DataPre

dataset_directory = '../../../datasets/'

def loadSynthData(model20, model40, model80, seq_len):
    synth_20 = TimeGAN.load(model20)
    synth_data_20 = synth_20.sample(seq_len)
    
    synth_40 = TimeGAN.load(model40)
    synth_data_40 = synth_40.sample(seq_len)
    
    synth_80 = TimeGAN.load(model80)
    synth_data_80 = synth_80.sample(seq_len)
    
    synth_data_20[:,:,14:18][synth_data_20[:,:,14:17] >= 0.5] = 1
    synth_data_20[:,:,14:18][synth_data_20[:,:,14:17] < 0.5] = 0
    
    synth_data_40[:,:,14:18][synth_data_40[:,:,14:17] >= 0.5] = 1
    synth_data_40[:,:,14:18][synth_data_40[:,:,14:17] < 0.5] = 0
        
    synth_data_80[:,:,14:18][synth_data_80[:,:,14:17] >= 0.5] = 1
    synth_data_80[:,:,14:18][synth_data_80[:,:,14:17] < 0.5] = 0
    
    return synth_data_20, synth_data_40, synth_data_80

def loadRealData(dsint20, dsint40, dsint80, dsdash20, dsdash40, dsdash80, num_cols, cat_cols, sample_size, randon, outliers):
    dp = DataPre()
    
    dp.loadDataSet(path20_int=dsint20, path40_int=dsint40, path80_int=dsint80, path20_dash=dsdash20, path40_dash=dsdash40, path80_dash=dsdash80)
    
    dp.preProcessData(num_cols, cat_cols=cat_cols, cond_col='q_size', random=randon)
    if outliers == False:
        dp.removeOutliers()
    
    real_data_20 = dp.processed_data.loc[dp.processed_data['q_size'] == 0].copy()
    real_data_20 = real_data_20[0:sample_size].copy()
    real_data_20 = real_data_20.values
    
    real_data_40 = dp.processed_data.loc[dp.processed_data['q_size'] == 1].copy()
    real_data_40 = real_data_40[0:sample_size].copy()
    real_data_40 = real_data_40.values
    
    real_data_80 = dp.processed_data.loc[dp.processed_data['q_size'] == 2].copy()
    real_data_80 = real_data_80[0:sample_size].copy()
    real_data_80 = real_data_80.values
    
    return real_data_20, real_data_40, real_data_80

def getStatistics(data):
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    
    return [percentile_25, median, percentile_75]

def genStatisctics(real_20, synth_20, real_40, synth_40, real_80, synth_80, sample_size, num_cols):    
    dict = {}

    for j, col in enumerate(num_cols): 
        dict[col] = [getStatistics(real_20[:,j][:sample_size]), 
                     getStatistics(synth_20[:,j][:sample_size]), 
                     getStatistics(real_40[:,j][:sample_size]), 
                     getStatistics(synth_40[:,j][:sample_size]),
                     getStatistics(real_80[:,j][:sample_size]), 
                     getStatistics(synth_80[:,j][:sample_size])]
    
    return dict

def createDataSet(lines, seq_len, data):
    dataset = np.zeros(lines*seq_len*17).reshape(lines*seq_len,17)

    for i in range(0,lines):
        for j in range(0, seq_len):
            dataset[(i*seq_len) + j] = data[i][j][:]
            
    return dataset

def roundFeatures(dataset):
    for i in range(0, len(dataset)):
        dataset[i] = np.round(dataset[i])

def getMetrics(data): 
    # [ (Upper_Quartile(Real) - Lower_Quartile(Real)) - (Upper_Quartile(Synth) - Lower_Quartile(Synth) ] + (Median(Real) - Median(Synth))
    
    metric20 = (abs( (data[0][2] - data[0][0]) - (data[1][2] - data[1][0])) +
               abs(  data[0][1] - data[1][1] ) )
    
    metric40 = (abs( (data[2][2] - data[2][0]) - (data[3][2] - data[3][0])) +
               abs(  data[2][1] - data[3][1] ) )

    metric80 = (abs( (data[4][2] - data[4][0]) - (data[5][2] - data[5][0])) +
               abs(  data[4][1] - data[5][1] ) )

    return metric20, metric40, metric80

def get_allfeatures_metrics(num_cols, arr, m, metrics):
    for j, col in enumerate(num_cols): 
        arr[0][m][j], arr[1][m][j], arr[2][m][j] = getMetrics(metrics.get(col))
        

####################### GENERATION #######################

sample_size = 2400

num_cols = ['enq_qdepth1','deq_timedelta1', 'deq_qdepth1',
            ' enq_qdepth2', ' deq_timedelta2', ' deq_qdepth2',
            'enq_qdepth3', 'deq_timedelta3', 'deq_qdepth3',
            'Buffer', 'ReportedBitrate', 'FPS', 'CalcBitrate',
            'q_size'] 
cat_cols = ['Resolution']

real_20, real_40, real_80 = loadRealData(dsint20= dataset_directory + 'log_INT_20ms.csv',
                                        dsint40= dataset_directory + 'log_INT_40ms.csv',
                                        dsint80= dataset_directory + 'log_INT_80ms.csv',
                                        dsdash20= dataset_directory + 'dash_20ms.csv',
                                        dsdash40= dataset_directory + 'dash_40ms.csv',
                                        dsdash80= dataset_directory + 'dash_80ms.csv',
                                        num_cols=num_cols,
                                        cat_cols=cat_cols,
                                        sample_size=4000, 
                                        randon=False, 
                                        outliers=False) 

scaler20 = MinMaxScaler().fit(real_20)
scaler40 = MinMaxScaler().fit(real_40)
scaler80 = MinMaxScaler().fit(real_80)

models = {}

size = 3600
data = np.zeros(3*9*len(num_cols)).reshape(3,9,len(num_cols))

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:0'):
    for i in range(0,1):
        for j in range(0,3):
            for k in range(0,3):
                seq_len=(50*(i)+50) 
                synth_20_norm, synth_40_norm, synth_80_norm = loadSynthData(model20= str('../saved_models/so20_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                model40= str('../saved_models/so40_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'),
                                                model80= str('../saved_models/so80_seqlen_'+ str((50*(i) + 50)) + '_hidim_' + str(20*(j)+20) + '_batch_' +  str(28*(k) + 100) + '.pkl'), 
                                                seq_len=int(size/seq_len))                                              
                
                real_20_norm =  real_data_loading(real_20, seq_len=seq_len)
                real_40_norm =  real_data_loading(real_40, seq_len=seq_len)
                real_80_norm =  real_data_loading(real_80, seq_len=seq_len)
                
                
                real_20_norm = createDataSet(int(size/seq_len),seq_len, real_20_norm)
                real_40_norm = createDataSet(int(size/seq_len),seq_len, real_40_norm)
                real_80_norm = createDataSet(int(size/seq_len),seq_len, real_80_norm)
                
                synth_20_norm = createDataSet(int(size/seq_len),seq_len, synth_20_norm)
                synth_40_norm = createDataSet(int(size/seq_len),seq_len, synth_40_norm)
                synth_80_norm = createDataSet(int(size/seq_len),seq_len, synth_80_norm)
                
                data_synth_20 = scaler20.inverse_transform(synth_20_norm)
                data_synth_40 = scaler40.inverse_transform(synth_40_norm)
                data_synth_80 = scaler80.inverse_transform(synth_80_norm)
                
                roundFeatures(data_synth_20)
                roundFeatures(data_synth_40)
                roundFeatures(data_synth_80)
                    
                models['so_seqlen_'+str((50*(i) + 50))+'_hidim_'+str(20*(j)+20)+'_batch_'+str(28*(k)+100)+'.pkl'] = [data_synth_20, 
                                                                                                                    data_synth_40,
                                                                                                                    data_synth_80]            
                metrics = genStatisctics(real_20_norm, synth_20_norm, real_40_norm, synth_40_norm, real_80_norm, synth_80_norm, sample_size, num_cols)

                print(metrics)
                get_allfeatures_metrics(num_cols, data, (i*9) + (j*3) + (k) , metrics)

    directory_path = '../saved_objects/'
    
    with open(directory_path + 'real9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump([real_20, real_40, real_80], file)
    
    with open(directory_path + 'models9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump(models, file) 
        
    with open(directory_path + 'metrics9_50_3600_norm_27.pkl', 'wb') as file:
        pickle.dump(data, file)

except RuntimeError as e:
  print(e)