import pandas as pd
from sklearn import cluster

import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DataPre:
    def __init__(self):
        pass
    
    
    def transformTimeStamp(self, df):
        df['timestamp'] = df['timestamp'] / 1000
        df['timestamp'] = df['timestamp'].astype(int)
        df.set_index('timestamp', inplace=True)
            
    def loadDataSet(self, path20_int, path40_int, path80_int, path20_dash, path40_dash, path80_dash):
        
        df20_dash = pd.read_csv(path20_dash, sep = ';')
        df20_int = pd.read_csv(path20_int, sep = ',')
        
        df40_dash = pd.read_csv(path40_dash, sep = ';')
        df40_int = pd.read_csv(path40_int, sep = ',')
        
        df80_dash = pd.read_csv(path80_dash, sep = ';')
        df80_int = pd.read_csv(path80_int, sep = ',')
        
        self.transformTimeStamp(df20_dash)
        self.transformTimeStamp(df40_dash)
        self.transformTimeStamp(df80_dash)
        
        df20_int = df20_int.loc[df20_int.groupby('timestamp')['deq_timedelta1'].idxmax()]
        df40_int = df40_int.loc[df40_int.groupby('timestamp')['deq_timedelta1'].idxmax()]
        df80_int = df80_int.loc[df80_int.groupby('timestamp')['deq_timedelta1'].idxmax()]
        
        #df32_int = df32_int.drop_duplicates(subset=['timestamp'])
        #df64_int = df64_int.drop_duplicates(subset=['timestamp'])
        
        df20_int.set_index('timestamp', inplace=True)
        df40_int.set_index('timestamp', inplace=True)
        df80_int.set_index('timestamp', inplace=True)
        
        
        merged20_df = pd.merge(df20_int, df20_dash, left_index=True, right_index=True).reset_index()
        merged40_df = pd.merge(df40_int, df40_dash, left_index=True, right_index=True).reset_index()
        merged80_df = pd.merge(df80_int, df80_dash, left_index=True, right_index=True).reset_index()
        
        merged80_df['q_size'] = 2
        merged40_df['q_size'] = 1
        merged20_df['q_size'] = 0
        
        self.dataset = pd.concat([merged20_df, merged40_df, merged80_df], ignore_index=True)

    def hotEncode(self):
        #creating instance of one-hot-encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')

        #perform one-hot encoding on 'team' column 
        encoder_df = pd.DataFrame(self.encoder.fit_transform(self.dataset[['Resolution']]).toarray())

        #merge one-hot encoded columns back with original DataFrame
        self.dataset = self.dataset.join(encoder_df).copy()
        self.dataset.drop('Resolution', axis=1, inplace=True)
    
    def inverseHotEncode(self, matrix):
        return self.encoder.inverse_transform(matrix)
    
    

    def showColumns(self):
         for i in self.dataset.columns:
            print(i)
        
    def preProcessData(self, sorted_cols, cat_cols, cond_col, random): 
        
        #self.dataset.drop('LOG', axis=1, inplace=True)
        #self.dataset.drop('Resolution', axis=1, inplace=True) 
        
        for i in sorted_cols:
            self.dataset[i].fillna(self.dataset[i].mean(), inplace=True)
        
        if len(cat_cols) > 0:
            self.hotEncode()
            cat_cols = [0,1,2]
        
        self.processed_data = self.dataset[ sorted_cols + cat_cols ].copy()
        self.num_cols = list(self.processed_data.columns[self.processed_data.columns != cond_col])
        self.cat_cols = cat_cols
        self.num_cols = sorted_cols

        if random == True:
            idx = np.random.permutation(self.processed_data.index)
            self.processed_data = self.processed_data.reindex(idx)
        
        
    def removeOutliers(self):
        #for i in self.processed_data.columns:
        for i in self.num_cols:
            q1 = self.processed_data[i].quantile(.25)
            q3 = self.processed_data[i].quantile(.75)
            iqr = q3 - q1
            lim_inf = q1 - (1.5*iqr) 
            lim_sup = q3 + (1.5*iqr)
            for (j, k) in zip(self.processed_data[i].values, self.processed_data.index):
                if ( (j < lim_inf) or (j > lim_sup) ):
                    self.processed_data = self.processed_data.drop(k) 
                    
    def removeAtributoscomMesmoValor(self):
        self.processed_data = self.processed_data.drop(columns=self.processed_data.columns[self.processed_data.nunique()==1], inplace=False)

    def clusterData(self, number_clusters):
        #For the purpose of this example we will only synthesize the minority class
        #Create a new class column using KMeans - This will mainly be useful if we want to leverage conditional GAN
        print("Dataset info: Number of records - {} Number of variables - {}".format(self.processed_data.shape[0], self.processed_data.shape[1]))
        algorithm = cluster.KMeans
        args, kwds = (), {'n_clusters':number_clusters, 'random_state':0}
        labels = algorithm(*args, **kwds).fit_predict(self.processed_data[ self.num_cols ])
        
        self.cluster = self.processed_data.copy()
        self.cluster['Class'] = labels