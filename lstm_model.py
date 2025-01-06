# -*- coding: utf-8 -*-
"""LSTM_ADAPMETEO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zyAWpS-YJslAyTzLxl8MKmwM5YVQq6UV

<a href="https://colab.research.google.com/github/AdaptiveMeteoSrl/meteoGNN/blob/main/LSTM_FINAL.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# -*- coding: utf-8 -*-
"""LSTM_METEO.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1zyAWpS-YJslAyTzLxl8MKmwM5YVQq6UV
<a href="https://justpaste.it/redirect/d2hl7/https%3A%2F%2Fcolab.research.google.com%2Fgithub%2FAdaptiveMeteoSrl%2FmeteoGNN%2Fblob%2Fmain%2FLSTM_FINAL.py" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""
__author__ = 'Enrico Bignozzi','Emanuele Antonelli', 'Raffaello Mastromarino', 'Niko Brimi'
__credits__ = ["Enrico Bignozzi", "Emanuele Antonelli", "Raffaello Mastromarino", "Niko Brimi", "Paolo Scaccia", "Paolo Antonelli"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Paolo Scaccia <paolo.scaccia@adaptivemeteo.com>", "Emanuele Antonelli <emaantonelli20@gmail.com>"
__email__      = "paolo.scaccia@adaptivemeteo.com", "emaantonelli20@gmail.com"
import numpy as np
from math import sqrt
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import time
import pickle as pc
import typing
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from functools import partial
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import math
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
from datetime import datetime, timedelta
from torch.utils.data import Sampler
import sys
import csv
#Classe che serve a selezionare indici da cui prendere sequenze tali per cui non ci siano misure mancanti
class SpecificIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)
#EarlyStopping è una procedura basata sull'andamento della Loss function
#Se la loss function di validazione, per un tot di epoche consecutive, non migliora, allora il training termina
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
#Metriche varie utili nella fase di training, validation e testing
def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

csv_file = sys.argv[1]
len_to_predict = int(sys.argv[2])
feature_to_predict = sys.argv[3]
station_to_predict = sys.argv[4]
last_column = feature_to_predict + "_STA{}".format(station_to_predict)
print("You are predicting{}".format(last_column))
df = pd.read_csv(csv_file).dropna()#carico i dati in un dataframe
df['DATE'] = pd.to_datetime(df['DATE'])
df['hour']=df['DATE'].dt.hour
df['month']=df.DATE.dt.month
col0= ['DATE', 'TEMP_STA0', 'TEMP_STA1', 'TEMP_STA2', 'TEMP_STA3', 'TEMP_STA4', 'HUM_STA0',
       'HUM_STA1', 'HUM_STA2', 'HUM_STA3', "HUM_STA4", 'PRESS_STA0',
       'PRESS_STA1', 'PRESS_STA2', 'PRESS_STA3', 'PRESS_STA4', 'PRO_X_STA0',
       'PRO_X_STA1', 'PRO_X_STA2', 'PRO_X_STA3', 'PRO_X_STA4', 'PRO_Y_STA0',
       'PRO_Y_STA1', 'PRO_Y_STA2', 'PRO_Y_STA3', 'PRO_Y_STA4', 'hour', 'month']#seleziono le colonne relative alle features da utilizzare

df=df[col0]
df["PRED_COL"] = df[last_column]
df.drop(columns = last_column, inplace = True)
df.rename(columns={"PRED_COL": last_column}, inplace=True)
co = df.columns[1:]
df = df.reset_index(drop=True)
#Qui si esegue il controllo sui dati mancanti
#Si chiede che di una sequenza di dati presa dal dataframe, questa non presenti dati mancanti
desired_interval = 1008                 #Deve essere più maggiore o uguale alla lunghezza della sequenza in input più il tempo nel futuro da prevedere (cioè desired_interval >= seq_len + pred_len)
df["DATE"]=pd.to_datetime(df["DATE"])
desired_duration = timedelta(days=7)
good_starts = []
for i in range(df.shape[0]-desired_interval):                                      #Crea vettore di indici buoni per selezionare sequenze senza salti
  if df["DATE"][i+desired_interval]-df["DATE"][i] == desired_duration:
    good_starts.append(df.index[i])
fea = df.shape[1]-1
good_starts_train = good_starts[0:int(len(good_starts)*0.7)]            #Divide il vettore good_starts in indici iniziali di train, validation e di test
good_starts_vali = good_starts[int(len(good_starts)*0.7):int(len(good_starts)*0.9)]
good_starts_vali = [l for l in good_starts_vali if l>(good_starts_train[-1]+desired_interval+1)]
good_starts_test = good_starts[int(len(good_starts)*0.9):int(1*len(good_starts))]
good_starts_test = [l for l in good_starts_test if l>(good_starts_vali[-1]+desired_interval+1)]  #Train, validation e test non possono sovrapporsi, dunque c'è un embargo fra i tre set
shuffled_list = good_starts_train.copy()                                                        #Fa lo shuffle del train
random.shuffle(shuffled_list)
tr = shuffled_list
va = good_starts_vali
te =  good_starts_test
gs=[tr,va,te]                                            #gs è un array che contiene a sua volta tre array di indici, uno con indici da cui leggere sequenze per train, e gli altri due per vali e test
df=df[:-1]
#'forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#Per l'obbiettivo preposto si utilizzano più features per prevederne una sola, per cui l'opzione usata è sempre e solo MS.
class Dataset():
    def __init__(self, root_path=None, flag='train', size=None,
                 features='MS',target=last_column, scale=True, timeenc=0, freq='10m'):
        # size [seq_len, label_len, pred_len]
        # info sugli input
        # size:lista contenente [seq_len, label_len, pred_len]
        # features: stringa che specifica il tipo di task di previsione
        # target: stringa che specifica la variabile target
        # scale: se True si effettua lo scaling dei dati con MinMax Scaler, altrimenti False
        # timeenc: intero indicante il tipo di encoding temporale
        # freq: stringa che specifica la frequenza
        if size == None:
            print("size è none. Forse ci sono problemi")
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'val','test']
        type_map = {'train': 0, 'val': 1, 'test':2} #Flag che specifica il tipo di splitting diverso per ogni set
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.__read_data__()
    def __read_data__(self): #Questo metodo effettua una lettura e un preprocessing dei dati (ossia uno scaling)
        self.scaler = MinMaxScaler() #Scaler che normalizza i dati tra 0 e 1
        df_raw = df
        #border1s e border2s sono liste usate per definire gli estremi degli intervalli sulla base del tipo di splitting
        #in questo caso non è necessario definire diversi borders per train, validation e test in quanto gli indici da cui
        #prendere le sequenze sono già specificati nel Sampler (classe all'inizio del codice)
        #Nel caso si volesse non utilizzare il sampler sarebbe tuttavia necessario specificare i borders sulla base della divisione in train, validation e test
        border1s = [0,
                    0,
                    0]
        border2s = [int(len(df_raw)*1),
                    int(len(df_raw)*1),
                    int(len(df_raw)*1)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            stop_index = good_starts_train[-1]+desired_interval
            train_data = df_data.iloc[:stop_index + 1]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.scaler.fit(train_data[self.target].values.reshape(-1, 1)) #Cosi' poi da andare a denormalizzare le previsioni per valutare le prestazioni sul test
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index): #Metodo usato per recuperare una specifica sequenza di dati dal dataset dato un certo indice (parte, insieme al Sampler, del DataLoader)
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end+self.pred_len]
        return seq_x,seq_y
    def __len__(self): #Ritorna il numero totale di sequenze nel dataset
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    def inverse_transform(self, data): # Effettua lo scaling inverso dei dati nel caso si sia scelto di riscalarli nel preprocessing (si fa per il test)
        return self.scaler.inverse_transform(data)
class Model(nn.Module):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""
    def __init__(
        self,
        seq_len,
        pred_len,
        n_features,
        batch_size,
        hidden,
        nlayers,
        dropout
    ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len=pred_len
        self.hidden=hidden
        self.n_features=n_features
        self.batch_size=batch_size
        self.nlayers=nlayers
        self.dropout = dropout                           #Modello con layer LSTM e batch normalization alternati
        self.bn1 = nn.BatchNorm1d(self.n_features)
        self.lstm1 = nn.LSTM(self.n_features, self.hidden, num_layers=self.nlayers, batch_first=True, dropout=self.dropout)
        self.bn2 = nn.BatchNorm1d(self.hidden)
        self.lstm2 = nn.LSTM(self.hidden, self.hidden * 2, num_layers=self.nlayers, batch_first=True, dropout=self.dropout)
        self.bn3 = nn.BatchNorm1d(self.hidden * 2)
        self.dense1 = nn.Linear(self.hidden * 2, self.hidden * 1)
        self.bn5 = nn.BatchNorm1d(self.hidden * 1)
        self.dense2 = nn.Linear(self.hidden * 1, 1)

    def forward(self, inputs):
        """input: BxSxF
        con B=Batch_size
            S=Lunghezza sequenza
            F=Numero di features
        """
        inputs=self.bn1(inputs.permute(0,2,1)).permute(0,2,1)                #I permute servono per fare le batch normalization appropriatamente
        lstm1_out,_ = self.lstm1(inputs)
        lstm1_out=self.bn2(lstm1_out.permute(0,2,1)).permute(0,2,1)
        lstm2_out,_ = self.lstm2(lstm1_out)
        lstm2_out=self.bn3(lstm2_out.permute(0,2,1)).permute(0,2,1)
        dense1_out = self.dense1(lstm2_out)
        dense1_out=self.bn5(dense1_out.permute(0,2,1)).permute(0,2,1)
        dense2_out = self.dense2(dense1_out)
        dense2_out = dense2_out[:, -self.pred_len:,:]        #Alla fine si seleziona soltanto l'ultimo elemento in uscita dal layer LSTM
        return dense2_out

class DNNModel(object):
    def __init__(self,batch_size,seq_len,lstm_units,lr,weight_decay, nlayers,dropout,epochs_early_stopping=20, pred_len = len_to_predict):
        self.embed='fixed'
        self.batch_size=batch_size
        self.freq='10m'
        self.pred_len= pred_len   #Questo è un input da riga di comando
        self.seq_len=seq_len
        self.n_features=27       #Features in ingresso
        self.lstm_units=lstm_units       #Dimensione hidden di output per LSTM
        self.train_epochs=500                #Numero epoche
        self.patience_early_stopping = 10    #Pazienza EarlyStopping
        self.patience_scheduler = 2          #Pazienza scheduler (Spiegato dopo)
        self.features='MS'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints='./checkpoints/'  #Serve a salvare i pesi relativi alla migliore prestazione del modello
        self.lr=lr                         #Learning rate
        self.nlayers=nlayers                #Numero di stacks di lstm (ogni layer lstm può processare le sequenze su più livelli, andando a carpire correlazioni più sottili)
        self.weight_decay = weight_decay    #Serve a tenere sotto controllo overfitting riducendo i pesi man mano che il training avanza
        self.dropout=dropout                #Serve a prevenire overfitting annullando l'effetto di alcuni pesi a caso in training
        self.model = self._build_model()
    def data_provider(self,flag):
        data_dict = {'ETTh1': Dataset}    #Dizionario che serve a prendere il dataset
        Data = data_dict['ETTh1']
        if flag == 'val':            #Condizioni poste sulla base della fase dell'algoritmo (per train, validation, test)
            shuffle_flag = False
            drop_last =  True
            batch_size =  1
            freq = self.freq
            v_s = 1                     #Indice vs serve a selezionare una serie di indici da usare dal vettore gs per il Sampler
        elif flag == 'test':
            shuffle_flag = False
            drop_last =  True
            batch_size =  1
            freq = self.freq
            v_s = 2
        else:
            shuffle_flag= False
            drop_last =  True
            batch_size =self.batch_size
            freq = self.freq
            v_s = 0
        data_set = Data(
            flag=flag,
            size=[self.seq_len, self.pred_len],         #Definizione del dataset e del DataLoader
            freq=self.freq
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last,
            sampler=SpecificIndicesSampler(gs[v_s])     #Sampler accede al vettore di indici buoni che ho definito (o train o validation) e sceglie indici da li cosi da non avere salti
            )                                           #in una singola sequenza. Non si può mettere Sampler e shuffle insieme, quindi lo shuffle è effettuato prima
        return data_set, data_loader
    #Crea il modello da addestrare
    def _build_model(self):
        model = Model(seq_len=self.seq_len,pred_len=self.pred_len,n_features=self.n_features,batch_size=self.batch_size, hidden=self.lstm_units,nlayers=self.nlayers,dropout=self.dropout).to(self.device)
        return model
    #Seleziona ottimizzatore
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model_optim
    #Seleziona la funzione di loss
    def _select_criterion(self):
        criterion = nn.MSELoss() # Altrimenti nn.HuberLoss(reduction='mean', delta=1.0), altrimenti nn.MSELoss()
        return criterion
    def _get_data(self, flag):
        data_set, data_loader = self.data_provider(flag)
        return data_set, data_loader

    #Fase di validation
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
          for i, (batch_x,batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            outputs = self.model(batch_x)
            f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0
            outputs = outputs[:, -1:, f_dim:]     #Prende ultimo elemento predetto (in realtà questo è gia fatto nella rete) dall'output della rete
            batch_y = batch_y[:, -1:, f_dim:].to(self.device)  #Prende ultimo valore della sequenza, vale a dire quello che si vuole prevedere, dalla sequenza presa dal dataset di validation
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    #Fase di train
    def train(self,setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.patience_early_stopping, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  #Utilizzo delle funzioni precedentemente definite per dare al modello impostazioni di training
        #Scheduler decide di abbassare il learning rate (cioè di permettere meno mobilità ai pesi) quando c'è uno stallo nella validation loss
        scheduler = ReduceLROnPlateau(model_optim, mode='min', patience=self.patience_scheduler, factor=0.5, verbose=False)
        #Training effettivo
        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()                                   #Zero grad, loss backward, optimizer step sono gli elementi essenziali nella fase di training in torch
                batch_x = batch_x.float().to(self.device)                 #In validation e in test NON VENGONO MESSI, perche i pesi sono fissi
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0    #Tutto simile a ciò che è stato definito in fase di validation
                outputs = outputs[:, -1:, f_dim:]
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, criterion)                    #Valutazione di validation e di test
            scheduler.step(vali_loss)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    #Fase di testing indipendente, ossia post-training
    #La valutazione di test è stata già fatta per ogni epoca, tuttavia dopo la fine del training, per visualizzare i risultati, si prende il miglior modello salvato
    #ossia quello che ha ottenuto la minore loss di validation (non si può decidere sulla base del test in quanto il programma non deve vedere il test in alcun modo)
    def vali_test(self, setting, test=True):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))  #Carica sul modello i pesi con migliore validation loss
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)
              outputs = self.model(batch_x)
              f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0
              outputs = outputs[:, -1:, f_dim:]
              batch_y = batch_y[:, -1:, f_dim:].to(self.device)
              outputs = outputs.detach().cpu().numpy()
              batch_y = batch_y.detach().cpu().numpy()
              pred = outputs
              true = batch_y
              preds.append(pred)
              trues.append(true)

        preds = np.array(preds)                                                #Genera i risultati da poter plottare e visualizzare
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return preds,trues
def _build_space(): #La funzione genera lo spazio di ricerca per gli iperaparametri da usare nel fine-tuning
    space = {
        'batch_size': hp.quniform('batch_size', 16, 526, 16),
        'seq_len': hp.quniform('seq_len', 10, 100, 1),
        'lstm_units': hp.quniform('lstm_units', 16, 256, 16),
        'lr': hp.uniform('lr', 0.000000001, 0.01),
        'nlayers': hp.quniform('nlayers', 1, 2, 1),
        'dropout': hp.uniform('dropout', 0, 0.3),
        "weight_decay": hp.uniform('weight_decay', 0.000000001, 0.0001),
        }
    return space

def _hyperopt_objective(hyperparameters, trials, trials_file_path, max_evals):#definisce l'obiettivo della minimizzazione per il processo di ottimizzazione degli iperparametri
    #info input:
    #hyperparameters: dizionario contenente gli iperparametri da valutare
    #trials:oggetto che conserva le informazioni rilevanti per il processo di ottimizzazione
    #trials_file_path: path del file in cui salvare i trials
    #max_evals: numero di valutazioni massimo della funzione di costo
    pc.dump(trials, open(trials_file_path, "wb"))
    setting = '{}'.format('EMANUELE')
    print(hyperparameters)
    forecaster = DNNModel(batch_size=int(hyperparameters['batch_size']),seq_len=int(hyperparameters['seq_len']),weight_decay=int(hyperparameters['weight_decay']),lstm_units=int(hyperparameters['lstm_units']),nlayers=int(hyperparameters['nlayers']),lr=(hyperparameters['lr']),dropout=(hyperparameters['dropout']))
    forecaster.train(setting).to("cpu")

    # Adattare a GPU
    # forecaster.train(setting).to("cpu")

    Yp_mean ,Y_test= forecaster.vali_test(setting,test=True)
    Y_test = Dataset(flag='test',size=[1,1,1],freq='10m').inverse_transform(Y_test.reshape(-1, Y_test.shape[-1])).flatten()
    Yp_mean = Dataset(flag='test',size=[1,1,1],freq='10m').inverse_transform(Yp_mean.reshape(-1, Yp_mean.shape[-1])).flatten()
    mae_validation = np.mean(MAE(Yp_mean, Y_test))#calcola la media del mae sul validation set e sul test set
    smape_validation = np.mean(RMSE(Yp_mean, Y_test))
    differenza=(abs(Yp_mean-Y_test)).flatten()#calcola gli errori in modulo sul test set
    print("errore max", max(differenza))#printa il massimo e il minimo dell'errore commesso
    print("errore min", min(differenza))
    print("  MAE: {:.3f} | RMSE: {:.3f} %".format(mae_validation, smape_validation))
    return_values = {'loss': mae_validation, 'MAE test': mae_validation,'RMSE test': smape_validation, 'hyper': hyperparameters,'status': STATUS_OK}#i risultati del processo vengono ritornati tramite un dizionario
    if trials.losses()[0] is not None:#la condizione controlla che nell'oggetto trials ci siano effettivamente delle losses. Trials.losses() riporta la lista di loss salvate nel processo di ottimizzazione
        MAEVal = trials.best_trial['result']['MAE test']#riporta il miglior valore del MAE durante il processo di ottimizzazione.
        sMAPEVal = trials.best_trial['result']['RMSE test']#riporta il miglior valore del RMSE durante il processo di ottimizzazione
        parametri = trials.best_trial['result']['hyper']#riporta la migliore scelta di iperparametri
        print('\n\nTested {}/{} iterations.'.format(len(trials.losses()) - 1,max_evals))
        print('Best MAE - Validation Dataset')
        print("  MAE: {:.3f} | RMSE: {:.3f} %".format(MAEVal, sMAPEVal))
    return return_values

def hyperparameter_optimizer(path_hyperparameters_folder=os.path.join('.', 'experimental_files'),
                             new_hyperopt=1, max_evals=3):
    if not os.path.exists(path_hyperparameters_folder):
        os.makedirs(path_hyperparameters_folder)
    trials_file_name = 'DNN_hyperparameters'
    trials_file_path = os.path.join(path_hyperparameters_folder, trials_file_name)
    if new_hyperopt:#Se new_hyperopt è True inizializza un nuovo oggetto trials
        trials = Trials()
    else:
        trials = pc.load(open(trials_file_path, "rb"))
    space = _build_space()
    fmin_objective = partial(_hyperopt_objective, trials=trials, trials_file_path=trials_file_path,
                             max_evals=max_evals)
    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)#fmin da hyperopt performa l'ottimizzazione utilizzando l'algoritmo Tree-structured Parzen Estimators

if __name__ == '__main__':
  import warnings
  warnings.filterwarnings("ignore")
  new_hyperopt = 1
  max_evals = 35
  path_hyperparameters_folder = "./experimental_files/"
  best_hyperparameters = hyperparameter_optimizer(path_hyperparameters_folder=path_hyperparameters_folder,new_hyperopt=new_hyperopt, max_evals=max_evals)
  trials_file_name = 'DNN_hyperparameters'
  trials_file_path = os.path.join(path_hyperparameters_folder, trials_file_name)
  trials = pc.load(open(trials_file_path, "rb"))
  for trial in trials.trials:
    print(trial['result'])
