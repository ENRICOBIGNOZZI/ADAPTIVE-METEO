# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1X2gVRihdkjO2MgDg_DfVeRvohzL7k1Qx
"""

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
import argparse


def MAE(pred, true):                                 #Metriche utili per la valutazione delle prestazioni
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

df= pd.DataFrame()

# Definisci il parser degli argomenti
parser = argparse.ArgumentParser(description='Graph')
parser.add_argument('--input_file', type=str, help='Percorso del file CSV di input')
parser.add_argument('--int_value', type=int, help='Un valore intero')

# Effettua il parsing degli argomenti dalla riga di comando
args = parser.parse_args()

# Se è stato specificato un file CSV come argomento
if args.input_file:
    df = pd.read_csv(args.input_file)
    # Ora df contiene i dati dal file CSV
    print(df)
else:
    # Altrimenti, chiedi all'utente di inserire il percorso del file CSV
    input_file = input('Inserisci il percorso del file CSV: ')
    
    if input_file:
        df = pd.read_csv(input_file)
        # Ora df contiene i dati dal file CSV
        print(df)
    else:
        print('Devi specificare un percorso per il file CSV.')

# Chiedi all'utente di inserire il valore intero dopo aver letto il file CSV
int_value = input('Inserisci un valore intero: ')

try:
    time_step = int(int_value)
    print(f'Il valore intero specificato è: {int_value}')
except ValueError:
    print('Il valore inserito non è un intero valido.')


df['DATE'] = pd.to_datetime(df['DATE'])

df.columns

col0=['DATE',
      'PRESS_STA0',
      'TEMP_STA0',
      'HUM_STA0',
      'PRESS_STA1',
      'TEMP_STA1',
      'HUM_STA1',
      'PRESS_STA2',
      'TEMP_STA2',
      'HUM_STA2',
      'PRESS_STA3',
      'TEMP_STA3',
      'HUM_STA3',
      'PRESS_STA4',
      'TEMP_STA4',
      'HUM_STA4',
      ]

df=df[col0]
co = df.columns[1:]
df = df.reset_index(drop=True)
df

desired_interval = 1008                                #Intervallo di tempo richiesto per cui le sequenze non presentino discontinuità
df["DATE"]=pd.to_datetime(df["DATE"])
desired_duration = timedelta(days=7)
good_starts = []
for i in range(df.shape[0]-desired_interval):                                      #Un ciclo for appende ad un vettore good_starts gli indici seguiti da sequenze continue
    if df["DATE"][i+desired_interval]-df["DATE"][i] == desired_duration:
        good_starts.append(df.index[i])
fea = df.shape[1]-1
good_starts_train = good_starts[0:int(len(good_starts)*0.8)]                                             # ---> Divisione del vettore di indici in train, validation e test
good_starts_vali = good_starts[int(len(good_starts)*0.8):int(len(good_starts)*0.9)]                      # ---> Evitando che sequenze nei tre diversi set si sovrappongano
good_starts_vali = [l for l in good_starts_vali if l>(good_starts_train[-1]+desired_interval+1)]
good_starts_test = good_starts[int(len(good_starts)*0.9):int(1*len(good_starts))]
good_starts_test = [l for l in good_starts_test if l>(good_starts_vali[-1]+desired_interval+1)]

shuffled_list = good_starts_train.copy()                                                       #Shuffle del vettore di indici per il train
random.shuffle(shuffled_list)
tr = shuffled_list
va = good_starts_vali
te =  good_starts_test
gs=[tr,va,te]                                                                                  #Vettore di tre vettori di indici da cui prendere le sequenze
df=df[:-1]

class SpecificIndicesSampler(Sampler):                           #Classe che serve a portare il vettore gs all'interno del Dataloader, definito in un'altra classe
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class Dataset():



    # features: stringa che specifica il tipo di task di previsione: M = multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    # verrà utilizzata soltanto l'opzione MS
    # target: stringa che specifica la variabile target
    # scale: se True si effettua lo scaling dei dati con lo Scaler, altrimenti False

    def __init__(self, root_path=None, flag='train', size=None,
                 features='MS',target='HUM_STA4', scale=True):

        # size [seq_len, pred_len]
        # info
        if size == None:
            print("size è none. Forse ci sono problemi")
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        # init
        assert flag in ['train', 'val','test']
        type_map = {'train': 0, 'val': 1, 'test':2}          #Mappa che specifica la fase di esecuzione
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler()                         #Effettua una lettura e un preprocessing dei dati
        df_raw = df                                          #Scaler che normalizza i dati
        border1 = 0
        border2 = int(len(df_raw)*1)



                                                             #Opzione di definizione dei borders per dati di train, validation e test
                                                             #In questo caso il Sampler gestisce gli indici da prendere per cui non è necessario fare
                                                             #una divisione del dataset

        if self.features == 'M' or self.features == 'MS':      #In questo caso viene utilizzato solo MS
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            stop_index = good_starts_train[-1]+desired_interval            #Scaling del dataset
            train_data = df_data.iloc[:stop_index + 1]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.scaler.fit(train_data[self.target].values.reshape(-1, 1))         # Fittare i dati da prevedere in modo che poi si possano riportare all'originale
        else:
            data = df_data.values
        self.data_x = data[border1:border2]                                      #Utilizza i borders prima definiti, sempre utile soltanto se fosse necessaria una divisione
        self.data_y = data[border1:border2]                                      #di dataset di train, validation e test


    def __getitem__(self, index):                                        #Usato per recuperare una specifica sequenza di dati dal dataset dato un certo indice di partenza
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_begin:s_end+self.pred_len]
        return seq_x,seq_y

    def __len__(self):                                                       #Ritorna il numero totale di sequenze nel dataset, poco utile in quanto le sequenze utilizzate sono
        return len(self.data_x) - self.seq_len - self.pred_len + 1           #meno di quelle previste a causa della richiesta di continuità temporale

    def inverse_transform(self, data):                                    #Effettua lo scaling inverso dei dati nel caso si sia scelto di riscalarli nel preprocessing
        return self.scaler.inverse_transform(data)

class Create_Mask(nn.Module):
    def __init__(self, n_nodes, seq_len):
        super(Create_Mask,self).__init__()

        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self):
        n = self.seq_len * self.n_nodes
        result_matrix_dim = (n, n)
        result_matrix = torch.zeros(result_matrix_dim)
        for i in range(self.seq_len):
            start_row = i * self.n_nodes
            end_row = (i + 1) * self.n_nodes
            result_matrix[start_row:end_row, start_row:end_row] = torch.ones((self.n_nodes, self.n_nodes))

        for i in range(n):
            for j in range(i+1, n):
                result_matrix[i, j] = 1

        for i in range(n):
            result_matrix[i,i]=0

        return(result_matrix.transpose(1,0))

class Model(nn.Module):
    """Modello Graph Convolutional seguito da una serie di Linear"""
    def __init__(
        self,
        seq_len,
        pred_len,
        n_features,
        batch_size,
        hidden

    ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden = hidden
        self.n_features = n_features
        self.batch_size = batch_size
        self.num_nodes=5

        self.W1 = nn.Parameter(torch.rand((self.num_nodes, self.num_nodes, self.n_features, self.hidden), dtype=torch.float32))  #Una matrice di pesi per ogni interazione nodo-nodo
        self.bias=nn.Parameter(torch.rand((self.seq_len, self.num_nodes, self.hidden), dtype=torch.float32))                     #L'interazione avviene soltanto fra nodi allo stesso tempo

        self.nodes=nn.Linear(5,1)                                                                                                #Tre linear combinano le informazioni spaziali e temporali
        self.feat=nn.Linear(self.hidden,1)
        self.time=nn.Linear(self.seq_len,1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, features, mask):
        A = mask
        features=features.reshape(self.batch_size,self.seq_len,self.num_nodes,self.n_features)
        prodotto1=torch.matmul(A,features)
        tutto = torch.einsum('ijkl,fklm->ijfm', prodotto1, self.W1)                           #Convoluzione
        conv1= torch.relu(tutto+self.bias)
        risultato = conv1
        risultato=self.feat(risultato)
        risultato=self.nodes(risultato.permute(0,1,3,2)).permute(0,1,3,2)
        risultato=self.time(risultato.permute(0,2,3,1)).permute(0,2,3,1)

        return risultato[:,-1,-1:,-1:]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):                       #EarlyStopping per evitare overfitting
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



class DNNModel(object):
    def __init__(self,batch_size,seq_len,hidden,lr,epochs_early_stopping=20):

        self.batch_size=batch_size
        self.pred_len=6*time_step
        self.seq_len=seq_len
        self.n_features=3
        self.n_nodes = 5
        self.hidden=hidden
        self.train_epochs=2
        self.features='MS'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints='./checkpoints/'
        self.lr=lr

        self.model = self._build_model()
        self.make_mask = Create_Mask(self.n_nodes, 1)

    def data_provider(self,flag):                                    #Considera se è fase di train, validation o test e seleziona il DataLoader di conseguenza
        data_dict = {'ETTh1': Dataset}
        Data = data_dict['ETTh1']

        if flag == 'val':
            drop_last =  True
            batch_size =  1
            v_s = 1

        elif flag == 'test':
            drop_last =  True
            batch_size =  1
            v_s = 2

        else:
            drop_last =  True
            batch_size =self.batch_size
            v_s = 0

        data_set = Data(
            flag=flag,
            size=[self.seq_len, self.pred_len],
        )

        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=drop_last,
            sampler=SpecificIndicesSampler(gs[v_s])                                 #Entra il vettore di Sampling sulla base di quale fase sia
            )
        return data_set, data_loader

    def _build_model(self):
        model = Model(seq_len=self.seq_len,pred_len=self.pred_len,n_features=self.n_features,batch_size=self.batch_size, hidden=self.hidden).to(self.device)
        return model                                         #Crea il modello

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_data(self, flag):
        data_set, data_loader = self.data_provider(flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):                              #Fase di Validation

        mask = self.make_mask().to(self.device)
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x, mask)
                f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0
                outputs = outputs[:, -1:, f_dim:]
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self,setting):                                                    #Fase di train

        mask = self.make_mask().to(self.device)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = ReduceLROnPlateau(model_optim, mode='min', patience=3, factor=0.5, verbose=False)
        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x, mask)
                f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0
                outputs = outputs[:, -1:, f_dim:]
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 10000 == 0:
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
            vali_loss= self.vali(vali_data, vali_loader, criterion)
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




    def vali_test(self, setting, test=True):                                            #Fase di test

        mask = self.make_mask().to(self.device)
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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
                outputs = self.model(batch_x, mask)
                f_dim = -1 if(self.features == 'MS' or self.features =='S')  else 0
                outputs = outputs[:, -1:, f_dim:]
                batch_y = batch_y[:, -1:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return preds,trues

def _build_space():                                                              #Genera lo spazio di ricerca per ottimizzare gli iperaparametri
    space = {
        'batch_size': 54, #hp.quniform('batch_size', 16, 256, 16),
        'seq_len': 50, #hp.quniform('seq_len', 50, 250, 10),
        'hidden': 24, #hp.quniform('hidden', 16, 64, 16),
        'lr': 0.001 #hp.uniform('lr', 0.001, 0.005),
        }
    return space



def _hyperopt_objective(hyperparameters, trials, trials_file_path, max_evals):             #Definisce l'obiettivo della minimizzazione per il processo di ottimizzazione degli iperparametri

    #Info input:
    #hyperparameters: dizionario contenente gli iperparametri da valutare
    #trials:oggetto che conserva le informazioni rilevanti per il processo di ottimizzazione
    #trials_file_path: path del file in cui salvare i trials
    #max_evals: numero di valutazioni massimo della funzione di costo

    pc.dump(trials, open(trials_file_path, "wb"))
    setting = '{}'.format('EMANUELE')
    print(hyperparameters)
    forecaster = DNNModel(batch_size=int(hyperparameters['batch_size']),
                          seq_len=int(hyperparameters['seq_len']),
                          hidden=int(hyperparameters['hidden']),
                          lr=(hyperparameters['lr']),
                          )

    forecaster.train(setting).to("cpu")
    Yp_mean ,Y_test= forecaster.vali_test(setting,test=True)
    Y_test = Dataset(flag='test',size=[1,1,1]).inverse_transform(Y_test.reshape(-1, Y_test.shape[-1])).flatten()
    Yp_mean = Dataset(flag='test',size=[1,1,1]).inverse_transform(Yp_mean.reshape(-1, Yp_mean.shape[-1])).flatten()
    mae_validation = np.mean(MAE(Yp_mean, Y_test))
    smape_validation = np.mean(RMSE(Yp_mean, Y_test))
    differenza=(abs(Yp_mean-Y_test)).flatten()              #Calcola gli errori in modulo sul test set
    print("errore max", max(differenza))                    #Printa il massimo e il minimo dell'errore commesso
    print("errore min", min(differenza))
    print("  MAE: {:.3f} | RMSE: {:.3f} %".format(mae_validation, smape_validation))
    return_values = {'loss': mae_validation, 'MAE test': mae_validation,'RMSE test': smape_validation, 'hyper': hyperparameters,'status': STATUS_OK} #I risultati del processo vengono ritornati tramite un dizionario
    if trials.losses()[0] is not None:              #La condizione controlla che nell'oggetto trials ci siano effettivamente delle losses. Trials.losses() riporta la lista di loss salvate nel processo di ottimizzazione
        MAEVal = trials.best_trial['result']['MAE test']   #Riporta il miglior valore del MAE durante il processo di ottimizzazione.
        sMAPEVal = trials.best_trial['result']['RMSE test']  #Riporta il miglior valore del RMSE durante il processo di ottimizzazione
        parametri = trials.best_trial['result']['hyper']   #Riporta la migliore scelta di iperparametri

        print('\n\nTested {}/{} iterations.'.format(len(trials.losses()) - 1,max_evals))
        print('Best MAE - Validation Dataset')
        print("  MAE: {:.3f} | RMSE: {:.3f} %".format(MAEVal, sMAPEVal))
    return return_values

def hyperparameter_optimizer(path_hyperparameters_folder=os.path.join('.', 'experimental_files'),
                             new_hyperopt=1, max_evals=1500):

    if not os.path.exists(path_hyperparameters_folder):
        os.makedirs(path_hyperparameters_folder)
    trials_file_name = 'DNN_hyperparameters'
    trials_file_path = os.path.join(path_hyperparameters_folder, trials_file_name)
    if new_hyperopt:                                                                  #Se new_hyperopt è True inizializza un nuovo oggetto trials
        trials = Trials()
    else:
        trials = pc.load(open(trials_file_path, "rb"))
    space = _build_space()

    fmin_objective = partial(_hyperopt_objective, trials=trials, trials_file_path=trials_file_path,
                             max_evals=max_evals)
    fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)      #fmin da hyperopt performa l'ottimizzazione utilizzando l'algoritmo Tree-structured Parzen Estimators

import warnings
warnings.filterwarnings("ignore")
new_hyperopt = 1
max_evals = 1
path_hyperparameters_folder = "./experimental_files/"
hyperparameter_optimizer(path_hyperparameters_folder=path_hyperparameters_folder,new_hyperopt=new_hyperopt, max_evals=max_evals)

trials_file_name = 'DNN_hyperparameters'
trials_file_path = os.path.join(path_hyperparameters_folder, trials_file_name)
trials = pc.load(open(trials_file_path, "rb"))
for trial in trials.trials:
    print(trial['result'])
