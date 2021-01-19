#import sklearn
from json import dumps
import random
from random import sample
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import ast
#from sklearn.linear_model import Ridge,LinearRegression
#from sklearn.ensemble import AdaBoostRegressor ,GradientBoostingRegressor
#from sklearn.feature_selection import f_regression
#from sklearn.model_selection import cross_val_score , RepeatedKFold ,train_test_split
#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.model_selection import train_test_split
#import seaborn as sns
#from sklearn.preprocessing import StandardScaler
#import cufflinks as cf
#import matplotlib.pyplot as plt
#from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#import warnings
#warnings.filterwarnings('ignore')
# evaluate adaboost ensemble for regression

def MAE_STD(DF,PointName):
    Real_Y=DF['Real_Y']
    STD_Y = Real_Y.std()
    Phase_I=DF['Phase I']
    Phase_II=DF['Phase II']

    Error_I = abs(Real_Y-Phase_I)
    Error_II= abs(Real_Y-Phase_II)
    A = int(len(Error_I)*0.95)  #95% Error
    Error_I=Error_I.sort_values()
    Error_I = Error_I.reset_index(drop=True)
    Error_II=Error_II.sort_values()
    Error_II = Error_II.reset_index(drop=True)
    MAE_I = mean_absolute_error(Real_Y,Phase_I)
    MAE_II = mean_absolute_error(Real_Y,Phase_II)
    errorI_95=Error_I[A]
    errorII_95=Error_II[A]
    ttt = {'Phase I 95%Error':errorI_95,'Phase I MAE':MAE_I,'Phase II 95%Error':errorII_95,
           'Phase II MAE':MAE_II,'STD':STD_Y}
    return ttt

def split_train_test(X,Y,num):
    Train_X = X.copy()
    Train_Y = Y.copy()
    train_X = Train_X.copy()[0:num]
    train_Y = Train_Y.copy()[0:num]
    test_X = Train_X.copy()[num:]
    test_Y = Train_Y.copy()[num:]
    return train_X,train_Y,test_X,test_Y

def Rate_pump(Y,num):
    Big_Data = {}
    for num in range(3,num+1):
        rand_ind = random.sample(range(0,10),num)
        rand_ind = np.array(rand_ind)
        rand_ind = np.array(rand_ind)
        rand_ind = np.sort(rand_ind)
        while True:
            if rand_ind[-1] > len(Y)-1:
                break
            else:
                rand_ind = np.concatenate((rand_ind,rand_ind[-num:]+10),axis=0)
        while rand_ind[-1] > len(Y)-1:
            rand_ind = np.delete(rand_ind,-1,axis = 0)
        sum_l = np.concatenate((rand_ind,[len(Y)-1]),axis=0)
        sum_l = np.unique(sum_l)
        percent = str(num*10)+'%'
        Big_Data[percent]=Y.index[sum_l].tolist()
    return Big_Data


def Rate_pump_bylot(df_lot, df, nums):
    Big_Data = {}
    df_0 = df.copy()
    df = df[:-1]
    df_lot = df_lot.loc[df.index]
    Y = pd.unique(df_lot['Lot ID'])
    for num in range(3, nums + 1):
        rand_ind = random.sample(range(0, 10), num)
        rand_ind = np.array(rand_ind)
        rand_ind = np.sort(rand_ind)
        while True:
            if rand_ind[-1] > len(Y) - 1:
                break
            else:
                rand_ind = np.concatenate((rand_ind, rand_ind[-num:] + 10), axis=0)
        while rand_ind[-1] > len(Y) - 1:
            rand_ind = np.delete(rand_ind, -1, axis=0)

        mask = df_lot['Lot ID'].isin(np.array(Y[rand_ind].tolist()))
        context_id = np.array(df_lot[mask].index)
        last_one = np.array([df_0.index[len(df_0) - 1]])
        sum_l = np.concatenate((context_id, last_one), axis=0)
        sum_l = np.unique(sum_l)
        percent = str(num * 10) + '%'
        Big_Data[percent] = sum_l.tolist()
        print(len(sum_l.tolist()))
    return Big_Data


def Predict_Pump(trainX, trainY, testX, testY, test_No, PointName):
    j = 0
    train_1 = len(trainX)

    First_train_X = trainX
    First_train_Y = trainY
    Real_Y = testY
    No_Y = testX.index

    model = XGBRegressor(objective='reg:squarederror',random_state=42)
    model.fit(First_train_X, First_train_Y)

    best_para = model.feature_importances_ > 0.001

    train_X = First_train_X[First_train_X.columns[best_para]]

    First_test_X = testX
    test_X = First_test_X[First_test_X.columns[best_para]]
    test_X = test_X.reset_index(drop=True)
    model.fit(train_X, First_train_Y)
    All_Phase_I = []
    All_Phase_II = []
    # print(type(test_No))
    # 測試資料#################   一進一出   ################
    for ind in tqdm(range(len(testX))):
        # print(test_X.index)
        test_X = First_test_X[First_test_X.columns[best_para]]
        test_X = test_X.reset_index(drop=True)
        test_X = test_X.iloc[ind:(ind + 1)]
        Pred_Pump = model.predict(test_X)
        All_Phase_I.append(Pred_Pump)
        # print(First_test_X.index[ind])
        if (First_test_X.index[ind] in test_No):

            # print(First_test_X.index[ind])
            First_train_X = pd.concat([First_train_X.iloc[1:train_1, :], First_test_X.iloc[ind:(ind + 1)]], axis=0)
            # First_train_X = First_train_X.reset_index(drop=True)
            First_train_Y = pd.concat([First_train_Y[1:train_1], Real_Y[ind:(ind + 1)]], axis=0)
            # First_train_Y = First_train_Y.reset_index(drop=True)

            model = XGBRegressor(objective='reg:squarederror',random_state=42)
            model.fit(First_train_X, First_train_Y)
            best_para = model.feature_importances_ > 0.001
            train_X = First_train_X[First_train_X.columns[best_para]]
            test_X = First_test_X[First_test_X.columns[best_para]]
            # test_X = test_X.reset_index(drop=True)
            model.fit(train_X, First_train_Y)

            for i in range(j + 1, 0, -1):
                Pred_Pump_II = model.predict(test_X.iloc[ind - (i - 1):ind - (i - 2)])
                All_Phase_II.append(Pred_Pump_II)
            j = 0
        else:
            j += 1

    All_Phase_I = np.array(All_Phase_I)
    All_Phase_II = np.array(All_Phase_II)
    Real_y = np.array(Real_Y)
    Error_I = abs(All_Phase_I[:, 0] - Real_y)
    Error_II = abs(All_Phase_II[:, 0] - Real_y[:len(All_Phase_II)])
    df = pd.DataFrame(
        {'Real_Y': Real_y, 'Phase I': All_Phase_I[:, 0], 'Phase II': All_Phase_II[:, 0], 'Error_I': Error_I,
         'Error_II': Error_II}, index=No_Y)
    ttt = MAE_STD(df, PointName)
    return df, ttt


def main(file):
    Rate_Data = {}
    condition_df = file
    df = ast.literal_eval(condition_df['Data'])
    df = pd.DataFrame(df)

    quantity = int(condition_df['train_quantity'])
    if condition_df['Flag'] == "context_id":
        print("By Context ID")
        # condition_df['subtract_number'] = []
        if condition_df['subtract_number'] == []:
            Rate_Data_1 = Rate_pump(df[quantity:], 10)
            Rate_Data = Rate_Data_1
            # print(len(Rate_Data['100%']))
            # print(len(Rate_Data['90%']))
            #print('0')
        else:
            Rate_Data_2 = {}
            condition_df['subtract_number'].append(df.index[-1])
            Rate_Data_2['subtract_number'] = condition_df['subtract_number']
            Rate_Data_2['100%'] = df.index.tolist()[quantity:]
            Rate_Data = Rate_Data_2
            # print(Rate_Data['subtract_number'][-1])
            # print(Rate_Data['100%'][-1])
            #print('1')
    elif condition_df['Flag'] == "lot_id":
        print("By Lot ID")
        df_lot = pd.DataFrame(condition_df['Lot'])
        Rate_Data_3 = Rate_pump_bylot(df_lot, df[quantity:], 10)
        Rate_Data = Rate_Data_3
        #print('2')

    
    # define the model
    model = ["LinearRegression()","AdaBoostRegressor()","GradientBoostingRegressor()","XGBRegressor(random_state=42)"]
    best_model = eval(model[3])
    #Rate_Data = Rate_pump(df, 10)
    Data_DC = {}
    ACC_DC = {}
    Con_ID = {}
    data_raw = df
    all_Site = condition_df['Y_point']
    quantity = int(condition_df['train_quantity'])
    for Rate_num in Rate_Data.items():
        print(Rate_num[0])
        ACC = {}
        Big_Data = {}
        for Y_name in all_Site:
            X = data_raw[condition_df['feature']].copy()
            X =X.fillna(axis=0,method='ffill')
            print(X.shape)
            print(Y_name)
            print(all_Site)
            Y = data_raw[Y_name].copy()
            train_X,train_Y,test_X,test_Y = split_train_test(X,Y,quantity)
            DF_2,ttt=Predict_Pump(train_X,train_Y,test_X,test_Y,Rate_num[1],Y_name)
            Big_Data[Y_name]=DF_2.to_dict()
            ACC[Y_name] = ttt
        Con_ID[Rate_num[0]] = Rate_num[1]
        Data_DC[Rate_num[0]]=Big_Data
        ACC_DC[Rate_num[0]]=ACC

    result = {"Detal": Data_DC, "Accuracy": ACC_DC, "Context_ID": Con_ID}
    result = dumps(result)

    return result

