import pandas as pd
import os
from os import path

class dataHandler():
    def __init__():
        self.train_set = "trainingData"
        self.test_set = "testData"
        self.cols = ['ID', 'V00MCMJSW', 'V00XRKL', 'P01BMI', 'V00AGE', 'V00JSW200', 'V00RPCFDS', 'GROUPTYPE']

    def setTrain(file):
        if not path.exists(file):
            print("Error: unable to find path to training data")
            return False

        self.train_set = file
        return True

    def setTest(file):
        if not path.exists(file):
            print("Error: unable to find path to test data")
            return False
        
        self.test_set = file
        return True

    def preprocess(balanceClasses=True, showData=False):
        df = pd.read_csv(self.train_set, usecols=self.cols)
        if df is None:
            print(f"Error: unable to load training data {self.train_set}")
            exit()

        df = df[df.grouptype != "Pain Only Progressor"]

        df['GROUPTYPE'] = df['GROUPTYPE'].apply(lambda x: '0'if x == "Non Progressor" else 1)

        if balanceClasses == True:
            balanced = df.groupby('GROUPTYPE')
            df = pd.DataFrame(balanced.apply(lambda x: x.sample(balanced.size().min()).reset_index(drop=True)))

        labels = df['GROUPTYPE']
        id = df['ID']
        df.drop[['ID', 'GROUPTYPE'], axis=1, inplace=True]

        df.fillna(0.0)

        if 'V00XRKL' in df.columns:
            df['V00XKRL'] = df['V00XRKL'].apply(lambda x: -1 if x == "1:01" else 0 if x == "2:02" else 1)

        # Normalize data
        # f(x) : x -> [0, 1]
        for col in df.columns:
            max = df.max()
            min = df.min()
            df[col] = (df[col] - min) / (max - min)

        training_data = pd.DataFrame().reindex_like(df)
        training_data.iloc[:] = df.iloc[:].astype(float)
        labels = labels.astype(float)

        if showData == True:
            print(df.head())
            nonprog = len(labels.loc[labels[:] == 0])
            prog = len(labels) - nonprog
            print(f"{prog + nonprog} total rows of training data after cleaning; {nonprog} non-progressor; {prog} progressor")

        return training_data, labels

    def getProgFromJSW():
        data = df.read_csv(self.train_set, cols=['ID', 'V00MCMJSW'])
        progressor = df[df.V00MCMJSW < 4.7]
        return progressor 
