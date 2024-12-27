import pandas as pd
import os
from os import path
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class dataHandler():
    def __init__(self):
        self.train_set = "trainingData"
        self.xray_set = "xraySet"
        self.test_set = "testData"
        self.cols = ['ID', 'V00MCMJSW', 'V01MCMJSW', 'V03MCMJSW', 'V05MCMJSW', 'V00XRKL', 'P01BMI', 'V00AGE', 'GROUPTYPE']
        self.colMax = {}
        self.colMin = {}

    def setTrain(self, file):
        if not path.exists(file):
            print("Error: unable to find path to training data")
            return False

        self.train_set = file
        return True

    def setTest(self, file):
        if not path.exists(file):
            print("Error: unable to find path to test data")
            return False
        
        self.test_set = file
        return True

    def setXray(self, file):
        if not path.exists(file):
            print("Error: unable to find path to xray data")
            return False

        self.xray_set = file
        return True

    def preprocess(self, balanceClasses=True, showData=False):
        df = pd.read_csv(self.train_set, usecols=self.cols)
        if df is None:
            print(f"Error: unable to load training data {self.train_set}")
            exit()

        df = df[df.GROUPTYPE != "Pain Only Progressor"]

        df['GROUPTYPE'] = df['GROUPTYPE'].apply(lambda x: '0'if x == "Non Progressor" else 1)

        if balanceClasses == True:
            balanced = df.groupby('GROUPTYPE')
            df = pd.DataFrame(balanced.apply(lambda x: x.sample(balanced.size().min()).reset_index(drop=True)))

        labels = df['GROUPTYPE']
        id = df['ID']
        df.drop(['ID', 'GROUPTYPE'], axis=1, inplace=True)

        if 'V00XRKL' in df.columns:
            df['V00XRKL'] = df['V00XRKL'].apply(lambda x: -1 if x == "1:01" else 0 if x == "2:02" else 1)

        df.fillna(0.0)

        # Normalize data
        # f(x) : x -> [0, 1]
        for col in df.columns:
            max = df[col].max()
            min = df[col].min()
            df[col] = (df[col] - min) / (max - min)
            self.colMax[col] = max
            self.colMin[col] = min

        training_data = pd.DataFrame().reindex_like(df)
        training_data.iloc[:] = df.iloc[:].astype(float)
        labels = labels.astype(float)

        if showData == True:
            print(df.head())
            nonprog = len(labels.loc[labels[:] == 0])
            prog = len(labels) - nonprog
            print(f"{prog + nonprog} total rows of training data after cleaning; {nonprog} non-progressor; {prog} progressor")

        return training_data, labels

    def getJSW(self):
        data = pd.read_csv(self.xray_set, usecols=['ID', 'V00MCMJSW'])

        return data

    def normalize(self, data):
        data = [float(i) for i in data]
        result = []
        i = 0
        for col in self.colMax:
            result.append((data[0] - self.colMin[col]) / (self.colMax[col] - self.colMin[col]))
            i += 1
        return result

    def xrayNormalize(self, output_size=(512, 512)):
        transform = transforms.Compose([transforms.Resize(output_size), transforms.ToTensor()])

        resized_images = {}

        df = pd.read_csv(self.train_set, usecols=['ID', 'GROUPTYPE'])

        #balanced = df.groupby('GROUPTYPE')
        #df = pd.DataFrame(balanced.apply(lambda x: x.sample(balanced.size().min()).reset_index(drop=True)))

        df['GROUPTYPE'] = df['GROUPTYPE'].apply(lambda x: 0.0 if x == "Non Progressor" else 1.0)
        id_list = df['ID'].astype(str).to_dict()
        condensed_id = []

        for root, dirs, files in os.walk(self.xray_set):

            names = [os.path.splitext(file) for file in files]
            condensed_id = [f for f in names if f in id_list]

            for file in files:
                pref, ext = os.path.splitext(file)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if condensed_id and pref not in id_list:
                        continue
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert("RGB")
                    resized_img = transform(img)
                    resized_img = torch.permute(resized_img, (0, 1, 2))
                    resized_images[int(pref)] = resized_img

        print(f"Normalized {len(resized_images)} x-ray images")

        n = len(df[df['GROUPTYPE'] == 0.0])
        print(n)

        df = df[~df['ID'].isin(condensed_id)]

        labels = dict(zip(df['ID'], df['GROUPTYPE']))


        images = {key: resized_images[key] for key in resized_images.keys() & labels.keys()}
        labels = {key: labels[key] for key in resized_images.keys() & labels.keys()}

        print(f"{len(labels)} labels")

        x = list(images.values())
        y = list(labels.values())

        n = y.count(0.0)
        print(f"total: {len(y)}; n = {n}")

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1)

        x_train = torch.stack(x_train, dim=0)
        x_val = torch.stack(x_val, dim=0)

        return x_train, x_val, y_train, y_val


