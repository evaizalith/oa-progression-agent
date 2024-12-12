import pandas as pd
import gymnasium as gym
import os
from os import path
from nn import oaClassifier
import torch

class xrayNetEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.patient = None     # Patient ID
        self.obs = None         # Current observation
        self.steps = 0
        self.answer = None
        self.patientData = None
        self.patientIDs = None
        self.observation_space = self.action_space = None
        self.n_rows = 0
        self.nn = oaClassifier()
        self.data_handler = None

    def getObs(self):
        return self.obs

    def getInfo(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, return_info=False):
        #self.obs = ("Diagnose a patient's knee osteoarthritis as either progressor or non-progressor using process[entity] and finish[answer].\n")
        self.obs = "Predict the OA progression for a patient with V00MCMJSW = 5.397, V01MCMJSW = 5.909, V03MCMJSW = 5.156, V05MCMJSW = 5.156, V00XRKL = 2:02, P01BMI = 29.2, V00AGE = 72.0"

        self.steps = 0
        self.answer = None
        observation = self.getObs()
        info = self.getInfo()
        if (return_info):
            return observation, info
        else:
            return observation

    def loadPatientData(self, df):
        self.patientData = df

        self.patientIDs = df["ID"]
        self.n_rows = len(self.patientIDs)

    def getJSWPred(self, df, id):
        patient = df.loc[df["ID"] == id]
        mcmjsw = patient["V00MCMJSW"]

        progressor = False

        if mcmjsw.item() < 4.7:
            progressor = True

        return progressor

    def searchStep(self, patientID):
        patientRow = self.patientData.loc[self.patientData["ID"] == patientID]

        if patientRow is None:
            self.obs = f"Could not find patient {patientID}"
            return
   
        # We drop some undesirable columns which would allow the model to cheat, or which may provide it with irrelevant information
        patientRow.drop(['ID', 'GROUPTYPE'], axis=1, inplace=True)
        
        patientRow['SIDE'] = patientRow['SIDE'].apply(lambda x: 0 if x == "1: Right" else 1)
        patientRow['P02SEX'] = patientRow['P02SEX'].apply(lambda x: 0 if x == "1: Male" else 1)

        data = pd.DataFrame().reindex_like(patientRow)
        data.iloc[:] = patientRow.iloc[:].astype(float)

        side = data['SIDE'].iloc[0]
        width = data['V00CFWDTH'].iloc[0]
        bmi = data['P01BMI'].iloc[0]
        age = data['V00AGE'].iloc[0]
        sex = data['P02SEX'].iloc[0]
        jsw = data['V00MCMJSW'].iloc[0]

        self.obs = f"Patient has SIDE = {side}, V00CFWDTH = {width}, P01BMI = {bmi}, V00AGE = {age}, P02SEX = {sex}, and V00MCMJSW = {jsw}"

    def process(self, entity):
        string_val = entity.split(",")

        string_val = [item.replace('1:01', '-1') for item in string_val]
        string_val = [item.replace('2:02', '0') for item in string_val]
        string_val = [item.replace('3:03', '1') for item in string_val]


        values = self.data_handler.normalize(string_val)
        x = self.nn.forward(torch.as_tensor(values, dtype=torch.float32))
       
        result = x.item()

        self.obs = f"The likelihood of progressing is {result}"

    def step(self, patientID, action):
        done = False

        if self.answer is not None:
            done = True
            return self.obs, done, self.getInfo()

        if action.startswith("retrieve[") and action.endswith("]"):
            entity = action[len("retrieve["):-1]
            self.searchStep(patientID)

        elif action.startswith("process[") and action.endswith("]"):
            entity = action[len("process["):-1]
            self.process(entity)

        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished"

        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Thought"
            
        else:
            self.obs = f"invalid action: {action}"

        self.steps += 1

        return self.obs, done, self.getInfo()
