import pandas as pd
import gymnasium as gym
import os
from os import path

class xrayEnv(gym.Env):
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

    def getObs(self):
        return self.obs

    def getInfo(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, return_info=False):
        self.obs = ("Diagnose a patient's knee osteoarthritis as either progressor or non-progressor using retrieve[entity] and finish[answer].\n")
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

    def searchStep(self, patientID, entity):
        patientRow = self.patientData.loc[self.patientData["ID"] == patientID]

        if patientRow is None:
            self.obs = f"Could not find patient {patientID}"
            return
    
        # Prevents model from cheating by looking at the answer in known patient data
        if entity == "GROUPTYPE":
            self.obs = f"I'm not allowed to look at column {entity}"
        else:
            try: 
                value = patientRow[entity].values[0]
                self.obs = f"Patient has {value} in column {entity}"
            except:
                self.obs = f"invalid entity: {entity}"

    def step(self, patientID, action):
        done = False

        if self.answer is not None:
            done = True
            return self.obs, done, self.getInfo()

        if action.startswith("retrieve[") and action.endswith("]"):
            entity = action[len("retrieve["):-1]
            self.searchStep(patientID, entity)

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

def test():
    df = pd.read_csv(f"data/xray_data.csv")
    assert(df is not None)

    env = xrayEnv()

    env.loadPatientData(df)
    assert(env.patientData is not None)
    assert(env.patientIDs is not None)

    env.searchStep(9001695, "V00WOMKP")
    assert(env.obs == "Patient has 0 in column V00WOMKP")

    env.searchStep(9001695, "GROUPTYPE")
    assert(env.obs == "I'm not allowed to look at column GROUPTYPE")

    print("All tests passed")

if __name__ == "__main__":
    test()
