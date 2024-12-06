import torch
import os
from os import path
import pandas as pd
from netEnv import xrayNetEnv
from huggingface_hub import login
from transformers import pipeline
from sklearn.model_selection import train_test_split

def llm(prompt, pipe, stop=["\n"]):
    response = pipe(
            prompt,
            max_new_tokens=100,
            truncation=True,
            do_sample=True,
            top_p=1,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
            )
    output = response[0]["generated_text"]

    for stop_token in stop:
        text_output = output.split(stop_token)[0]

    return text_output

def think(env, idx, prompt, pat_id, pipe):
    question = env.reset()

    print(idx, question)

    prompt += question
    prompt += "\n"
    n_calls = 0
    n_badcalls = 0

    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought: {i}:", pipe, stop=[f"\nObservation {i}:"])


        try:
            thought, action = thought_action.strip().split("f\nAction{i}")
        except:
            #print(f"Thought {i}: ", thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", pipe, stop=[f"\n"]).strip()

        obs, done, info = env.step(pat_id, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')

        #print(f"Observation {i}: ", obs)

        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str

        print(step_str)

        if done:
            break

    if not done:
        obs, done, info = env.step(pat_id, "finish[]")

    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return info

# Loads in data file 
def prepareEnv(file, env):
    if not path.exists(file):
        print("Unable to find data {path}")
        exit()
    
    cols = ['ID', 'SIDE', 'V00CFWDTH', 'P01BMI', 'V00AGE', 'P02SEX', 'V00MCMJSW', 'GROUPTYPE']
    df = pd.read_csv(file, usecols=cols)

    if df is None:
        print("Error: unable to load file {file} into dataframe")
        exit()

    n_rows = len(df["ID"])
    copy = df.copy(deep=True)
    env.loadPatientData(copy)

    labels = df['GROUPTYPE']
    df.drop(['ID', 'GROUPTYPE'], axis=1, inplace=True)
    
    df.fillna(0.0)
    training_data = df

    # Discretize SIDE, PO2SEX
    training_data['SIDE'] = training_data['SIDE'].apply(lambda x: 0 if x == "1: Right" else 1)
    training_data['P02SEX'] = training_data['P02SEX'].apply(lambda x: 0 if x == "1: Male" else 1)

    x = pd.DataFrame().reindex_like(training_data)
    x.iloc[:] = training_data.iloc[:].astype(float)

    # Discretize labels into binary 0 (non-progressor) and 1 (progressor)
    y = labels.apply(lambda x: '0' if x == "Non Progressor" else '1')
    y = y.astype(float)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    env.nn.train(10, 30, x_train, y_train, x_val, y_val)

    return env, n_rows

def main():
    login()

    pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            device_map="auto")

    pipe("The key to life is")

    env = xrayNetEnv()
    env, n_rows = prepareEnv("data/xray_data.csv", env)

    instruction = """
    Diagnose a patient with either a progressor or a non-progressor utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be two types:

(1) retrieve[patient], which collects patient data

(2) process[data], which provides the patient data to your internal neural network and allows you to determine whether or not the patient is a progressor or a non-progressor

(3) finish[answer], which returns the answer and diagnoses the patient. The action "finish[progressor]" will diagnose a patient as a progressor; while the action "finish[non-progressor]" will diagnose a patient as a non-progressor.

Here are some examples.
    """

    #examples = """Thought 1: First, I retrieve patient data.
    #Action 1: retrieve[patient]
    #Observation 1: Patient has SIDE = 0.0, V00CFWDTH = 85.94, P01BMI = 22.9, V00AGE = 77.0, P02SEX = 1.0, and V00MCMJSW = 2.493
    #Thought 2: Next, I will pass this data to the neural network.
    #Action 2: process[0.0, 85.94, 22.9, 77.0, 1.0, 2.493]
    #Observation 2: The likelihood of progressing is 0.7875
    #Thought 3: Because the likelihood of progressing is 0.7875, which is greater than 0.5, I can say that this is a progressor.
    #Action 3: finish[progressor]
    #"""

    examples = "Question: Predict the OA progression for a patient with SIDE = 0.0, V00CFWDTH = 85.94, P01BMI = 22.9, V00AGE = 77.0, P02SEX = 1.0, and V00MCMJSW = 2.493\nThought 1: I will store this data as the vector [0.0, 85.94, 22.9, 77.0, 1.0, 2.493] and submit it to my neural network.\nAction 1: process[0.0, 85.94, 22.9, 77.0, 1.0, 2.493]\nObservation 1: The likelihood of progression is 0.83\nThought 2: The likelihood of progression is greater than 0.5, therefore the patient's condition is likely to progress.\nAction 2: finish[progressor]\nQuestion: Predict the OA progression for a patient with SIDE = 1.0, V00CFWDTH = 83.48, P01BMI = 22.4, P02SEX = 0.0, and V00MCMJSW = 3.9.\nThought 1: I will store this data as the vector [1.0, 83.48, 22.4, 0.0, 3.9] and submit it to my neural network.\nAction 1: process[1.0, 83.48, 22.4, 0.0, 3.9]\nObservation 1: The likelihood of progression is 0.27\nThought 2: The likelihood of progression is less than 0.5, therefore the patient's condition is unlikely to progress.\nAction 2: finish[non-progressor]"

    question = "Predict the OA progression for a patient with SIDE = 1.0, V00CFWDTH = 90.0, P01BMI = 25.0, V00AGE = 50.0, P02SEX = 1.0, and V00MCMJSW = 3.5"

    prompt = instruction + examples #+ question

    infos = []
    patients = env.patientData.sample(n=10)

    env.searchStep(9002817)
    print(env.obs)
    env.process("0.0, 85.94, 22.9, 77.0, 1.0, 2.493")
    print(env.obs)

    for i in range(0, 1):
        print("================")
        print(f"Generation {i}")
        patient = patients.iloc[i]
        pat_id = patient["ID"]
        info = think(env, i, prompt, pat_id, pipe)
        infos.append(info)
        print(info)

    print(infos)

if __name__ == "__main__":
    main()
