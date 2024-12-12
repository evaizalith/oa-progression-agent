import torch
import os
from os import path
import pandas as pd
from netEnv import xrayNetEnv
from dataHandler import dataHandler
from huggingface_hub import login
from transformers import pipeline
from sklearn.model_selection import train_test_split

def llm(prompt, pipe, stop=["\n"]):
    response = pipe(
            prompt,
            max_new_tokens=200,
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
            print(f"Thought {i}: ", thought_action)
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

def main():
    login()

    pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            device_map="auto")

    pipe("The key to life is")

    env = xrayNetEnv()
    data_handler = dataHandler() 
    data_handler.setTrain("data/Clinical_FNIH.csv")
    data_handler.setTest("data/clinical_data.csv")
    x, y = data_handler.preprocess(showData=False)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

    env.nn.train(50, 64, x_train, y_train, x_val, y_val)

    data_handler.setXray("data/xray_data.csv")
    xray = data_handler.getJSW()

    env.patientData = pd.read_csv("data/Clinical_FNIH.csv")
    env.data_handler = data_handler

    print(env.process("5.397, 5.909, 5.156, 5.156, 2:02, 29.2, 72.0"))

    instruction = """
    Diagnose a patient with either a progressor or a non-progressor utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be two types:

(1) retrieve[patient], which collects patient data

(2) process[data], which provides the patient data to your internal neural network and allows you to determine whether or not the patient is a progressor or a non-progressor

(3) finish[answer], which returns the answer and diagnoses the patient. The action "finish[progressor]" will diagnose a patient as a progressor; while the action "finish[non-progressor]" will diagnose a patient as a non-progressor.

Here are some examples.
    """

    examples = "Question: Predict the OA progression for a patient with V00MCMJSW = 4.488, V01MCMJSW = 3.9, V03MCMJSW = 3.9, V05MCMJSW = 0.0, V00XRKL = 2:02, P01BMI = 28.6, V00AGE = 52\nThought 1: The patient's V00MCMJSW is above 4.7, which would indicate that this patient's condition will not progress. I should double check using my neural network. Thought 2: I will store this data as the vector [4.488, 3.9, 3.9, 0.0, 2:02, 28.6, 52.0] and submit it to my neural network.\nAction 1: process[4.488, 3.9, 3.9, 0.0, 2:02, 28.6, 52.0]\nObservation 1: The likelihood of progression is 0.83\nThought 2: The likelihood of progression is greater than 0.5, therefore the patient's condition is likely to progress.\nAction 2: finish[progressor]Question: Predict the OA progression for a patient with V00MCMJSW = 2.886, V01MCMJSW = 1.781, V03MCMJSW = 1.649, V05MCMJSW = 1.3, V00XRKL = 3:03, P01BMI = 36.5, V00AGE = 61\nThought 1: The patient's V00MCMJSW is below 4.7, which would indicate that this patient's condition will likely progress. I should double check using my neural network. Thought 2: I will store this data as the vector [2.886, 1.781, 1.649, 1.3, 3:03, 36.5, 61.0] and submit it to my neural network.\nAction 1: process[2.886, 1.781, 1.649, 1.3, 3:03, 36.5, 61.0]\nObservation 1: The likelihood of progression is 0.94\nThought 2: The likelihood of progression is greater than 0.5, therefore the patient's condition is likely to progress.\nAction 2: finish[progressor]\nQuestion: Predict the OA progression for a patient with V00MCMJSW = 5.141, V01MCMJSW = 4.868, V03MCMJSW = 4.681, V05MCMJSW = 4.783, V00XRKL = 2:02, P01BMI = 36.0, V00AGE = 64\nThought 1: The patient's V00MCMJSW is above 4.7, which would indicate that this patient's condition will not progress. I should double check using my neural network. Thought 2: I will store this data as the vector [5.141, 4.868, 4.681, 4.783, 2:02, 36.0, 64.0] and submit it to my neural network.\nAction 1: process[5.141, 4.868, 4.681, 4.783, 2:02, 36.0, 64.0]\nObservation 1: The likelihood of progression is 0.32\nThought 2: The likelihood of progression is less than 0.5, therefore the patient's condition is unlikely to progress.\nAction 2: finish[non-progressor]"

    question = "Predict the OA progression for a patient with V00MCMJSW = 5.397, V01MCMJSW = 5.909, V03MCMJSW = 5.156, V05MCMJSW = 5.156, V00XRKL = 2:02, P01BMI = 29.2, V00AGE = 72.0"

    prompt = instruction + examples #+ question

    infos = []

    #env.searchStep(9002817)
    #print(env.obs)
    #env.process("0.0, 85.94, 22.9, 77.0, 1.0, 2.493")
    #print(env.obs)

    think(env, 1, prompt, 0, pipe)

    #for i in range(0, 1):
    #    print("================")
    #    print(f"Generation {i}")
    #    patient = patients.iloc[i]
    #    pat_id = patient["ID"]
    #    info = think(env, i, prompt, pat_id, pipe)
    #    infos.append(info)
    #    print(info)

    #print(infos)

if __name__ == "__main__":
    main()
