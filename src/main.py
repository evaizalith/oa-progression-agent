import torch
import os
from os import path
import pandas as pd
from xrayEnv import xrayEnv
from huggingface_hub import login
from transformers import pipeline

def llm(prompt, pipe, stop=["\n"]):
    response = pipe(
            prompt,
            max_new_tokens=100,
            truncation=True,
            do_sample=False,
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
            print("ohh...", thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", pipe, stop=[f"\n"]).strip()

        obs, done, info = env.step(pat_id, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
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

    df = pd.read_csv(file)
    n_rows = len(df["ID"])
    env.loadPatientData(df)
    return env, n_rows

def main():
    login()

    pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
            device_map="auto")

    pipe("The key to life is")

    env = xrayEnv()
    env, n_rows = prepareEnv("data/xray_data.csv", env)

    instruction = """
    Diagnose a patient with either a progressor or a non-progressor utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) retrive[entity], which searches the patient data and returns the requested information. Valid values for entity include: SIDE, V00CFWDTH, V00MCMJSW, V00JSW175, V00JSW200, V00JSW250, V00JSW300, V00JSW225, V00TPCFDS, V00BMANG, V00JSW150, V00JSW275, V00LJSW850, V00LJSW900, V00LJSW700, V00LJSW825, V00LJSW750, V00LJSW875, V00LJSW725, V00LJSW775, V00LJSW800, V00XMJSW, V00MJSWBB, V00XRJSM, V00XRKL, V00XRJSL, V00MCMJSW, P01KPMEDCV, P01BMI, V00AGE, P02SEX, P02HISP, P02RACE, V00WOMKP, V00WOMADL, V00BARCDJD, V00INCPLL, V00INCPLM, V00NOMMJSW, V00NOLJSWX, V00IMPIXSZ.

For example, the action "retrieve[SIDE]" will return the target knee; and "retrieve[PO2SEX]" will return the patient's PO2SEX score. Use different entities to access different data points.

(2) finish[answer], which returns the answer and diagnoses the patient. The action "finish[progressor]" will diagnose a patient as a progressor; while the action "finish[non-progressor]" will diagnose a patient as a non-progressor.

Here are some examples.
    """

    examples = "Thought 1: I should check the patient's V00WOMKP score.\nAction 1: retrieve[V00WOMPK]\nObservation 1: Patient has 2 in column V00WOMPK.\nThought 2: I should check the patient's V00JSW175 score.\nAction 2: retrieve[V00JSW175]\nObservation 2: Patient has 5.67 in column V00JSW175\n.Thought 3: I should check the patient's V00JSW250.\nAction 3: retrieve[V00JSW250]\nObservation 3: Patient has 5.55 in column V00JSW250.\nThought 4: With all this data, we can possibly diagnose this patient as a progressor.\nAction 4: finish[progressor]\nObservation 4: Episode finished"


    prompt = instruction + examples

    infos = []
    patients = env.patientData.sample(n=10)

    for i in range(0, 10):
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
