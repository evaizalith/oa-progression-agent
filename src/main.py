import os
from os import path
import openai
import pandas as pd
from xrayEnv import xrayEnv

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt, temperature=0, max_token=100, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=stop)
    return response["choices"][0]["text"]

def think(env, idx, prompt, pat_id):
    question = env.reset()

    print(idx, question)

    prompt += question
    prompt += "\n"
    n_calls = 0
    n_badcalls = 0

    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought: {i}:", stop=[f"\nObservation {i}:"])

        try:
            thought, action = thought_action.strip().split("f\nAction{i}")
        except:
            print("Bad: ", thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()

        obs, done, info = env.step(env, pat_id, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction{i}: {action}\nObservation {i}: obs\n"
        prompt += step_str

        if done:
            break

    if not done:
        obs, done, info = step(env, "finish[]")

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
    openai.api_key = os.environ["OPENAI_API_KEY"]

    if openai.api_key is None:
        print("Error: OpenAI API key not found!")
        exit()

    env = xrayEnv()
    env, n_rows = prepareEnv("data/xray_data.csv", env)

    instruction = """
    Diagnose a patient with either a progressor or a non-progressor utilizing interleaving Thought, Action, and Observation steps. Thought can reason about the current situation, and Action can be two types:
(1) retrive[entity], which searches the patient data and returns the requested information. Valid entities include V00WOMKP, V00DIL(R)KN14, P01BP30, V00MOSFLP, V00MMTMB, V00MBMSTMC, P01KSX, P01L(R)XRKOA2, V00AMTPD, V0NSAIDS.
(2) finish[answer], which returns the answer and diagnoses the patient.
    """

    infos = []
    patients = env.patientData.sample(n=10)

    for i in range(0, 10):
        patient = patients.iloc[i]
        pat_id = patient["ID"]
        info = think(env, i, instruction, pat_id)

if __name__ == "__main__":
    main()
