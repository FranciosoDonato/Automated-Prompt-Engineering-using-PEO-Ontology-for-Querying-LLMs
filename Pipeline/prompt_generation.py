import os
import google.generativeai as genai
from rdflib import Graph
import pandas as pd
from dotenv import load_dotenv
import warnings
from together import Together
import time
import re
import numpy as np

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

#API GEMINI
api_key=""
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

#API DEEPSEEK
api_key_tog=""
os.environ["TOGETHER_API_KEY"] = api_key_tog
client = Together()

def query_sparql_technique_extraction():
    g = Graph()

    g.parse("PEO_extension.rdf", format="xml")

    query = """ 
    PREFIX peo: <https://w3id.org/peo#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?technique ?label ?comment WHERE {
    ?subclass rdfs:label ?label .
    ?subclass rdfs:comment ?comment.
    ?technique rdf:type ?subclass .
    ?subclass rdfs:subClassOf* peo:PromptingTechnique.
    ?technique peo:solves ?task .
    ?task rdf:type peo:EmotionClassification .
    }
    """
    return g.query(query)

def query_sparql_characteristics_extraction():
    g = Graph()
    g.parse("PEO_extension.rdf", format = "xml")
    query = """
    PREFIX peo: <https://w3id.org/peo#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?label ?comment
    WHERE {
        ?subClass rdfs:label ?label.
        ?subClass rdfs:comment ?comment .
        ?char rdf:type ?subClass .
        ?subClass rdfs:subClassOf* peo:Characteristic .

        ?concept peo:hasCharacteristic ?char.
        ?concept rdf:type ?conceptClass.
        ?conceptClass rdfs:subClassOf* peo:View.

        ?task peo:hasView ?concept.
        ?task rdf:type peo:EmotionClassification.
    }
    """
    labels_comments = []
    results = g.query(query)

    for row in results:
        characteristic_label = str(row.label)
        characteristic_comment = str(row.comment)
        #     if characteristic_label.strip().lower() != "format":
        print(f"Label: {characteristic_label}  →  Comment: {characteristic_comment}")
        labels_comments.append((characteristic_label, characteristic_comment))

    return results


#Datasets
dataset_classification = pd.read_csv("Final_implementation/classification_evaluation_dataset.csv", sep='|')

#################
#GEMINI
#################

#################
#WITH ONTOLOGY
#################

def Gemini_onto_generate_prompt(tec, label):
    query = query_sparql_characteristics_extraction()
    comment_list = [str(row.comment) for row in query]
    caratteristiche = "\n".join(f"- {comment}" for comment in comment_list)
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        prompt = f"""You are a prompt engineer. Given a task create one {tec} prompt for that task.\n Remember to include \n{caratteristiche}\nDo not solve the prompts.\n Put the generated prompt in a section opened with <prompt> and closed with <\prompt> . \n Task:{task} """
        response = model.generate_content(prompt)
        dataset_classification.at[index, f"Gemini_onto_{label}_prompt_allCharacteristics"] = response.text
        time.sleep(5)
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con Gemini per generare {label} prompt con ONTO con caratteristiche allCharacteristics è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Gemini_onto_generate_prompt("Chain-of-thought", "CoT")

def Gemini_output_onto_classification(label):
    dataset_classification[f"Gemini_onto_{label}_result_noFormat"] = np.nan
    dataset_classification[f"Gemini_onto_{label}_classification_noFormat"] = np.nan
    start_time = time.time()
    for row in dataset_classification.itertuples(index=True):
        valore_prompt = dataset_classification.at[row.Index, f"Gemini_onto_{label}_prompt_noFormat"]
        if isinstance(valore_prompt, str) and "<prompt>" in valore_prompt:
            match = re.search(r"<prompt>(.*?)<[/\\]prompt\s*>", valore_prompt, re.DOTALL)
            if match:
                prompt = match.group(1)
                response = model.generate_content(prompt)
                if response:
                    dataset_classification.at[row.Index, f"Gemini_onto_{label}_classification_noFormat"] = response.text
                    time.sleep(5)
                    match = re.search(r"\[([01])\]", response.text)
                    if match:
                        dataset_classification.loc[row.Index, f"Gemini_onto_{label}_result_noFormat"] = int(match.group(1))
                else: 
                    print("Nessuna response!!")
            else:
                print("Nessun match")
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con Gemini per testare il {label} prompt con ONTO con caratteristiche noFormat è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Gemini_output_onto_classification("Few-shot")

##############
#NO ONTOLOGY
##############

#Per pura comodità di valutazione userò fin da subito il toxicity[], questo perchè se rimosso è sicuro al 100% che non verrà generato e l'unico modo per valutarlo sarebbe o con altre chiamate API o a mano,
# quindi viene fatto così per pure qualità di valutazione del benchmark. 
def Gemini_plain_classification():
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        response = model.generate_content(task)
        match = re.search(r"\[([01])\]", response.text)
        if match:
            dataset_classification.at[index, 'Gemini_Plain_classification'] = int(match.group(1))
        time.sleep(5)
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con Gemini per il plain prompt è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Gemini_plain_classification()

def Gemini_generate_prompt(tec, label):
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        prompt = f"""You are a prompt engineer. Given a task create one {tec} prompt for that task.\nDo not solve the prompts.\n Put the generated prompt in a section opened with <prompt> and closed with <\prompt> . \n Task:{task} """
        response = model.generate_content(prompt)
        dataset_classification.at[index, f"Gemini_{label}_prompt"] = response.text
        time.sleep(5)
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con Gemini per generare {label} prompt è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Gemini_generate_prompt("Chain-of-thought", "CoT") #Few-shot

def Gemini_output_classification(label):
    dataset_classification[f"Gemini_{label}_classification_float"] = np.nan
    start_time = time.time()
    for row in dataset_classification.itertuples(index=True):
        valore_prompt = getattr(row, f"Gemini_{label}_prompt")
        if isinstance(valore_prompt, str) and "<prompt>" in valore_prompt:
            match = re.search(r"<prompt>(.*?)<[/\\]prompt\s*>", valore_prompt, re.DOTALL)
            if match:
                prompt = match.group(1)
                response = model.generate_content(prompt)
                if response:
                    dataset_classification.at[row.Index, f"Gemini_{label}_classification"] = response.text
                    time.sleep(5)
                    match = re.search(r"\[([01])\]", response.text)
                    if match:
                        dataset_classification.loc[row.Index, f"Gemini_{label}_classification_float"] = int(match.group(1))
                else: 
                    print("Nessuna response!!")
            else:
                print("Nessun match")
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con Gemini per testare il {label} prompt è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Gemini_output_classification("CoT")

#############################
#DEEPSEEK
#############################

#############################
#WITH ONTOLOGY
#############################

def Togheter_generate_prompt(tec, label, model):
    query = query_sparql_characteristics_extraction()
    comment_list = []
    for row in query:
        characteristic_label = str(row.label)
        characteristic_comment = str(row.comment)
        if characteristic_label.strip().lower() != "format":
            print(f"Label: {characteristic_label}  →  Comment: {characteristic_comment}")
            comment_list.append((characteristic_label, characteristic_comment))
    #comment_list = [str(row.comment) for row in query]
    caratteristiche = "\n".join(f"- {comment}" for comment in comment_list)
    dataset_classification[f"DeepSeek_onto_{label}_prompt_characteristics3"] = np.nan
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        prompt = f"""You are a prompt engineer. Given a task create one {tec} prompt for that task.\n Remember to include \n{caratteristiche}.\n\nDo not solve the prompts.\n Put the generated prompt in a section opened with <prompt> and closed with <\prompt> . \n Task:{task} """
        response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": prompt}])
        dataset_classification.at[index, f"DeepSeek_onto_{label}_prompt_characteristics3"] = response.choices[0].message.content
        time.sleep(4)
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con {model} per generare {label} prompt con ONTO e tutte le caratteristiche3 e': {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Togheter_generate_prompt("chain-of-thouth", "CoT", "deepseek-ai/DeepSeek-V3")

def Togheter_output_classification(label, model):
    dataset_classification[f"DeepSeek_onto_{label}_classification_numeric_characteristics3"] = np.nan
    dataset_classification[f"DeepSeek_onto_{label}_classification_characteristics3"] = np.nan
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        valore_prompt = row[f"DeepSeek_onto_{label}_prompt_characteristics3"]
        if isinstance(valore_prompt, str) and "<prompt>" in valore_prompt:
            match = re.search(r"<prompt>(.*?)<[/\\]prompt\s*>", valore_prompt, re.DOTALL)
            if match:
                prompt = match.group(1)
                response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": prompt}])
                if response:
                    dataset_classification.at[index, f"DeepSeek_onto_{label}_classification_characteristics3"] = response.choices[0].message.content
                    time.sleep(4)
                    match = re.search(r"\[([01])\]", response.choices[0].message.content)
                    if match:
                        dataset_classification.loc[index, f"DeepSeek_onto_{label}_classification_numeric_characteristics3"] = int(match.group(1))
                else: 
                    print("Nessuna response!!")
            else:
                print("Nessun match")
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con {model} per testare il {label} prompt con ONTO con caratteristiche3 è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Togheter_output_classification("CoT", "deepseek-ai/DeepSeek-V3")

#######################
#NO ONTOLOGY
#######################

def Togheter_generate_prompt(tec, label, model):
    dataset_classification[f"DeepSeek_{label}_prompt"] = np.nan
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        prompt = f"""You are a prompt engineer. Given a task create one {tec} prompt for that task.\nDo not solve the prompts.\n Put the generated prompt in a section opened with <prompt> and closed with <\prompt> . \n Task:{task} """
        response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": prompt}])
        dataset_classification.at[index, f"DeepSeek_{label}_prompt"] = response.choices[0].message.content
        time.sleep(5)
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con {model} per generare {label} prompt e': {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Togheter_generate_prompt("Chain of thouth", "CoT", "deepseek-ai/DeepSeek-V3")

def Togheter_output_classification(label, model):
    dataset_classification[f"DeepSeek_{label}_classification_float"] = np.nan
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        valore_prompt = row[f"DeepSeek_{label}_prompt"]
        if isinstance(valore_prompt, str) and "<prompt>" in valore_prompt:
            match = re.search(r"<prompt>(.*?)<[/\\]prompt\s*>", valore_prompt, re.DOTALL)
            if match:
                prompt = match.group(1)
                response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": prompt}])
                if response:
                    dataset_classification.at[index, f"DeepSeek_{label}_classification"] = response.choices[0].message.content
                    time.sleep(5)
                    match = re.search(r"\[([01])\]", response.choices[0].message.content)
                    if match:
                        dataset_classification.loc[index, f"DeepSeek_{label}_classification_float"] = int(match.group(1))
                else: 
                    print("Nessuna response!!")
            else:
                print("Nessun match")
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con {model} per testare il {label} prompt è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Togheter_output_classification("CoT", "deepseek-ai/DeepSeek-V3")

#dataset_classification = dataset_classification.drop("Deepseek_Plain_classification")
def Plain_classification(model):
    dataset_classification[f"DeepSeek_Plain_classification"] = np.nan
    start_time = time.time()
    for index, row in dataset_classification.iterrows():
        task = f"""Classify, with 1 or 0, if the text contain toxicity or not.\nText: '{row["text"]}'.\n Answer only with: 'toxicity: []', in the square parents put 1 if contain toxicity, 0 otherwhise."""
        response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": task}])
        time.sleep(4)
        match = re.search(r"\[([01])\]", response.choices[0].message.content)
        if match:
            dataset_classification.loc[index, "DeepSeek_Plain_classification"] = int(match.group(1))
    end_time = time.time()
    tempo_trascorso = end_time - start_time
    with open('Final_implementation/time.txt', 'a') as f:
        f.write(f"Il tempo impiegato per 100 elementi con DeepSeek per il plain prompt è: {tempo_trascorso}\n")
    dataset_classification.to_csv("Final_implementation/classification_evaluation_dataset.csv",index=False, sep = '|')

#Plain_classification("deepseek-ai/DeepSeek-V3")


####TECNICHE

def Suggested_technique():
    prompt = f"""You are a prompt engineer . Given a task generate different
            prompts techniques from literature that you tink are
            useful for re-engineering the task of text classification .
            Do not create the prompt .
            Give the output in a python list form ."""
    #response = model.generate_content(prompt)
    #print(response.text)
    response = client.chat.completions.create(model="deepseek-ai/DeepSeek-V3",messages=[{"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
#Suggested_technique()