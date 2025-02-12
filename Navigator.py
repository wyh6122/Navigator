from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from langchain.llms import OpenAI
import os
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from neo4j.exceptions import ServiceUnavailable, SessionExpired, ClientError
import time
import logging
from time import sleep
import re
from typing import List, Tuple
from neo4j import Session
from typing import List, Tuple
from neo4j import GraphDatabase, Session

def extract_outputs(output_all):
    re_output1 = r"Output 1:(.*?)Output 2:"
    re_output2 = r"Output 2:(.*?)Output 3:"
    re_output3 = r"Output 3:(.*)"
    
    output1 = re.search(re_output1, output_all, re.DOTALL)
    output2 = re.search(re_output2, output_all, re.DOTALL)
    output3 = re.search(re_output3, output_all, re.DOTALL)
    
    return (
        output1.group(1).strip() if output1 else "",
        output2.group(1).strip() if output2 else "",
        output3.group(1).strip() if output3 else ""
    )


def chat_35(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content


def chat_4(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
    {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content


def prompt_extract_keyword(input_text):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Extract the relevant keywords from the following medical questions. These keywords include, but are not limited to, disease names, symptom descriptions, medication use, and diagnostic test plans.'\n\n### Input:\n
    <CLS>Doctor, I have been constantly fainting and feeling dizzy along with palpitations, increased heart rate, and weakness. What could be the cause of these symptoms?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been constantly fainting and feeling dizzy along with palpitations, increased heart rate, and weakness. What could be the cause of these symptoms?<SEP>The extracted entities are Fainting, Dizziness, Palpitations, Increased heart rate, Weakness<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Extract the relevant keywords from the following medical questions. These keywords include, but are not limited to, disease names, symptom descriptions, medication use, and diagnostic test plans.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat(chat_prompt_with_values.to_messages()).content

    question_kg = re.findall(re1,response_of_KG)
    return question_kg


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines


def is_unable_to_answer(response):
 
    analysis = openai.Completion.create(
    engine="text-davinci-002",
    prompt=response,
    max_tokens=1,
    temperature=0.0,
    n=1,
    stop=None,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    if not score.isdigit():   
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        return paths,exist_entity


def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease


def process_neighbors(match_kg, get_entity_neighbors_func):
    neighbor_list = []
    neighbor_list_disease = []
    
    for match_entity in match_kg:
        disease_flag = 0
        neighbors, disease = get_entity_neighbors_func(match_entity, disease_flag)
        neighbor_list.extend(neighbors)

        while disease:
            new_disease = [d for d in disease if d in match_kg]

            if new_disease:
                for disease_entity in new_disease:
                    disease_flag = 1
                    neighbors, disease = get_entity_neighbors_func(disease_entity, disease_flag)
                    neighbor_list_disease.extend(neighbors)
            else:
                for disease_entity in disease:
                    disease_flag = 1
                    neighbors, disease = get_entity_neighbors_func(disease_entity, disease_flag)
                    neighbor_list_disease.extend(neighbors)
    
    if len(neighbor_list) <= 5:
        neighbor_list.extend(neighbor_list_disease)
    
    return neighbor_list


def prompt_path_finding(path_input):
    template = """
    There are some knowledge graph path. They follow entity->relationship->entity format.
    \n\n
    {Path}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})

    response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    return response_of_KG_path


def prompt_neighbor(neighbor):
    template = """
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text=' ')

    response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content

    return response_of_KG_neighbor


def final_answer(input_text,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                SystemMessage(content="You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation. "),
                HumanMessage(content="Patient input:"+ input_text),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                HumanMessage(content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output1: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output2: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"                                                                                                                                         
                             + """
Output 1:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output 2:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output 3: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """  
                             )

                                   ]
        
    result = chat(messages)
    output_all = result.content
    return output_all


def final_answer_Navigator(input_text,response_of_KG_list_path,response_of_KG_neighbor):
    messages  = [
                SystemMessage(content="You are an excellent AI doctor, and you can diagnose diseases and recommend tests and medications based on the symptoms in the conversation, the provided medical knowledge, and your own knowledge."),
                HumanMessage(content="Patient input:"+ input_text),
                AIMessage(content="You have some medical knowledge information in the following:\n\n" +  '###' + (response_of_KG_list_path if response_of_KG_list_path is not None else "") + '\n\n' + '###' + (response_of_KG_neighbor if response_of_KG_neighbor is not None else "")),
                HumanMessage(content="What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease? Think step by step.\n\n\n"
                            + "Output_a: The answer includes disease and tests and recommened medications.\n\n"
                             +"Output_b: Show me inference process as a string about extract what knowledge from which Path-based Evidence or Neighor-based Evidence, and in the end infer what result. \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                             +"Output_c: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                             + "There is a sample:\n"                                                                                                                                         
                             + """
Output_a:
Based on the symptoms described, the patient may have laryngitis, which is inflammation of the vocal cords. To confirm the diagnosis, the patient should undergo a physical examination of the throat and possibly a laryngoscopy, which is an examination of the vocal cords using a scope. Recommended medications for laryngitis include anti-inflammatory drugs such as ibuprofen, as well as steroids to reduce inflammation. It is also recommended to rest the voice and avoid smoking and irritants.

Output_b:
Path-based Evidence 1('Patient'->'has been experiencing'->'hoarse voice')->Path-based Evidence 2('hoarse voice'->'could be caused by'->'laryngitis')->Neighbor-based Evidence 1('laryngitis'->'requires'->'physical examination of the throat')->Neighbor-based Evidence 2('physical examination of the throat'->'may include'->'laryngoscopy')->result 1('laryngitis')->Path-based Evidence 3('laryngitis'->'can be treated with'->'anti-inflammatory drugs and steroids')->Neighbor-based Evidence 3('anti-inflammatory drugs and steroids'->'should be accompanied by'->'resting the voice and avoiding irritants').

Output_c: 
Patient(Path-based Evidence 1)
└── has been experiencing(Path-based Evidence 1)
    └── hoarse voice(Path-based Evidence 1)(Path-based Evidence 2)
        └── could be caused by(Path-based Evidence 2)
            └── laryngitis(Path-based Evidence 2)(Neighbor-based Evidence 1)
                ├── requires(Neighbor-based Evidence 1)
                │   └── physical examination of the throat(Neighbor-based Evidence 1)(Neighbor-based Evidence 2)
                │       └── may include(Neighbor-based Evidence 2)
                │           └── laryngoscopy(Neighbor-based Evidence 2)(result 1)(Path-based Evidence 3)
                ├── can be treated with(Path-based Evidence 3)
                │   └── anti-inflammatory drugs and steroids(Path-based Evidence 3)(Neighbor-based Evidence 3)
                └── should be accompanied by(Neighbor-based Evidence 3)
                    └── resting the voice and avoiding irritants(Neighbor-based Evidence 3)
                                    """  
                             )

                                   ]
        
    result = chat(messages)
    output_all_Navigator = result.content
    return output_all_Navigator


def prompt_document(question,instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n
    {question}
    \n\n
    You have some medical knowledge information in the following:
    {instruction}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content

    return response_document_bm25


def Doctor_Agent_check(disease_label,question, evidences):
    system_message = f"You are a professional assistant in the field of {disease_label}, skilled in filtering relevant neighbor-based evidence based on patient descriptions. Please strictly follow the user's instructions to ensure precise and relevant support."

    prompt_template = f"""
    Question:
    {question}

    List of Neighbor-based Evidence:
    {evidences}

    Please retain those neighbor-based evidences that help in diagnosing the disease potentially associated with the patient's symptoms, recommend necessary tests, and suggest required medications.
    Please ensure that while retaining these evidences, you **keep the original content of each evidence intact**, without simplifying or modifying any details.

    Only return the filtered neighbor-based evidence without any additional explanations or comments.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            temperature=0,  
            max_tokens=1000,  
            n=1
        )
        
        filtered_evidences = response['choices'][0]['message']['content'].strip()
        
        if not filtered_evidences:
            filtered_evidences = "no evidence"
            return filtered_evidences
        else:
            return filtered_evidences
        
    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return None


def prompt_identify_medical_fields(question):
    template = """
    You are an efficient medical assistant. Your task is to provide the two most relevant medical fields involved in diagnosing the patient's condition based on the patient's description. Focus on the key symptoms and signs mentioned in the description.

    Patient input:  
    {question}

    Based on the symptoms and signs described in the question, determine the two most relevant medical fields for diagnosis. Avoid irrelevant or overly broad fields. Provide the list of relevant medical fields.

    Output only the list of relevant medical fields:

    """
    prompt = PromptTemplate(
        template = template,
        input_variables = ["question"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question, text={})

    response_medical_fields = chat(chat_prompt_with_values.to_messages()).content

    return response_medical_fields


def MedEye(question_description, match_kg, model="gpt-4"):
    prompt = f"""
    Below is a patient description and a list of entities. Please filter the entities based on the following criteria:
    1. The entity is relevant to the patient description.
    2. The entity helps in making a diagnosis or recommending tests or medications for the patient.

    Problem description:
    {question_description}

    Entity list:
    {', '.join(match_kg)}

    Please list the filtered entities and sort them based on priority: disease-related entities should be placed at the top, followed by symptom-related entities, and then other types of entities (separate by commas):
    """

    try:

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in filtering and analyzing data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.3,
        )
 
        gpt_output = response.choices[0].message['content'].strip()
        
        filtered_entities = [entity.strip() for entity in gpt_output.split(',') if entity.strip() in match_kg]

        return filtered_entities

    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return []



if __name__ == "__main__":
    YOUR_OPENAI_KEY = 'YOUR_OPENAI_KEY'#replace this to your key

    os.environ['OPENAI_API_KEY']= YOUR_OPENAI_KEY
    openai.api_key = YOUR_OPENAI_KEY

    # 1. build neo4j knowledge graph datasets
    uri = "YOUR_URL"
    username = "YOUR_USER"
    password = "YOUR_PASSWORD"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()


    ##############################build KG 

    session.run("MATCH (n) DETACH DELETE n")# clean all

    # read triples
    df = pd.read_csv('./data/chatdoctor5k/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])


    for index, row in df.iterrows():
      head_name = row['head']
      tail_name = row['tail']
      relation_name = row['relation']

      query = (
          "MERGE (h:Entity { name: $head_name }) "
          "MERGE (t:Entity { name: $tail_name }) "
          "MERGE (h)-[r:`" + relation_name + "`]->(t)"
      )
      session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)

# # 2. OpenAI API based keyword extraction and match entities

    OPENAI_API_KEY = YOUR_OPENAI_KEY
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('./output/chatdoctor5k/output.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'Label', 'Navigator','MindMap','GPT3.5','BM25_retrieval','Embedding_retrieval','KG_retrieval','GPT4'])

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/chatdoctor5k/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    docs_dir = './data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)
   
    with open("./data/chatdoctor5k/NER_chatgpt_new.json", "r") as f:
        for index, line in enumerate(f.readlines()[337:338],338):
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n","")
            input = input.replace("<OOS>","<EOS>")
            input = input.replace(":","") + "<END>"
            input_text = re.findall(re3, input)

            if input_text == []:
                continue
            print(f'Question{index}: {input_text[0]}')
            print()  

            disease_label=prompt_identify_medical_fields(input_text[0])


            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
            output_text = re.findall(re3,output)
            print('answer:',output_text[0])
            print()
                 
            question_kg = re.findall(re1,input)
            if len(question_kg) == 0:
                question_kg = re.findall(re2,input)
                if len(question_kg) == 0:
                    print("<Warning> no entities found", input)
                    continue
            question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
            question_kg = question_kg.replace("\n","")
            question_kg = question_kg.split(", ")
            print("question_kg:",question_kg)
            print()

            answer_kg = re.findall(re1,output)
            if len(answer_kg) == 0:  
                answer_kg = re.findall(re2,output)
                if len(answer_kg) == 0:
                    print("<Warning> no entities found", output)
                    continue
            answer_kg = answer_kg[0].replace("<END>","").replace("<EOS>","")
            answer_kg = answer_kg.replace("\n","")
            answer_kg = answer_kg.split(", ")
            
            match_kg = []
            sx_match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
           

            for kg_entity in question_kg:
                
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                max_index = cos_similarities.argmax()
                          
                match_kg_i = entity_embeddings["entities"][max_index]
                while match_kg_i.replace(" ","_") in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i.replace(" ","_"))             
            print('match_kg:',match_kg)
            print()
            sx_match_kg = MedEye(input_text[0], match_kg, model="gpt-4")
            if not sx_match_kg:  
                print(f"<Warning> No matching entities found for Question {index}. Skipping to next.")
                continue

            print('sx_match_kg:', sx_match_kg)
            print()



            # # 4. neo4j knowledge graph path finding
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]
                
                result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                        
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    result_path = combine_lists(*paths_list)
                
                
                    if result_path != []:
                        result_path_list.extend(result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        result_path = {}
                        single_path = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]
                    else:
                        result_path = []
                                                  
                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in result_path_list:
                                if len(result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        single_path = result_path_list[0]
                    except:
                        single_path = result_path_list
                    
            else:
                result_path = {}
                single_path = {}            
            

            # # 4.1  After entity filtering and ranking，neo4j knowledge graph path finding
            if len(sx_match_kg) > 0:
                start_entity = sx_match_kg[0]
                candidate_entity = sx_match_kg[1:]
                
                sx_result_path_list = []
                while 1:
                    flag = 0
                    paths_list = []
                    while candidate_entity != []:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)                        
                        paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                        path_list = []
                        if paths == [''] or paths == []:
                            flag = 1
                            if candidate_entity == []:
                                flag = 0
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            break
                        else:
                            for p in paths:
                                path_list.append(p.split('->'))
                            if path_list != []:
                                paths_list.append(path_list)
                        
                        if exist_entity != {}:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity
                    sx_result_path = combine_lists(*paths_list)
                
                
                    if sx_result_path != []:
                        sx_result_path_list.extend(sx_result_path)                
                    if flag == 1:
                        continue
                    else:
                        break
                    
                start_tmp = []
                for path_new in sx_result_path_list:
                
                    if path_new == []:
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])
                
                if len(start_tmp) == 0:
                        sx_result_path = {}
                        sx_single_path = {}
                else:
                    if len(start_tmp) == 1:
                        sx_result_path = sx_result_path_list[:5]
                    else:
                        sx_result_path = []
                                                  
                        if len(start_tmp) >= 5:
                            for path_new in sx_result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    sx_result_path.append(path_new)
                                    start_tmp.remove(path_new[0])
                                if len(sx_result_path) == 5:
                                    break
                        else:
                            count = 5 // len(start_tmp)
                            remind = 5 % len(start_tmp)
                            count_tmp = 0
                            for path_new in sx_result_path_list:
                                if len(sx_result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in start_tmp:
                                        if count_tmp < count:
                                            sx_result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in start_tmp:
                                                sx_result_path.append(path_new)
                                                count_tmp += 1

                                        if len(start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                    try:
                        sx_single_path = sx_result_path_list[0]
                    except:
                        sx_single_path = sx_result_path_list
                    
            else:
                sx_result_path = {}
                sx_single_path = {}            
            

            # 处理 match_kg  # 5. neo4j knowledge graph neighbor entities
            neighbor_list = process_neighbors(match_kg, get_entity_neighbors)
            
            # 处理 sx_match_kg  #5.1  After entity filtering and ranking，neo4j knowledge graph neighbor entities
            sx_neighbor_list = process_neighbors(sx_match_kg, get_entity_neighbors)
            

            # 6. knowledge gragh path based prompt generation
            if len(match_kg) != 1 or 0:
                response_of_KG_list_path = []
                if result_path == {}:
                    response_of_KG_list_path = []
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)
                        
                    path = "\n".join(result_new_path)
                    print("path:\n",path)
                    print()
                    response_of_KG_list_path = prompt_path_finding(path)
                    if is_unable_to_answer(response_of_KG_list_path):
                        response_of_KG_list_path = prompt_path_finding(path)
                    print("response_of_KG_list_path:\n",response_of_KG_list_path)
                    print()
            else:
                response_of_KG_list_path = '{}'

            response_single_path = prompt_path_finding(single_path)
            if is_unable_to_answer(response_single_path):
                response_single_path = prompt_path_finding(single_path)


            # 6.1 After entity filtering and ranking,knowledge gragh path based prompt generation
            if len(sx_match_kg) != 1 or 0:
                sx_response_of_KG_list_path = []
                if sx_result_path == {}:
                    response_of_KG_list_path = []
                else:
                    sx_result_new_path = []
                    for total_path_i in sx_result_path:
                        path_input = "->".join(total_path_i)
                        sx_result_new_path.append(path_input)
                        
                    new_path = "\n".join(sx_result_new_path)
                    print("new_path:\n",new_path)
                    print()
                    sx_response_of_KG_list_path = prompt_path_finding(new_path)
                    if is_unable_to_answer(sx_response_of_KG_list_path):
                        sx_response_of_KG_list_path = prompt_path_finding(new_path)
                    print("sx_response_of_KG_list_path:\n",sx_response_of_KG_list_path)
                    print()
            else:
                sx_response_of_KG_list_path = '{}'

            sx_response_single_path = prompt_path_finding(sx_single_path)
            if is_unable_to_answer(sx_response_single_path):
                sx_response_single_path = prompt_path_finding(sx_single_path)


            # 7. knowledge gragh neighbor entities based prompt generation   
            response_of_KG_list_neighbor = []
            neighbor_new_list = []
            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)
            # print("neighbor_new_list:\n",neighbor_new_list)
            neighbor_input = ""
            if len(neighbor_new_list) == 0:
               print("No neighbor data available")
               neighbor_input = "No valid data"  
            else:
                if len(neighbor_new_list) > 5:
                    neighbor_input = "\n".join(neighbor_new_list[:5])
                else:
                    neighbor_input = "\n".join(neighbor_new_list)
            print("neighbor_input:\n",neighbor_input)
            print()

            response_of_KG_list_neighbor = prompt_neighbor(neighbor_input)
            while is_unable_to_answer(response_of_KG_list_neighbor):
                print("Unable to answer, retrying...")
                response_of_KG_list_neighbor = prompt_neighbor(neighbor_input)
            print("response_of_KG_list_neighbor:\n", response_of_KG_list_neighbor)
            print()

            ##doctor agent examination
            response_of_KG_neighbor_accurate=Doctor_Agent_check(disease_label, input_text[0], response_of_KG_list_neighbor)
            print("response_of_KG_neighbor_accurate:\n", response_of_KG_neighbor_accurate)
            print()


            # 7.1 MedEye for filtering and ranking+doctor agent examination->knowledge gragh neighbor entities based prompt generation   
            sx_response_of_KG_neighbor = []
            neighbor_new_list = []
            for neighbor_i in sx_neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)

            sx_neighbor_input = ""
            if len(neighbor_new_list) == 0:
               print("No neighbor data available")
               sx_neighbor_input = "No valid data"  
            else:
                if len(neighbor_new_list) > 5:
                    sx_neighbor_input = "\n".join(neighbor_new_list[:5])
                else:
                    sx_neighbor_input = "\n".join(neighbor_new_list)
            print("sx_neighbor_input:\n",sx_neighbor_input)
            print()

            sx_response_of_KG_neighbor = prompt_neighbor(sx_neighbor_input)
            while is_unable_to_answer(sx_response_of_KG_neighbor):
                print("Unable to answer, retrying...")
                sx_response_of_KG_neighbor = prompt_neighbor(sx_neighbor_input)
            print("sx_response_of_KG_neighbor:\n", sx_response_of_KG_neighbor)
            print()
    
            ##doctor agent examination
            sx_response_of_KG_neighbor_accurate=Doctor_Agent_check(disease_label, input_text[0], sx_response_of_KG_neighbor)
            print("sx_response_of_KG_neighbor_accurate:\n", sx_response_of_KG_neighbor_accurate)
            print()

            
            # # 8.1  Navigator:prompt-based medical diaglogue answer generation
            output_all_Navigator = final_answer_Navigator(input_text[0],sx_response_of_KG_list_path,sx_response_of_KG_neighbor_accurate)
            if is_unable_to_answer(output_all_Navigator):
                output_all_Navigator= final_answer_Navigator(input_text[0],sx_response_of_KG_list_path,sx_response_of_KG_neighbor_accurate)

            re4 = r"Output_a:(.*?)Output_b:"
            re5 = r"Output_b:(.*?)Output_c:"

            output_a = re.findall(re4, output_all_Navigator, flags=re.DOTALL)
            if len(output_a) > 0:
                output_a = output_a[0]
            else:
                continue
             
            output_b = re.findall(re5, output_all_Navigator, flags=re.DOTALL)
            if len(output_b) > 0:
                output_b = output_b[0]
            else:
                continue
                
            output_c_index = output_all_Navigator.find("Output_c:")
            if output_c_index != -1:
                output_c = output_all_Navigator[output_c_index + len("Output_c:"):].strip()
            
            output_A =output_a
            print('Navigator:\n',output_A)


            #  # # 8.1.1  Navigator(only-Doctor Agent): prompt-based medical diaglogue answer generation
            # output_all_Navigator_AD = final_answer_Navigator(input_text[0],response_of_KG_list_path,response_of_KG_neighbor_accurate)
            # if is_unable_to_answer(output_all_Navigator_AD):
            #     output_all_Navigator_AD= final_answer_Navigator(input_text[0],response_of_KG_list_path,response_of_KG_neighbor_accurate)

            # re4 = r"Output_a:(.*?)Output_b:"
            # re5 = r"Output_b:(.*?)Output_c:"

            # output_a = re.findall(re4, output_all_Navigator_AD, flags=re.DOTALL)
            # if len(output_a) > 0:
            #     output_a = output_a[0]
            # else:
            #     continue
             
            # output_b = re.findall(re5, output_all_Navigator_AD, flags=re.DOTALL)
            # if len(output_b) > 0:
            #     output_b = output_b[0]
            # else:
            #     continue
                
            # output_c_index = output_all_Navigator_AD.find("Output_c:")
            # if output_c_index != -1:
            #     output_c = output_all_Navigator_AD[output_c_index + len("Output_c:"):].strip()
            
            # print('Navigator(only-DA):\n',output_a)


            # ###8.1.2 Navigator(only-MedEye): prompt-based medical diaglogue answer generation
            # output_all_Navigator_ME = final_answer_Navigator(input_text[0], sx_response_of_KG_list_path, sx_response_of_KG_neighbor)

            # if is_unable_to_answer(output_all_Navigator_ME):
            #     output_all_Navigator_ME = final_answer_Navigator(input_text[0], sx_response_of_KG_list_path, sx_response_of_KG_neighbor)

            # re4_sx = r"Output_a:(.*?)Output_b:"
            # re5_sx = r"Output_b:(.*?)Output_c:"

            # output_a_sx = re.findall(re4_sx, output_all_Navigator_ME, flags=re.DOTALL)
            # if len(output_a_sx) > 0:
            #     output_a_sx = output_a_sx[0]
            # else:
                
            #     continue

            # output_b_sx = re.findall(re5_sx, output_all_Navigator_ME, flags=re.DOTALL)
            # if len(output_b_sx) > 0:
            #     output_b_sx = output_b_sx[0]
            # else:

            #     continue

            # output_c_index_sx = output_all_Navigator_ME.find("Output_c:")
            # if output_c_index_sx != -1:
            #     output_c_sx = output_all_Navigator_ME[output_c_index_sx + len("Output_c:"):].strip()

            # print('Navigator(only-MedEye):\n', output_a_sx)



            # # 9. Experiment 1: MindMap
            output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_list_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text[0],response_of_KG_list_path,response_of_KG_list_neighbor)

            
            re4 = r"Output 1:(.*?)Output 2:"
            re5 = r"Output 2:(.*?)Output 3:"

            output1 = re.findall(re4, output_all, flags=re.DOTALL)
            if len(output1) > 0:
                output1 = output1[0]
            else:
                continue
             
            output2 = re.findall(re5, output_all, flags=re.DOTALL)
            if len(output2) > 0:
                output2 = output2[0]
            else:
                continue
                
            output3_index = output_all.find("Output 3:")
            if output3_index != -1:
                output3 = output_all[output3_index + len("Output 3:"):].strip()
            
            print('MindMap:\n',output1)

         
            ## 10. Experiment 2: chatgpt
            try:
                chatgpt_result = chat_35(str(input_text[0]))
            except:
                sleep(40)
                chatgpt_result = chat_35(str(input_text[0]))
            print('\nGPT-3.5:',chatgpt_result)
            

            ## 11. Experiment 3: document retrieval + bm25
            document_dir = "./data/chatdoctor5k/document"
            document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]
            corpus = []
            for path in document_paths:
                with open(path, "r", encoding="utf-8") as f:
                    corpus.append(f.read().lower().split())
            dictionary = corpora.Dictionary(corpus)
            bm25_model = BM25Okapi(corpus)

            bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))

            query = input_text[0]
            query_tokens = query.lower().split()
            tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
            tfidf_query = tfidf_model[dictionary.doc2bow(query_tokens)]
            best_document_index, best_similarity = 0, 0  

            bm25_scores = bm25_index[tfidf_query]
            for i, score in enumerate(bm25_scores):
                if score > best_similarity:
                    best_similarity = score
                    best_document_index = i

            with open(document_paths[best_document_index], "r", encoding="utf-8") as f:
                best_document_content = f.read()

            document_bm25_result = prompt_document(input_text[0],best_document_content)
            if is_unable_to_answer(document_bm25_result):
                document_bm25_result = prompt_document(input_text[0],best_document_content)
            
            print('\nBM25_retrieval:\n',document_bm25_result)


            ### 12. Experiment 4: document + embedding retrieval
            model = Word2Vec.load("./data/chatdoctor5k/word2vec.model")
            ques_vec = np.mean([model.wv[token] for token in input_text[0].split()], axis=0)
            similarities = []
            for doc in docs:
                doc_vec = np.mean([model.wv[token] for token in doc.split()], axis=0)
                similarity = cosine_similarity([ques_vec], [doc_vec])[0][0]
                similarities.append(similarity)

            max_index = np.argmax(similarities)
            most_similar_doc = docs[max_index]
           
            document_embedding_result = prompt_document(input_text[0],most_similar_doc)
            if is_unable_to_answer(document_embedding_result):
                document_embedding_result = prompt_document(input_text[0],most_similar_doc)
            print('\nEmbedding retrieval:\n',document_embedding_result)


            ### 13. Experiment 5: kg retrieval
            kg_retrieval = prompt_document(input_text[0],response_single_path)
            if is_unable_to_answer(kg_retrieval):
                kg_retrieval = prompt_document(input_text[0],response_single_path)
            print('\nKG_retrieval:\n',kg_retrieval)


            ### 14. Experimet 6: gpt4
            try:
                gpt4_result = chat_4(str(input_text[0]))
            except:
                gpt4_result = chat_4(str(input_text[0]))
            print('\nGPT4:\n',gpt4_result)

            
            ## save the final result
            with open('./output/chatdoctor5k/output.csv', 'a+', newline='',encoding='utf-8') as f6:
                writer = csv.writer(f6)
                writer.writerow([f'{input_text[0]}', f'{output_text[0]}', f'{output_a}', f'{output1}',f'{chatgpt_result}', f'{document_bm25_result}', f'{document_embedding_result}', f'{kg_retrieval}',f'{gpt4_result}'])
                f6.flush()    