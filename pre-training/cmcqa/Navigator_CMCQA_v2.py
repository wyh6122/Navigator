# This is a Chinese version of MindMap.
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
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import time
import logging
from time import sleep
from neo4j.exceptions import ServiceUnavailable


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


def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "OPTIONAL MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        
        paths = set()
        short_path = 0
        
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            if path is not None:
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
                entities[i] = entities[i]
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i]
                    path_str += "->" + relations[i] + "->"

            if short_path == 1:
                paths = {path_str}
                break
            else:
                paths.add(path_str) 

        if short_path == 0:
            exist_entity = {}

        if len(paths) > 5: 
            paths = sorted(paths, key=len)[:4]

        return list(paths), exist_entity


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


def get_entity_neighbors(entity_name):
   
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
        
     
        neighbors = record["neighbor_entities"]
        
        
        neighbor_list.append([entity_name, rel_type, 
                            ','.join([x for x in neighbors])
                            ])

    return neighbor_list


def prompt_path_finding(path_input):
    template = """
    以下是一些知识图谱路径，遵循“实体->关系->实体”的格式。
    \n\n
    {Path}
    \n\n
    使用以上知识图谱路径知识，分别将它们翻译为自然语言总结描述。用单引号标注实体名和关系名。并将它们命名为路径证据1, 路径证据2....\n输出尽量精简，减少token数，但不能丢失事实。\n\n

    输出:
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
    system_message = f"你是一名专业的助手，擅长根据知识图谱路径帮助生成自然语言描述。请严格按照用户的指示执行。"
    
    prompt_template = f"""
    以下是一些知识图谱路径，遵循“实体->关系->实体”的格式：
    \n\n
    {neighbor}
    \n\n
    请根据以上知识图谱路径，将它们翻译为自然语言描述。每条描述要用单引号标注实体名和关系名，描述格式为：“实体1（关系）实体2”。 并依次命名为“邻居证据1”，“邻居证据2”... \n\n

    输出:
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
        
        response_of_KG_neighbor = response['choices'][0]['message']['content'].strip()
        
        return response_of_KG_neighbor

    except openai.error.OpenAIError as e:
        print(f"OpenAI API 错误: {e}")
        return None


def prompt_neighbor_with_retry(neighbor, max_retries=3):
    retries = 0
    while retries < max_retries:
        response_of_KG_neighbor = prompt_neighbor(neighbor)
        if response_of_KG_neighbor is not None:
            return response_of_KG_neighbor
        retries += 1
        print(f"Retrying... Attempt {retries}")
    return "无法获得有效回应，已重试多次"


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim

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
    
def prompt_comparation(reference,output1,output2):
    template = """
    Reference: {reference}
    \n\n
    output1: {output1}
    \n\n
    output2: {output2}
    \n\n
    According to the reference output, which output is better. If the answer is output1, output '1'. If the answer is output2, output '2'.
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["reference","output1","output2"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(reference = reference,
                                 output1 = output1,
                                 output2 = output2)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(reference = reference,\
                                                        output1 = output1,\
                                                        output2 = output2,\
                                                        text={})

    response_of_comparation = chat(chat_prompt_with_values.to_messages()).content

    return response_of_comparation

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

def final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor):
    if isinstance(response_of_KG_list_path, list):
        response_of_KG_list_path = "\n".join(map(str, response_of_KG_list_path))
    else:
        response_of_KG_list_path = str(response_of_KG_list_path)

    if isinstance(response_of_KG_neighbor, list):
        response_of_KG_neighbor = "\n".join(map(str, response_of_KG_neighbor))
    else:
        response_of_KG_neighbor = str(response_of_KG_neighbor)

    messages  = [
                SystemMessage(content="你是一名优秀的AI医生,你可以根据对话中的症状诊断疾病并推荐药物和提出治疗与检查方案。"),
                HumanMessage(content="患者输入:"+ input_text),
                AIMessage(content="你拥有以下医学证据知识:\n\n"+ "### " + response_of_KG_list_path + "\n\n"+ "### " + response_of_KG_neighbor),
                HumanMessage(content="参考提供的路径证据和邻居证据知识，根据患者输入的症状描述，请问患者患有什么疾病?确认疾病需要什么检查来诊断?推荐的治疗疾病的药物和食物是什么?忌吃什么?一步步思考。\n\n\n"
                            + "输出1：回答应包括疾病和检查已经推荐的药物和食物。\n\n"
                             +"输出2：展示推理过程，即从哪个路径证据或邻居证据中提取什么知识，最终推断出什么结果。 \n 将推理过程转化为以下格式:\n 路径证据标号('实体名'->'关系名'->...)->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->... \n\n"
                             +"输出3：画一个决策树。在输出2的的推理过程中，单引号中的实体或关系与用括号包围的证据来源一起作为一个节点。\n\n"
                             + "以下是一个样例，参考其中的格式:\n"
                             + """
输出1：
根据所描述的症状，患者可能患有喉炎，这是声带的炎症。为了确认诊断，患者应该接受喉咙的身体检查，可能还需要喉镜检查，这是一种使用镜检查声带的检查。治疗喉炎的推荐药物包括抗炎药物，如布洛芬，以及减少炎症的类固醇。还建议让声音休息，避免吸烟和刺激物。

输出2：
路径证据1(“患者”->“症状”->“声音沙哑”)->路径证据2(“声音沙哑”->“可能疾病”->“喉炎”)->邻居证据1(“喉咙体检”->“可能包括”->“喉镜检查”)->邻居证据2(“喉咙体检”->“可能疾病”->“喉炎”)->路径证据3(“喉炎”->“推荐药物”->“消炎药和类固醇”)-邻居证据3(“消炎药和类固醇”->“注意事项”->“休息声音和避免刺激”)。

输出3:：
患者(路径证据1)
└── 症状(路径证据1)
    └── 声音沙哑(路径证据1)(路径证据2)
        └── 可能疾病(路径证据2)
            └── 喉炎(路径证据2)(邻居证据1)
                ├── 需要(邻居证据1)
                │   └── 喉咙体检(邻居证据1)(邻居证据2)
                │       └── 可能包括(邻居证据2)
                │           └── 喉炎(邻居证据2)(结果1)(路径证据3)
                ├── 推荐药物(路径证据3)
                │   └── 消炎药和类固醇(路径证据3)(结果2)(邻居证据3)
                └── 注意事项(邻居证据3)
                    └── 休息声音和避免刺激(邻居证据3)
                                    \n\n\n"""
                            + "参考以上样例的格式得到针对患者输入的输出。\n并分别命名为“输出1”，“输出2”，”输出3“。"
                             )

                                   ]
        
    result = chat(messages)
    output_all = result.content
    return output_all

def final_answer_Navigator(input_text, response_of_KG_list_path, response_of_KG_neighbor):
    if isinstance(response_of_KG_list_path, list):
        response_of_KG_list_path = "\n".join(map(str, response_of_KG_list_path))
    else:
        response_of_KG_list_path = str(response_of_KG_list_path)

    if isinstance(response_of_KG_neighbor, list):
        response_of_KG_neighbor = "\n".join(map(str, response_of_KG_neighbor))
    else:
        response_of_KG_neighbor = str(response_of_KG_neighbor)

    messages = [
        SystemMessage(content="你是一名优秀的AI医生，能够根据对话中的症状、提供的医疗知识以及你自身的知识来诊断疾病并推荐药物和提出治疗与检查方案。"),
        HumanMessage(content="患者输入:" + input_text),
        AIMessage(content="你拥有以下医学证据知识:\n\n"
                   + "### " + response_of_KG_list_path + "\n\n"
                   + "### " + response_of_KG_neighbor),
        HumanMessage(content="参考提供的路径证据和邻居证据知识，结合自身的知识，根据患者输入的症状描述，请问患者患有什么疾病?确认疾病需要什么检查来诊断?推荐的治疗疾病的药物和食物是什么?忌吃什么?一步步思考。\n\n\n"
                            + "输出a：回答应包括疾病和检查已经推荐的药物和食物。\n\n"
                            + "输出b：展示推理过程，即从哪个路径证据或邻居证据中提取什么知识，最终推断出什么结果。 \n 将推理过程转化为以下格式:\n"
                            + "路径证据标号('实体名'->'关系名'->...)->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->路径证据标号('实体名'->'关系名'->...)->邻居证据标号('实体名'->'关系名'->...)->结果标号('实体名')->...\n\n"
                            + "输出c：画一个决策树。在输出2的的推理过程中，单引号中的实体或关系与用括号包围的证据来源一起作为一个节点。\n\n"
                            + "以下是一个样例，参考其中的格式:\n"
                            + """
输出1：
根据所描述的症状，患者可能患有喉炎，这是声带的炎症。为了确认诊断，患者应该接受喉咙的身体检查，可能还需要喉镜检查，这是一种使用镜检查声带的检查。治疗喉炎的推荐药物包括抗炎药物，如布洛芬，以及减少炎症的类固醇。还建议让声音休息，避免吸烟和刺激物。

输出2：
路径证据1(“患者”->“症状”->“声音沙哑”)->路径证据2(“声音沙哑”->“可能疾病”->“喉炎”)->邻居证据1(“喉咙体检”->“可能包括”->“喉镜检查”)->邻居证据2(“喉咙体检”->“可能疾病”->“喉炎”)->路径证据3(“喉炎”->“推荐药物”->“消炎药和类固醇”)-邻居证据3(“消炎药和类固醇”->“注意事项”->“休息声音和避免刺激”)。

输出3：
患者(路径证据1)
└── 症状(路径证据1)
    └── 声音沙哑(路径证据1)(路径证据2)
        └── 可能疾病(路径证据2)
            └── 喉炎(路径证据2)(邻居证据1)
                ├── 需要(邻居证据1)
                │   └── 喉咙体检(邻居证据1)(邻居证据2)
                │       └── 可能包括(邻居证据2)
                │           └── 喉炎(邻居证据2)(结果1)(路径证据3)
                ├── 推荐药物(路径证据3)
                │   └── 消炎药和类固醇(路径证据3)(结果2)(邻居证据3)
                └── 注意事项(邻居证据3)
                    └── 休息声音和避免刺激(邻居证据3)
                                    \n\n\n"""
                            + "参考以上样例的格式得到针对患者输入的输出。\n并分别命名为“输出1”，“输出2”，“输出3”。"
                            )
    ]
    
    result = chat(messages)
    output_all_my = result.content
    return output_all_my


def prompt_document(question,instruction):
    template = """
    你是一个优秀的AI医生，你可以根据对话中的症状诊断疾病并推荐药物。\n\n
    患者输入:\n
    {question}
    \n\n
    以下是您的一些医学知识信息:
    {instruction}
    \n\n
    病人得了什么病?患者需要做哪些检查来确诊?推荐哪些药物可以治愈这种疾病?
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


def comparation_score(response_of_comparation,compare_name):
    count = 0
    countTrue = 0
    if response_of_comparation == '1' or 'output1':
        response_of_comparation = 'MindMap'
        count += 1
        countTrue += 1
    else:
        response_of_comparation = compare_name
        count += 1
    score = countTrue/count
    return response_of_comparation


def prompt_identify_medical_fields(question):
    template = """
    你是一位有用的助手。你的任务是根据患者的描述，提供患者病情涉及的最相关的两个医学领域。
     \n\n
    患者输入:\n
    {question}
    \n\n
    根据问题中描述的症状，确定最相关医学领域。请提供相关医学领域的列表。
    \n\n
    只输出相关医学领域的列表：
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    
    llm_chain = LLMChain(prompt=prompt, llm=chat_model)

    response_medical_fields = llm_chain.run(question)

    return response_medical_fields


def Doctor_Agent_check(disease_label, question, evidences):
    
    system_message = f"你是一名专业的{disease_label}领域助手，擅长根据患者描述筛选相关医学证据。请严格按照用户的指示执行。"
    
    prompt_template = f"""
    患者描述：
    {question}
    
    邻居证据列表：
    {evidences}
    
    请保留以下内容的相关邻居证据：
    1.诊断患者可能患有的疾病
    2.患者可能患有的疾病所需的检查或测试
    3.对于患者描述，推荐的治疗药物和食物
    4.对于患者描述，忌吃的食物

    
    保持的原始格式。只返回筛选后的邻居证据，不要包含任何额外的解释或评论。
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
        
        return filtered_evidences

    except openai.error.OpenAIError as e:
        print(f"OpenAI API 错误: {e}")
        return None


def MedEye(question_description, match_kg, model="gpt-4"):
    prompt = f"""
    下面是一个患者描述和一组实体列表。请根据以下标准筛选实体： 
    1. 实体与患者描述相关。
    2. 实体有助于对患者描述做出诊断或建议、推荐。

    问题描述： 
    {question_description}

    实体列表： 
    {', '.join(match_kg)}

    请列出筛选后的实体，并根据优先级进行排序，症状和疾病类型的实体排在前面（用逗号分隔）： 
    """

    try:

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助医学的助手，擅长筛选和分析数据。"},
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
    df = pd.read_csv('./data/kg_triples_small.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])


    for index, row in df.iterrows():
        head_name = row['head']
        tail_name = row['tail']
        relation_name = row['relation']
       
        try:
            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:`" + relation_name + "`]->(t)"
            )
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)

        except:
            continue
        
# # 2. OpenAI API based keyword extraction and match entities

    OPENAI_API_KEY = YOUR_OPENAI_KEY
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)


    with open('./output/cmcqa/output.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'Label', 'Navigator','MindMap','GPT3.5','BM25_retrieval','Embedding_retrieval','KG_retrieval','GPT4'])

    with open('./data/CMCQA/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
        
    with open('./data/CMCQA/keyword_embeddings.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)


    docs_dir = './data/CMCQA/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)

    with open("./data/CMCQA/Chinese_doctor_qa_ner_NEW.json", "r") as f:
        for index, line in enumerate(f.readlines()[:240], ): 
            flag_openai = 1
            count = 0
            x = json.loads(line)
            input_text = x["question"]
                        
            if input_text == []:
                continue

            print(f'Question:{index}: {input_text}') 
            print()

            #根据问题描述，确定最相关医学领域
            disease_label=prompt_identify_medical_fields(input_text)
            
            output_text = x["answer"]
            print('answer:\n',output_text)
            print()
  
            question_kg = x["question_kg"]
            question_kg = question_kg.split(",")
            if len(question_kg) == 0:
                print("<Warning> no entities found", input)
                continue
            print("question_kg:",question_kg)
            print()

            match_kg = []
            entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
        

            for kg_entity in question_kg:
            
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]

                max_index = cos_similarities.argmax()
                if cos_similarities.argmax() < 0.5:
                    continue
                        
                match_kg_i = entity_embeddings["entities"][max_index]

                while match_kg_i in match_kg:
                    cos_similarities[max_index] = 0
                    max_index = cos_similarities.argmax()
                    match_kg_i = entity_embeddings["entities"][max_index]

                match_kg.append(match_kg_i)
            print('match_kg:',match_kg)
            print()
            sx_match_kg=MedEye(input_text, match_kg, model="gpt-4")
            if not sx_match_kg:  
                print(f"<Warning> No matching entities found for Question {index}. Skipping to next.")
                continue

            print('sx_match_kg:',sx_match_kg)
            print()


            # 4. neo4j knowledge graph path finding 
            if len(match_kg) != 1 or 0:
                start_entity = match_kg[0]
                candidate_entity = match_kg[1:]

                result_path_list = []
                while True:
                    flag = 0
                    paths_list = []
                    while candidate_entity:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
                        path_list = []

                        if not paths:  
                            flag = 1
                            if not candidate_entity:  
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            continue
                        else:
                            for p in paths:
                                if p:  
                                    path_list.append(p.split('->'))
                            if path_list:
                                paths_list.append(path_list)
                        
                        if exist_entity:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity

                    result_path = combine_lists(*paths_list)  
                    
                    if result_path:
                        result_path_list.extend(result_path)

                    if flag == 1:
                        continue
                    else:
                        break

                result_path_list = list(set(tuple(path) for path in result_path_list))

                start_tmp = []
                for path_new in result_path_list:
                    if not path_new: 
                        continue
                    if path_new[0] not in start_tmp:
                        start_tmp.append(path_new[0])

                if not start_tmp:
                    result_path = {}
                    kg_retrieval = {}
                else:
                    if len(start_tmp) == 1:
                        result_path = result_path_list[:5]  
                    else:
                        result_path = []

                        if len(start_tmp) >= 5:
                            for path_new in result_path_list:
                                if not path_new:  
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
                                    if not path_new:
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
                    kg_retrieval = result_path_list[0]
                except:
                    kg_retrieval = result_path_list

            else:
                result_path = {}
                kg_retrieval = {}


        # # 4.1  After entity filtering and ranking+neo4j knowledge graph path finding 
            if len(sx_match_kg) != 1 or 0:
                start_entity = sx_match_kg[0]
                candidate_entity = sx_match_kg[1:]

                sx_result_path_list = []
                while True:
                    flag = 0
                    sx_paths_list = []
                    while candidate_entity:
                        end_entity = candidate_entity[0]
                        candidate_entity.remove(end_entity)
                        paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
                        path_list = []

                        if not paths:  
                            flag = 1
                            if not candidate_entity:  
                                break
                            start_entity = candidate_entity[0]
                            candidate_entity.remove(start_entity)
                            continue
                        else:
                            for p in paths:
                                if p:  
                                    path_list.append(p.split('->'))
                            if path_list:
                                sx_paths_list.append(path_list)

                        if exist_entity:
                            try:
                                candidate_entity.remove(exist_entity)
                            except:
                                continue
                        start_entity = end_entity

                    if not sx_paths_list:
                        sx_result_path = {}
                        sx_kg_retrieval = {}

                        break  
                    else:

                        sx_result_path = combine_lists(*sx_paths_list) 

                        if sx_result_path:
                            sx_result_path_list.extend(sx_result_path)

                        if flag == 1:
                            continue
                        else:
                            break

                sx_result_path_list = list(set(tuple(path) for path in sx_result_path_list))

                sx_start_tmp = []
                for path_new in sx_result_path_list:
                    if path_new == []:
                        continue
                    if path_new[0] not in sx_start_tmp:
                        sx_start_tmp.append(path_new[0])

                if not sx_start_tmp:
                    sx_result_path = {}
                    sx_kg_retrieval = {}
                else:
                    if len(sx_start_tmp) == 1:
                        sx_result_path = sx_result_path_list[:5] 
                    else:
                        sx_result_path = []

                        if len(sx_start_tmp) >= 5:
                            for path_new in sx_result_path_list:
                                if path_new == []:
                                    continue
                                if path_new[0] in sx_start_tmp:
                                    sx_result_path.append(path_new)
                                    sx_start_tmp.remove(path_new[0])
                                if len(sx_result_path) == 5:
                                    break
                        else:
                            count = 5 // len(sx_start_tmp)
                            remind = 5 % len(sx_start_tmp)
                            count_tmp = 0
                            for path_new in sx_result_path_list:
                                if len(sx_result_path) < 5:
                                    if path_new == []:
                                        continue
                                    if path_new[0] in sx_start_tmp:
                                        if count_tmp < count:
                                            sx_result_path.append(path_new)
                                            count_tmp += 1
                                        else:
                                            sx_start_tmp.remove(path_new[0])
                                            count_tmp = 0
                                            if path_new[0] in sx_start_tmp:
                                                sx_result_path.append(path_new)
                                                count_tmp += 1

                                        if len(sx_start_tmp) == 1:
                                            count = count + remind
                                else:
                                    break

                try:
                    sx_kg_retrieval = sx_result_path_list[0]
                except:
                    sx_kg_retrieval = sx_result_path_list

            else:
                sx_result_path = {}
                sx_kg_retrieval = {}


        # # # 5. neo4j knowledge graph neighbor entities
            neighbor_list = []
        
            for match_entity in match_kg:
                disease_flag = 0
                neighbors = get_entity_neighbors(match_entity)
                neighbor_list.extend(neighbors)

            if result_path != {}:
                if len(neighbor_list) > 5:
                    new_neighbor = []
                    for neighbor_new in neighbor_list:
                        if "疾病" not in neighbor_new[1] and "症状" not in neighbor_new[1]:# 更改图谱后修改这里
                            new_neighbor.append(neighbor_new)

                    neighbor_list = new_neighbor

                if len(neighbor_list) > 5:
                    neighbor_list_tmp = []
                    for neighbor in neighbor_list:
                        if neighbor[1] == '常用药品':
                            neighbor_list_tmp.append(neighbor)
                        if len(neighbor_list_tmp) >= 5:
                            break
                    if len(neighbor_list_tmp) < 5:
                        for neighbor in neighbor_list:
                            if neighbor[1] == '诊断检查':
                                neighbor_list_tmp.append(neighbor)
                            if len(neighbor_list_tmp) >= 5:
                                break

                    if len(neighbor_list_tmp) < 5:
                        for neighbor in neighbor_list:
                            neighbor_list_tmp.append(neighbor)
                            if len(neighbor_list_tmp) >= 5:
                                break
                        

                    neighbor_list = neighbor_list_tmp

            # print("neighbor_list",neighbor_list)
            # print()


    # # # 5.1 After entity filtering and ranking，neo4j knowledge graph neighbor entities

            sx_neighbor_list = []

            for sx_match_entity in sx_match_kg:
                disease_flag = 0
                sx_neighbors = get_entity_neighbors(sx_match_entity)
                

                for sx_neighbor in sx_neighbors:

                    if sx_neighbor not in sx_neighbor_list:
                        sx_neighbor_list.append(sx_neighbor)

            if sx_result_path != {}: 
                if len(sx_neighbor_list) > 5:
                    new_sx_neighbor = []
                    for sx_neighbor_new in sx_neighbor_list:
                        if "疾病" not in sx_neighbor_new[1] and "症状" not in sx_neighbor_new[1]:  
                            new_sx_neighbor.append(sx_neighbor_new)

                    sx_neighbor_list = new_sx_neighbor

                if len(sx_neighbor_list) > 5:
                    sx_neighbor_list_tmp = []
                    for sx_neighbor in sx_neighbor_list:
                        if sx_neighbor[1] == '常用药品':
                            sx_neighbor_list_tmp.append(sx_neighbor)
                        if len(sx_neighbor_list_tmp) >= 5:
                            break
                    if len(sx_neighbor_list_tmp) < 5:
                        for sx_neighbor in sx_neighbor_list:
                            if sx_neighbor[1] == '诊断检查':
                                sx_neighbor_list_tmp.append(sx_neighbor)
                            if len(sx_neighbor_list_tmp) >= 5:
                                break

                    if len(sx_neighbor_list_tmp) < 5:
                        for sx_neighbor in sx_neighbor_list:
                            sx_neighbor_list_tmp.append(sx_neighbor)
                            if len(sx_neighbor_list_tmp) >= 5:
                                break

                    sx_neighbor_list = sx_neighbor_list_tmp



    #         # 6. knowledge gragh path based prompt generation

            if len(match_kg) != 1 and len(match_kg) != 0 and result_path != {}:
                response_of_KG_list_path = []
                
                if result_path == [] or result_path == {}:
                    response_of_KG_list_path = '{}'
                else:
                    result_new_path = []
                    for total_path_i in result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path.append(path_input)

                    path = "\n".join(result_new_path[:5])
                    try:
                        response_of_KG_list_path = prompt_path_finding(path)
                        if is_unable_to_answer(response_of_KG_list_path):
                            response_of_KG_list_path = prompt_path_finding(path)

                    except Exception as e:
                        print("Error while fetching path:", e)
                        response_of_KG_list_path = '{}'

            else:
                response_of_KG_list_path = '{}'
        
            response_kg_retrieval = prompt_path_finding(kg_retrieval)
            if is_unable_to_answer(response_kg_retrieval):
                response_kg_retrieval = prompt_path_finding(kg_retrieval)



    #         # 6.1  After entity filtering and ranking，knowledge gragh path based prompt generation

            if len(sx_match_kg) != 1 and len(sx_match_kg) != 0 and sx_result_path != {}:
                response_of_KG_list_path_sx = []
                
                if sx_result_path == [] or sx_result_path == {}:
                    response_of_KG_list_path_sx = '{}'
                else:
                    result_new_path_sx = []
                    for total_path_i in sx_result_path:
                        path_input = "->".join(total_path_i)
                        result_new_path_sx.append(path_input)


                    path = "\n".join(result_new_path_sx[:5])  
                    try:
                        response_of_KG_list_path_sx = prompt_path_finding(path)
                        if is_unable_to_answer(response_of_KG_list_path_sx):
                            response_of_KG_list_path_sx = prompt_path_finding(path)

                    except Exception as e:
                        print("Error while fetching path for sx_match_kg:", e)
                        response_of_KG_list_path_sx = '{}'

            else:
                response_of_KG_list_path_sx = '{}'

            response_kg_retrieval_sx = prompt_path_finding(kg_retrieval)
            if is_unable_to_answer(response_kg_retrieval_sx):
                response_kg_retrieval_sx = prompt_path_finding(kg_retrieval)


            # # 7.  knowledge graph neighbor entities based prompt generation 
            response_of_KG_neighbor = []
            neighbor_new_list = []

            for neighbor_i in neighbor_list:
                neighbor = "->".join(neighbor_i)
                neighbor_new_list.append(neighbor)
            neighbor_input = ""

            if len(neighbor_new_list) == 0:
                neighbor_input = "无有效证据"  
            else:
                if len(neighbor_new_list) > 5:
                    neighbor_input = "\n".join(neighbor_new_list[:5])
                else:
                    neighbor_input = "\n".join(neighbor_new_list)

            if neighbor_input != "无有效证据":
                response_of_KG_neighbor = prompt_neighbor_with_retry(neighbor_input)
                while is_unable_to_answer(response_of_KG_neighbor):
                    print("Unable to answer, retrying...")
                    response_of_KG_neighbor = prompt_neighbor_with_retry(neighbor_input)
                
                response_of_KG_neighbor_accurate = Doctor_Agent_check(disease_label, input_text, response_of_KG_neighbor)

            else:
                response_of_KG_neighbor_accurate = "无有效证据"


            # # 7.1 MedEye for filtering and ranking+doctor agent examination,knowledge graph neighbor entities based prompt generation
            response_of_KG_neighbor_sx = []
            sx_neighbor_new_list = []

            for sx_neighbor_i in sx_neighbor_list:
                sx_neighbor = "->".join(sx_neighbor_i)
                sx_neighbor_new_list.append(sx_neighbor)

            sx_neighbor_input = ""

            if len(sx_neighbor_new_list) == 0:
                sx_neighbor_input = "无有效证据" 
            else:
                if len(sx_neighbor_new_list) > 5:
                    sx_neighbor_input = "\n".join(sx_neighbor_new_list[:5])
                else:
                    sx_neighbor_input = "\n".join(sx_neighbor_new_list)

            if sx_neighbor_input != "无有效证据":
                response_of_KG_neighbor_sx = prompt_neighbor_with_retry(sx_neighbor_input)
                
                while is_unable_to_answer(response_of_KG_neighbor_sx):

                    print("Unable to answer, retrying...")
                    response_of_KG_neighbor_sx = prompt_neighbor_with_retry(sx_neighbor_input)

                response_of_KG_neighbor_accurate_sx = Doctor_Agent_check(disease_label, input_text, response_of_KG_neighbor_sx)
            else:
                response_of_KG_neighbor_accurate_sx = "无有效证据"



        # # # 8 MedEye for filtering and ranking+doctor agent examination，prompt-based medical diaglogue answer generation
            output_all_sx = final_answer_Navigator(input_text, response_of_KG_list_path_sx, response_of_KG_neighbor_accurate_sx) 
            # 检查是否无法回答，若是，则重新调用生成
            if is_unable_to_answer(output_all_sx):
                output_all_sx = final_answer_Navigator(input_text, response_of_KG_list_path_sx, response_of_KG_neighbor_accurate_sx)

            # 正则表达式匹配
            re4_sx = r"输出1：(.*?)输出2："
            re5_sx = r"输出2：(.*?)输出3："

            flag_wrong_sx = 0

            output1_sx = re.findall(re4_sx, output_all_sx, flags=re.DOTALL)
            if len(output1_sx) > 0:
                output1_sx = output1_sx[0]
            else:
                flag_wrong_sx = 1

                print("output1_sx 为空，重新调用 final_answer_my 方法生成新的结果...")
                output_all_sx = final_answer_Navigator(input_text, response_of_KG_list_path_sx, response_of_KG_neighbor_accurate_sx)

                output1_sx = re.findall(re4_sx, output_all_sx, flags=re.DOTALL)
                if len(output1_sx) > 0:
                    output1_sx = output1_sx[0]
                else:
                    output1_sx = "重新生成的输出仍然为空"

            output2_sx = re.findall(re5_sx, output_all_sx, flags=re.DOTALL)
            if len(output2_sx) > 0:
                output2_sx = output2_sx[0]
            else:
                flag_wrong_sx = 1


            output3_index_sx = output_all_sx.find("输出3：")
            if output3_index_sx != -1:
                output3_sx = output_all_sx[output3_index_sx + len("输出3："):].strip()

            print('Navigator:\n', output1_sx)

    #         # # 9. Experiment 1: MindMaP
            output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)
            if is_unable_to_answer(output_all):
                output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)


            re4 = r"输出1：(.*?)输出2："
            re5 = r"输出2：(.*?)输出3："

            flag_wrong = 0

            output1 = re.findall(re4, output_all, flags=re.DOTALL)
            if len(output1) > 0:
                output1 = output1[0]
            else:
                flag_wrong = 1

                print("output1 为空，重新调用 final_answer 方法生成新的结果...")
                output_all = final_answer(input_text, response_of_KG_list_path, response_of_KG_neighbor)

                output1 = re.findall(re4, output_all, flags=re.DOTALL)
                if len(output1) > 0:
                    output1 = output1[0]
                else:
                    output1 = "重新生成的输出仍然为空"

            output2 = re.findall(re5, output_all, flags=re.DOTALL)
            if len(output2) > 0:
                output2 = output2[0]
            else:
                flag_wrong = 1

            output3_index = output_all.find("输出3：")
            if output3_index != -1:
                output3 = output_all[output3_index + len("输出3："):].strip()

            print('MindMap:\n', output1)


            ## 10. Experiment 2: chatgpt
            try:
                chatgpt_result = chat_35(str(input_text))
            except:
                sleep(40)
                chatgpt_result = chat_35(str(input_text))
            print('\nGPT-3.5:\n',chatgpt_result)
            
            ## 11. Experiment 3: document retrieval + bm25
            document_dir = "./data/CMCQA/document"
            document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]
            corpus = []
            for path in document_paths:
                with open(path, "r", encoding="utf-8") as f:
                    corpus.append(f.read().lower().split())
            dictionary = corpora.Dictionary(corpus)
            bm25_model = BM25Okapi(corpus)

            bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))

            query = input_text
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

            document_bm25_result = prompt_document(input_text,best_document_content)
            if is_unable_to_answer(document_bm25_result):
                document_bm25_result = prompt_document(input_text,best_document_content)
            
            print('\nBM25_retrieval:\n',document_bm25_result)

            ### 12. Experiment 3: document + embedding retrieval
            model = Word2Vec.load("./data/CMCQA/word2vec.model")
            ques_vec = np.mean([model.wv[token] for token in input_text.split()], axis=0)
            similarities = []
            for doc in docs:
                doc_vec = np.mean([model.wv[token] for token in doc.split()], axis=0)
                similarity = cosine_similarity([ques_vec], [doc_vec])[0][0]
                similarities.append(similarity)

            max_index = np.argmax(similarities)
            most_similar_doc = docs[max_index]
           
            document_embedding_result = prompt_document(input_text,most_similar_doc)
            if is_unable_to_answer(document_embedding_result):
                document_embedding_result = prompt_document(input_text,most_similar_doc)
            print('\nEmbedding retrieval:\n',document_embedding_result)

            ### 13. Experiment 5: kg retrieval
            kg_retrieval = prompt_document(input_text,kg_retrieval)
            if is_unable_to_answer(kg_retrieval):
                kg_retrieval = prompt_document(input_text,kg_retrieval)
            print('\nKG_retrieval:\n',kg_retrieval)


            ### 14. Experimet 6: gpt4
            try:
                gpt4_result = chat_4(str(input_text))
            except:
                gpt4_result = chat_4(str(input_text))
            print('\nGPT4:\n',gpt4_result)

            
            # ### save the final result
            with open('./output/cmcqa/output.csv', 'a+', newline='',encoding='utf-8') as f6:
                writer = csv.writer(f6)
                writer.writerow([f'{input_text}', f'{output_text}', f'{output1_sx}', f'{output1}',f'{chatgpt_result}', f'{document_bm25_result}', f'{document_embedding_result}', f'{kg_retrieval}',f'{gpt4_result}'])
                f6.flush()