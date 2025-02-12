import openai
import csv
from time import sleep
import os

openai.api_key = "<OPENAI-KEY>"

def prompt_comparation(reference, output1, output2):
    template = """
    Reference: {reference}
    \n\n
    output1: {output1}
    \n\n
    output2: {output2}
    \n\n
    According to the facts of disease diagnosis and drug and tests recommendation in reference output, which output is better match. If the output1 is better match, output '1'. If the output2 is better match, output '0'. If they are same match, output '2'. 
    """
    
    prompt = template.format(reference=reference, output1=output1, output2=output2)

    # 将 prompt 作为输入传递给 GPT-4 模型进行生成
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an excellent AI doctor."},
            {"role": "user", "content": prompt}
        ]
    ) 
    response_of_comparation = response.choices[0].message.content

    return response_of_comparation


def calculate_win_rate(wins, ties, loses):
    total = wins + ties + loses
    return (wins / total) * 100 if total > 0 else 0

def calculate_tie_rate(wins, ties, loses):
    total = wins + ties + loses
    return (ties / total) * 100 if total > 0 else 0

def calculate_lose_rate(wins, ties, loses):
    total = wins + ties + loses
    return (loses / total) * 100 if total > 0 else 0




input_file = './output/chatdoctor5k/output.csv'
output_file = './output/chatdoctor5k/output_gpt4_preference_total.csv'

total_win_rate_mindmap = 0
total_win_rate_gpt35 = 0
total_win_rate_bm25 = 0
total_win_rate_embedding = 0
total_win_rate_kg = 0
total_win_rate_gpt4 = 0

total_tie_rate_mindmap = 0
total_tie_rate_gpt35 = 0
total_tie_rate_bm25 = 0
total_tie_rate_embedding = 0
total_tie_rate_kg = 0
total_tie_rate_gpt4 = 0

total_lose_rate_mindmap = 0
total_lose_rate_gpt35 = 0
total_lose_rate_bm25 = 0
total_lose_rate_embedding = 0
total_lose_rate_kg = 0
total_lose_rate_gpt4 = 0

total_rows = 0

with open(input_file, 'r', newline="") as f_input, open(output_file, 'a+', newline='') as f_output:
    reader = csv.reader(f_input)
    writer = csv.writer(f_output)

    header = next(reader)
    header.extend(["output2_winrate", "output3_winrate", "output4_winrate", "output5_winrate", "output6_winrate", "output7_winrate"])
    writer.writerow(header)

    for row in reader:
        if len(row) < 9:
            print(f"跳过不完整的行: {row}")
            continue  
        output1_text = [row[2].strip("\n")]  
        output2_text = [row[3].strip("\n")]  
        output3_text = [row[4].strip("\n")]  
        output4_text = [row[5].strip("\n")]  
        output5_text = [row[6].strip("\n")]  
        output6_text = [row[7].strip("\n")]  
        output7_text = [row[8].strip("\n")]  

        references = [row[1].strip("\n")]

        win_counts = {
            'MindMap': 0,
            'GPT-3.5': 0,
            'BM25 Retriever': 0,
            'Embedding Retriever': 0,
            'KG Retriever': 0,
            'GPT-4': 0
        }
        tie_counts = {
            'MindMap': 0,
            'GPT-3.5': 0,
            'BM25 Retriever': 0,
            'Embedding Retriever': 0,
            'KG Retriever': 0,
            'GPT-4': 0
        }
        lose_counts = {
            'MindMap': 0,
            'GPT-3.5': 0,
            'BM25 Retriever': 0,
            'Embedding Retriever': 0,
            'KG Retriever': 0,
            'GPT-4': 0
        }

        flag = 0
        while flag == 0:
            try:

                response_of_comparation1 = prompt_comparation(references, output1_text, output2_text)
                response_of_comparation1 = int(response_of_comparation1.strip().replace("'", ""))
                if response_of_comparation1 == 1:
                    win_counts['MindMap'] += 1
                elif response_of_comparation1 == 2:
                    tie_counts['MindMap'] += 1
                elif response_of_comparation1 == 0:
                    lose_counts['MindMap'] += 1

                response_of_comparation2 = prompt_comparation(references, output1_text, output3_text)
                response_of_comparation2 = int(response_of_comparation2.strip().replace("'", ""))
                if response_of_comparation2 == 1:
                    win_counts['GPT-3.5'] += 1
                elif response_of_comparation2 == 2:
                    tie_counts['GPT-3.5'] += 1
                elif response_of_comparation2 == 0:
                    lose_counts['GPT-3.5'] += 1

                response_of_comparation3 = prompt_comparation(references, output1_text, output4_text)
                response_of_comparation3 = int(response_of_comparation3.strip().replace("'", ""))
                if response_of_comparation3 == 1:
                    win_counts['BM25 Retriever'] += 1
                elif response_of_comparation3 == 2:
                    tie_counts['BM25 Retriever'] += 1
                elif response_of_comparation3 == 0:
                    lose_counts['BM25 Retriever'] += 1

                response_of_comparation4 = prompt_comparation(references, output1_text, output5_text)
                response_of_comparation4 = int(response_of_comparation4.strip().replace("'", ""))
                if response_of_comparation4 == 1:
                    win_counts['Embedding Retriever'] += 1
                elif response_of_comparation4 == 2:
                    tie_counts['Embedding Retriever'] += 1
                elif response_of_comparation4 == 0:
                    lose_counts['Embedding Retriever'] += 1

                response_of_comparation5 = prompt_comparation(references, output1_text, output6_text)
                response_of_comparation5 = int(response_of_comparation5.strip().replace("'", ""))
                if response_of_comparation5 == 1:
                    win_counts['KG Retriever'] += 1
                elif response_of_comparation5 == 2:
                    tie_counts['KG Retriever'] += 1
                elif response_of_comparation5 == 0:
                    lose_counts['KG Retriever'] += 1

                response_of_comparation6 = prompt_comparation(references, output1_text, output7_text)
                response_of_comparation6 = int(response_of_comparation6.strip().replace("'", ""))
                if response_of_comparation6 == 1:
                    win_counts['GPT-4'] += 1
                elif response_of_comparation6 == 2:
                    tie_counts['GPT-4'] += 1
                elif response_of_comparation6 == 0:
                    lose_counts['GPT-4'] += 1

                flag = 1
            except Exception as e:
                print(f"发生错误: {e}，重新尝试...")
                sleep(10)

        win_rate_mindmap = calculate_win_rate(win_counts['MindMap'], tie_counts['MindMap'], lose_counts['MindMap'])
        win_rate_gpt35 = calculate_win_rate(win_counts['GPT-3.5'], tie_counts['GPT-3.5'], lose_counts['GPT-3.5'])
        win_rate_bm25 = calculate_win_rate(win_counts['BM25 Retriever'], tie_counts['BM25 Retriever'], lose_counts['BM25 Retriever'])
        win_rate_embedding = calculate_win_rate(win_counts['Embedding Retriever'], tie_counts['Embedding Retriever'], lose_counts['Embedding Retriever'])
        win_rate_kg = calculate_win_rate(win_counts['KG Retriever'], tie_counts['KG Retriever'], lose_counts['KG Retriever'])
        win_rate_gpt4 = calculate_win_rate(win_counts['GPT-4'], tie_counts['GPT-4'], lose_counts['GPT-4'])

        tie_rate_mindmap = calculate_tie_rate(win_counts['MindMap'], tie_counts['MindMap'], lose_counts['MindMap'])
        tie_rate_gpt35 = calculate_tie_rate(win_counts['GPT-3.5'], tie_counts['GPT-3.5'], lose_counts['GPT-3.5'])
        tie_rate_bm25 = calculate_tie_rate(win_counts['BM25 Retriever'], tie_counts['BM25 Retriever'], lose_counts['BM25 Retriever'])
        tie_rate_embedding = calculate_tie_rate(win_counts['Embedding Retriever'], tie_counts['Embedding Retriever'], lose_counts['Embedding Retriever'])
        tie_rate_kg = calculate_tie_rate(win_counts['KG Retriever'], tie_counts['KG Retriever'], lose_counts['KG Retriever'])
        tie_rate_gpt4 = calculate_tie_rate(win_counts['GPT-4'], tie_counts['GPT-4'], lose_counts['GPT-4'])

        lose_rate_mindmap = calculate_lose_rate(win_counts['MindMap'], tie_counts['MindMap'], lose_counts['MindMap'])
        lose_rate_gpt35 = calculate_lose_rate(win_counts['GPT-3.5'], tie_counts['GPT-3.5'], lose_counts['GPT-3.5'])
        lose_rate_bm25 = calculate_lose_rate(win_counts['BM25 Retriever'], tie_counts['BM25 Retriever'], lose_counts['BM25 Retriever'])
        lose_rate_embedding = calculate_lose_rate(win_counts['Embedding Retriever'], tie_counts['Embedding Retriever'], lose_counts['Embedding Retriever'])
        lose_rate_kg = calculate_lose_rate(win_counts['KG Retriever'], tie_counts['KG Retriever'], lose_counts['KG Retriever'])
        lose_rate_gpt4 = calculate_lose_rate(win_counts['GPT-4'], tie_counts['GPT-4'], lose_counts['GPT-4'])


        total_win_rate_mindmap += win_rate_mindmap
        total_win_rate_gpt35 += win_rate_gpt35
        total_win_rate_bm25 += win_rate_bm25
        total_win_rate_embedding += win_rate_embedding
        total_win_rate_kg += win_rate_kg
        total_win_rate_gpt4 += win_rate_gpt4

        total_tie_rate_mindmap += tie_rate_mindmap
        total_tie_rate_gpt35 += tie_rate_gpt35
        total_tie_rate_bm25 += tie_rate_bm25
        total_tie_rate_embedding += tie_rate_embedding
        total_tie_rate_kg += tie_rate_kg
        total_tie_rate_gpt4 += tie_rate_gpt4

        total_lose_rate_mindmap += lose_rate_mindmap
        total_lose_rate_gpt35 += lose_rate_gpt35
        total_lose_rate_bm25 += lose_rate_bm25
        total_lose_rate_embedding += lose_rate_embedding
        total_lose_rate_kg += lose_rate_kg
        total_lose_rate_gpt4 += lose_rate_gpt4


        row.extend([win_rate_mindmap, win_rate_gpt35, win_rate_bm25, win_rate_embedding, win_rate_kg, win_rate_gpt4])
        writer.writerow(row)

        total_rows += 1

avg_win_rate_mindmap = total_win_rate_mindmap / total_rows
avg_win_rate_gpt35 = total_win_rate_gpt35 / total_rows
avg_win_rate_bm25 = total_win_rate_bm25 / total_rows
avg_win_rate_embedding = total_win_rate_embedding / total_rows
avg_win_rate_kg = total_win_rate_kg / total_rows
avg_win_rate_gpt4 = total_win_rate_gpt4 / total_rows


avg_tie_rate_mindmap = total_tie_rate_mindmap / total_rows
avg_tie_rate_gpt35 = total_tie_rate_gpt35 / total_rows
avg_tie_rate_bm25 = total_tie_rate_bm25 / total_rows
avg_tie_rate_embedding = total_tie_rate_embedding / total_rows
avg_tie_rate_kg = total_tie_rate_kg / total_rows
avg_tie_rate_gpt4 = total_tie_rate_gpt4 / total_rows

avg_lose_rate_mindmap = total_lose_rate_mindmap / total_rows
avg_lose_rate_gpt35 = total_lose_rate_gpt35 / total_rows
avg_lose_rate_bm25 = total_lose_rate_bm25 / total_rows
avg_lose_rate_embedding = total_lose_rate_embedding / total_rows
avg_lose_rate_kg = total_lose_rate_kg / total_rows
avg_lose_rate_gpt4 = total_lose_rate_gpt4 / total_rows


print(f"与MindMap对比: 胜率: {avg_win_rate_mindmap}%, 平率: {avg_tie_rate_mindmap}%, 败率: {avg_lose_rate_mindmap}%")
print(f"与GPT-3.5对比: 胜率: {avg_win_rate_gpt35}%, 平率: {avg_tie_rate_gpt35}%, 败率: {avg_lose_rate_gpt35}%")
print(f"与BM25 Retriever对比: 胜率: {avg_win_rate_bm25}%, 平率: {avg_tie_rate_bm25}%, 败率: {avg_lose_rate_bm25}%")
print(f"与Embedding Retriever对比: 胜率: {avg_win_rate_embedding}%, 平率: {avg_tie_rate_embedding}%, 败率: {avg_lose_rate_embedding}%")
print(f"与KG Retriever对比: 胜率: {avg_win_rate_kg}%, 平率: {avg_tie_rate_kg}%, 败率: {avg_lose_rate_kg}%")
print(f"与GPT-4对比: 胜率: {avg_win_rate_gpt4}%, 平率: {avg_tie_rate_gpt4}%, 败率: {avg_lose_rate_gpt4}%")







