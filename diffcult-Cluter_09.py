from torch.utils.data import DataLoader
from data_load_09  import read_file

from DKT_emb_cluster_dec import myKT_DKT
import torch
import numpy as np
import random
import json
import math

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1025)  # 你可以选择你想要的任何种子值

#读取数据
train_students = read_file("../data/2009_skill_builder_data_corrected/train_assist2009.txt")
test_students = read_file("../data/2009_skill_builder_data_corrected/test_assist2009.txt")

def  huizong(data):
    # 创建一个新的列表来存储转换后的元组
    students_problem= []
    students_quesion = []
    students_answer = []
    # 遍历每个学生
    for student in data:

        length=int(student[0])
        # 获取学生元组的题目，答案（一个字符串），并去掉末尾的换行符
        # problem = student[1].strip()
        question = student[2].strip()
        answer=student[3].strip()

        # 将字符串分割成单独的数字
        # p = problem.split(',')
        q = question.split(',')
        a = answer.split(',')
        # students_problem.extend(p[0:length])
        students_quesion.extend((q[0:length]))
        students_answer.extend(a[0:length])


    return  students_quesion,students_answer

q,a=huizong(train_students)


from collections import defaultdict
# 初始化字典
question_dict = defaultdict(lambda: {'total': 0, 'correct': 0})

# 遍历所有的答题结果
for question, result in zip(q, a):
    question_dict[question]['total'] += 1  # 更新题目的出现次数
    if int(result) == 1:
        question_dict[question]['correct'] += 1  # 如果答题结果是正确的，更新题目的正确次数

# 输出结果
# for question, stats in question_dict.items():
#
#     print(f"Question ID: {question}, Total: {stats['total']}, Correct: {stats['correct']}")

# for question, stats in sorted(question_dict.items(), key=lambda item: int(item[0])):
#     print(f"Question ID: {question}, Total: {stats['total']}, Correct: {stats['correct']},, 正确率: {stats['correct']/stats['total']}")

print("---------------------------------------------------------------------------------------------------------------------------------------------------------")
#下面的代码计算知识点出现的频率及正确率
# 计算最大的 total
max_total = max(stats['total'] for stats in question_dict.values())
print(f"max_total={max_total}")
print("---------------------------------------------------------------------------------------------------------------------------------------------------------")

# 创建一个新的字典来存储结果
new_dict = {}

for question, stats in question_dict.items():
    # 计算 correct/total 和 total/max_total
    correct_ratio = stats['correct'] / stats['total'] if stats['total'] != 0 else 0
    total_ratio = stats['total'] / max_total if max_total != 0 else 0

    # 将结果存入新的字典
    new_dict[question] = {
        'correct_ratio': correct_ratio,
        'total_ratio': total_ratio,
    }

#输出新的字典
for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
    print(f"Question ID: {question}, Correct Ratio: {stats['correct_ratio']}, Total Ratio: {stats['total_ratio']}")

print("---------------------------------------------------------------------------------------------------------------------------------------------------------")



import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gate(frequency, correctness, weight_f, weight_c):
    gate_f = sigmoid(weight_f*(frequency))  # 1-frequency 不出现的概率
    gate_c = sigmoid(weight_c*(1-correctness))  # 错误率的权重

    return gate_f * (frequency) + gate_c * (1-correctness)  # 返回值越小
    # return gate_c * (1 - correctness)  #频率大 错误率高 返回值越大

# 假设你的权重是这些值
weight_f = 0.5
weight_c = 0.5


for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
    frequency = stats['total_ratio']
    correctness = stats['correct_ratio']
    output = gate(frequency, correctness, weight_f, weight_c)
    print(f"Question ID: {question},  correct_ratio:{correctness}, total_ratio:{frequency}, gate: {output}")




# bloom_categories = {
#     '知识记忆': [],
#     '理解理念': [],
#     '应用技能': [],
#     '分析评估': [],
#     '创造设计': [],
#     '评价解决': []
# }


bloom_categories = {
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': []
}



# for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
#     frequency = stats['total_ratio']
#     correctness = stats['correct_ratio']
#     output = gate(frequency, correctness, weight_f, weight_c)
#
#     # thresholds = [0.15, 0.25, 0.4, 0.6, 0.7]  # 根据需要调整阈值
#     # thresholds=split_error_rates_list
#     thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # 根据需要调整阈值
#     thresholds_total_ratio = [0.2, 0.4, 0.5, 0.6, 0.8]  # 根据需要调整阈值
#     if output < thresholds[0] and frequency<thresholds_total_ratio[0]:
#         bloom_categories['1'].append(question)
#     elif output < thresholds[1] and frequency<thresholds_total_ratio[1]:
#         bloom_categories['2'].append(question)
#     elif output < thresholds[2] and frequency<thresholds_total_ratio[2]:
#         bloom_categories['3'].append(question)
#     elif output < thresholds[3] and frequency<thresholds_total_ratio[3]:
#         bloom_categories['4'].append(question)
#     elif output < thresholds[4] and frequency<thresholds_total_ratio[4]:
#         bloom_categories['5'].append(question)
#     else:
#         bloom_categories['6'].append(question)

correct_sort=[]


frequency_sort=[]

#按大小排序
sorted_dict_correct = sorted(new_dict.items(), key=lambda item: ( item[1]['correct_ratio']))
sorted_dict_total = sorted(new_dict.items(), key=lambda item: ( item[1]['total_ratio']))

correct_values = {}
for item in sorted_dict_correct:
    key = item[0]  # 键
    value = item[1]['correct_ratio']  # 对应的值
    correct_values[key] = value
    correct_dict = dict(correct_values)
print(correct_dict)


#把正确率结果写入文件
with open('../data/bloom_categories_correct_assist2009.json', 'w') as file:
    json.dump(correct_dict, file)

total_values = {}
for item in sorted_dict_total:
    key = item[0]  # 键
    value = item[1]['total_ratio']  # 对应的值
    total_values[key] = value
    total_dict = dict(total_values)
print(total_dict)

#把频率结果写入文件
with open('../data/bloom_categories_total_assist2009.json', 'w') as file:
    json.dump(total_dict, file)

print("++++++++++++++++++++++++++++++++++++++++++++++++++")
# print(sorted_dict_correct)
for question, stats in sorted_dict_correct:
    # frequency = stats['total_ratio']
    correctness = stats['correct_ratio']
    # 进行后续操作
    correct_sort.append((correctness))
    # print(f"Question ID: {question},  total_ratio:{correctness}")





for question, stats in sorted_dict_total:
    frequency = stats['total_ratio']
    # correctness = stats['correct_ratio']
    # 进行后续操作
    frequency_sort.append((frequency))
    # print(f"Question ID: {question},  total_ratio:{correctness}")


lenght = len(sorted_dict_correct)  # 列表长度
#获取从低到高排好序的list中前百分之N的那个数字
def top_percent(list,legth,percent):

    ten_percent_range = int(legth * percent)  # 前percent%的索引范围
    # 提取前percent%的数据
    extracted_data = list[:ten_percent_range]
    # 获取前percent%数据中的最大值
    max_value = (max(extracted_data))
    return max_value

thresholds_correct_ratio = [top_percent(correct_sort,lenght,0.1), top_percent(correct_sort,lenght,0.3), top_percent(correct_sort,lenght,0.6),top_percent(correct_sort,lenght,0.8), top_percent(correct_sort,lenght,0.9)]  # 根据需要调整阈值
thresholds_total_ratio = [ top_percent(frequency_sort,lenght,0.3), top_percent(frequency_sort,lenght,0.5),top_percent(frequency_sort,lenght,0.7)] # 根据需要调整阈值


#按正确率和频率共同决定分类结果
# for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
#     frequency = stats['total_ratio']
#     correctness = stats['correct_ratio']
#     if correctness < thresholds_correct_ratio[0] and frequency<thresholds_total_ratio[0]:
#         bloom_categories['6'].append(question)
#     elif correctness < thresholds_correct_ratio[1] and frequency<thresholds_total_ratio[0]:
#         bloom_categories['5'].append(question)
#     elif correctness < thresholds_correct_ratio[2] and frequency<thresholds_total_ratio[1]:
#         bloom_categories['4'].append(question)
#     elif correctness < thresholds_correct_ratio[3] and frequency<thresholds_total_ratio[2]:
#         bloom_categories['3'].append(question)
#     elif correctness < thresholds_correct_ratio[4] and frequency<thresholds_total_ratio[0]:
#         bloom_categories['2'].append(question)
#     else:
#         bloom_categories['1'].append(question)

#只考虑正确率
for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
    frequency = stats['total_ratio']
    correctness = stats['correct_ratio']
    if correctness < thresholds_correct_ratio[0] :
        bloom_categories['6'].append(question)
    elif correctness < thresholds_correct_ratio[1] :
        bloom_categories['5'].append(question)
    elif correctness < thresholds_correct_ratio[2] :
        bloom_categories['4'].append(question)
    elif correctness < thresholds_correct_ratio[3] :
        bloom_categories['3'].append(question)
    elif correctness < thresholds_correct_ratio[4] :
        bloom_categories['2'].append(question)
    else:
        bloom_categories['1'].append(question)

# 打印结果
for category, questions in bloom_categories.items():
    print(f"{category}: {questions}")
print("---------------------------------------------------------------------------------------------------------------------------------------------------------")



import json

reversed_bloom_categories = {}

for key, value_list in bloom_categories.items():
    for value in value_list:
        reversed_bloom_categories[value] = key
#把结果写入文件
with open('../data/bloom_categories_assist2009.json', 'w') as file:
    json.dump(reversed_bloom_categories, file)


#按大小排序
# sorted_dict = sorted(new_dict.items(), key=lambda item: ( item[1]['total_ratio']))
#
# for question, stats in sorted_dict:
#     frequency = stats['total_ratio']
#     correctness = stats['correct_ratio']
#     # 进行后续操作
#     print(f"Question ID: {question},  total_ratio:{frequency}")

