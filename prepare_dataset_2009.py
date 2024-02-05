#!/usr/bin/env python
# coding: utf-8
import random
import pandas as pd
import tqdm
import numpy as np
import pandas as pd
import chardet


from EduData import get_data
MAX_STEP = 50

get_data("assistment-2009-2010-skill", "../data")
# # get_data("assistment-2012-2013-non-skill", "../data")
# # get_data("assistment-2015", "../data")
# get_data("assistment-2017", "../data")

data = pd.read_csv(
    '../data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv',
    usecols=['order_id', 'user_id', 'sequence_id', 'problem_id','skill_id', 'correct'], encoding='ISO-8859-1' ,
).dropna(subset=['skill_id'])  #删除'skill_id'列中的缺失值
print(data.shape)

raw_question = data.skill_id.unique().tolist()
num_skill = len(raw_question)


raw_problem = data.problem_id.unique().tolist()
num_problem = len(raw_problem)

# question id from 1 to (num_skill )
questions = { q: i+1  for i, q in enumerate(raw_question) }  #字典

# problem id from 1 to (num_problem )
problem = { p: i+1 for i, p in enumerate(raw_problem) }

print("number of skills: %d" % num_skill)
print("number of problems: %d" % num_problem)

def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.user_id == student_id]) #data[data.user_id == student_id] 过滤出data.user_id 等于student_id的所有行
        all_sequences.extend([student_sequence]) #添加[student_sequence]到all_sequences
    return all_sequences

def parse_student_seq(student):
    seq = student.sort_values('order_id')
    q = [questions[q] for q in seq.skill_id.tolist()]  #把每个question id 对应的 1 to (num_skill)的序号取出来，用连续的小数字表示
    p = [problem[p] for p in seq.problem_id.tolist()]  # 把每个problem id 对应的 1 to (num_problem)的序号取出来，用连续的小数字表示
    a = seq.correct.tolist()
    return p, q, a

# [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
sequences = parse_all_seq(data.user_id.unique())


def train_test_split(data, train_size=0.7, shuffle=True, seed=1025):
    random.seed(seed)
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]



train_sequences, test_sequences = train_test_split(sequences)

def sequences2tl(sequences, trgpath):
    with open(trgpath, 'a', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write into file: '):
            problems, questions, answers = seq
            # seq_len = len(questions)
            seq_len = np.count_nonzero(questions)
            # if(seq_len>=10):
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(p) for p in problems]) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')

# save triple line format for other tasks

def split_and_pad_records(sequences, max_step):
    result = []
    for triplet in sequences:
        questions, knowledge_points, answers = triplet
        for i in range(0, len(questions), max_step):
            sub_q = list(questions[i:i+max_step])
            sub_k = list(knowledge_points[i:i+max_step])
            sub_a = list(answers[i:i+max_step])
            # pad the sub-record with 0 if its length is less than step
            if len(sub_q) < max_step:
                sub_q = list(np.pad(sub_q, (0, max_step-len(sub_q)), 'constant', constant_values=0))
                sub_k = list(np.pad(sub_k, (0, max_step-len(sub_k)), 'constant', constant_values=0))
                sub_a = list(np.pad(sub_a, (0, max_step-len(sub_a)), 'constant', constant_values=0))
            result.append((sub_q, sub_k, sub_a))
    return result



sequences2tl(split_and_pad_records(train_sequences,MAX_STEP), '../data/2009_skill_builder_data_corrected/train_assist2009.txt')
sequences2tl(split_and_pad_records(test_sequences,MAX_STEP), '../data/2009_skill_builder_data_corrected/test_assist2009.txt')










