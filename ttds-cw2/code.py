# 2020314 all the code

import sys
import csv
import math
import re
import string
import numpy as np
import operator
import random
from collections import defaultdict
from stemming.porter2 import stem
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.svm import SVC

input_system_results_csv = './system_results.csv'
input_qrels_csv = './qrels.csv'
output_ir_eval_csv = './ir_eval.csv'
check_ir_format_py = './check_ir_format.py'
output_classification_eval_csv = './classification_eval.csv'
test_corpora_tsv = './test.tsv'

# system_number,query_number,doc_number,rank_of_doc,score
system_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# query_id,doc_id,relevances
qrels_dict = defaultdict(lambda: defaultdict(int))
system_number = 6
query_number = 10

input_corpora_tsv = './train_and_dev.tsv'
check_stopword_txt = './englishST.txt'
corpora_names = ['Quran', 'OT', 'NT']
corpora2documents = defaultdict(list)
r = re.compile(r'[\s{}]+'.format(re.escape(string.punctuation)))

k = 20
lda = LDA(n_components=k, random_state=0) 
bag_of_all_terms = defaultdict(lambda: defaultdict(int))
corpora_token_score_mi = defaultdict(lambda: defaultdict(float))
corpora_token_score_chi = defaultdict(lambda: defaultdict(float))

def P_10():
    '''
    precision at cutoff 10
    (only top 10 retrieved documents in the list are considered for each query)
    '''
    cutoff = 10
    precision_system_query_dict = defaultdict(lambda: defaultdict(float))
    for system in system_results_dict.keys():
        for query in system_results_dict[system].keys():
            # for system-query pair
            rel_ret_number = 0
            # get query relevant document list
            query_relevant_docs = list(qrels_dict[query].keys())
            # get top cutoff document list
            # check documents relevant
            for key, value in sorted(system_results_dict[system][query].items(), key=lambda item:int(item[1][0])):
                if key in query_relevant_docs:
                    rel_ret_number += 1
                if int(value[0]) >= cutoff:
                    break
            precision = rel_ret_number/cutoff
            precision_system_query_dict[system][query] = precision
    return precision_system_query_dict
    
def R_50():
    '''
    recall at cutoff 50
    '''
    cutoff = 50
    recall_system_query_dict = defaultdict(lambda: defaultdict(float))
    for system in system_results_dict.keys():
        for query in system_results_dict[system].keys():
            # for system-query pair
            rel_ret_number = 0
            # get query relevant document list
            query_relevant_docs = list(qrels_dict[query].keys())
            # get top cutoff document list
            # check documents relevant
            for key, value in sorted(system_results_dict[system][query].items(), key=lambda item:int(item[1][0])):
                if key in query_relevant_docs:
                    rel_ret_number += 1
                if int(value[0]) >= cutoff:
                    break
            recall = rel_ret_number/len(query_relevant_docs)
            recall_system_query_dict[system][query] = recall
    return recall_system_query_dict

def R_precision():
    '''
    cutoff = relevant doc number
    '''
    r_precision_system_query_dict = defaultdict(lambda: defaultdict(float))
    for system in system_results_dict.keys():
        for query in system_results_dict[system].keys():
            # for system-query pair
            rel_ret_number = 0
            # get query relevant document list
            query_relevant_docs = list(qrels_dict[query].keys())
            for key, value in sorted(system_results_dict[system][query].items(), key=lambda item:int(item[1][0])):
                if key in query_relevant_docs:
                    rel_ret_number += 1
                if int(value[0]) >= len(query_relevant_docs):
                    break
            r_precision = rel_ret_number/len(query_relevant_docs)
            r_precision_system_query_dict[system][query] = r_precision
    return r_precision_system_query_dict

def AP():
    '''
    average precision
    hint:   for all previous scores, the value of relevance should be considered as 1.
            Being 1, 2, or 3 should not make a difference on the score.
    '''
    ap_system_query_dict = defaultdict(lambda: defaultdict(float))
    for system in system_results_dict.keys():
        for query in system_results_dict[system].keys():
            # for system-query pair
            precision_list = []
            # get query relevant document list
            query_relevant_docs = list(qrels_dict[query].keys())
            rel_appearance_number = 0
            for key, value in sorted(system_results_dict[system][query].items(), key=lambda item:int(item[1][0])):
                if rel_appearance_number >= len(query_relevant_docs):
                    break
                # need to find the next relevant doc
                if key in query_relevant_docs:
                    rel_appearance_number += 1
                    new_precision = rel_appearance_number/int(value[0])
                    precision_list.append(new_precision)
            ap = sum(precision_list)/len(query_relevant_docs)
            ap_system_query_dict[system][query] = ap
    return ap_system_query_dict

def nDCG_k(k):
    cutoff = k
    nDCG_k_system_query_dict = defaultdict(lambda: defaultdict(float))
    for system in system_results_dict.keys():
        for query in system_results_dict[system].keys():
            # for system-query pair
            DCG = []
            iDCG = []
            DCG_k = 0
            iDCG_k = 1
            nDCG_k = 0
            # get query relevant document list
            query_relevant_docs = list(qrels_dict[query].keys())
            for key, value in sorted(system_results_dict[system][query].items(), key=lambda item:int(item[1][0])):
                DG = 0
                if key in query_relevant_docs:
                    division = 1
                    if int(value[0]) != 1:
                        division = math.log2(int(value[0]))
                    DG = int(qrels_dict[query][key])/division
                if len(DCG) != 0:
                    DCG.append(DG+DCG[-1])
                else:
                    DCG.append(DG)
                if int(value[0]) >= cutoff:
                    break
            ideal_index = 1
            for key, value in sorted(qrels_dict[query].items(), key=lambda item:int(item[1]), reverse=True):
                division = 1
                if ideal_index != 1:
                    division = math.log2(ideal_index)
                iDG = int(value)/division
                if len(iDCG) != 0:
                    iDCG.append(iDG+iDCG[-1])
                else:
                    iDCG.append(iDG)
                ideal_index += 1
                if ideal_index > cutoff:
                    break  
            # calculate nDCG@10
            if len(DCG) and len(DCG) < cutoff:
                DCG_k = DCG[-1]
            else:
                DCG_k = DCG[cutoff-1]
            if len(iDCG) and len(iDCG) < cutoff:
                iDCG_k = iDCG[-1]
            else:
                iDCG_k = iDCG[cutoff-1]
            if iDCG_k == 0:
                iDCG_k = 1
            nDCG_k = DCG_k/iDCG_k
            nDCG_k_system_query_dict[system][query] = nDCG_k
    return nDCG_k_system_query_dict

def nDCG_10():
    '''
    normalized discount cumulative gain at cutoff 10
    use equation in lecture 9
    '''
    cutoff = 10
    return nDCG_k(cutoff)

def nDCG_20():
    '''
    normalized discount cumulative gain at cutoff 20
    use equation in lecture 9
    '''
    cutoff = 20
    return nDCG_k(cutoff)

def create_ir_eval_csv(p_10_dict, r_50_dict, r_precision_dict, ap_dict, ndcg_10_dict, ndcg_20_dict):
    '''
    1) A comma-separated-values with certain format
    system_number (1-6)
    10 queries and mean (across 10 queries)
    6 evaluation results
    system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20
    2) round the scores to 3 decimal points (e.g.: 0.046317 --> 0.046)
    '''
    header = ['system_number','query_number','P@10','R@50','r-precision','AP','nDCG@10','nDCG@20']
    # output_ir_eval
    with open(output_ir_eval_csv, 'w', newline='') as csvfile:
        ir_eval_writer = csv.writer(csvfile, delimiter=',')
        ir_eval_writer.writerow(header)
        for system_id in range(system_number):
            for query_id in range(query_number):
                line = []
                line.append(str(system_id+1))
                line.append(str(query_id+1))
                line.append("{:.3f}".format(p_10_dict[str(system_id+1)][str(query_id+1)]))
                line.append("{:.3f}".format(r_50_dict[str(system_id+1)][str(query_id+1)]))
                line.append("{:.3f}".format(r_precision_dict[str(system_id+1)][str(query_id+1)]))
                line.append("{:.3f}".format(ap_dict[str(system_id+1)][str(query_id+1)]))
                line.append("{:.3f}".format(ndcg_10_dict[str(system_id+1)][str(query_id+1)]))
                line.append("{:.3f}".format(ndcg_20_dict[str(system_id+1)][str(query_id+1)]))
                ir_eval_writer.writerow(line)
            mean_line = []
            mean_line.append(str(system_id+1))
            mean_line.append('mean')
            mean_line.append("{:.3f}".format(sum(p_10_dict[str(system_id+1)].values())/query_number))
            mean_line.append("{:.3f}".format(sum(r_50_dict[str(system_id+1)].values())/query_number))
            mean_line.append("{:.3f}".format(sum(r_precision_dict[str(system_id+1)].values())/query_number))
            mean_line.append("{:.3f}".format(sum(ap_dict[str(system_id+1)].values())/query_number))
            mean_line.append("{:.3f}".format(sum(ndcg_10_dict[str(system_id+1)].values())/query_number))
            mean_line.append("{:.3f}".format(sum(ndcg_20_dict[str(system_id+1)].values())/query_number))
            ir_eval_writer.writerow(mean_line)
    # check file correct format using check_ir_format

def evaluation_combo():
    create_ir_eval_csv(P_10(), R_50(), R_precision(), AP(), nDCG_10(), nDCG_20())
    sys.exit(0)

def read_system_results_csv(file):
    with open(file, newline='') as csvfile:
        system_results_reader = csv.reader(csvfile, delimiter=',')
        next(system_results_reader)
        for row in system_results_reader:
            system_results_dict[row[0]][row[1]][row[2]] = row[3:]

def read_qrels_csv(file):
    with open(file, newline='') as csvfile:
        qrels_reader = csv.reader(csvfile, delimiter=',')
        next(qrels_reader)
        for row in qrels_reader:
            qrels_dict[row[0]][row[1]] = row[2]

def EVAL():
    # EVAL
    # read input_system_results_csv and input_qrels_csv    
    read_system_results_csv(input_system_results_csv)
    read_qrels_csv(input_qrels_csv)
    # get evaluation results
    evaluation_combo()
    
def text_preprocess(text, corpora):
    '''
    tokenization, stopword removal
    '''
    tokens = []
    with open(check_stopword_txt, 'r') as stopperReader:
        stopper = stopperReader.read().splitlines()
        stopperReader.close()
    for token in r.split(text.lower().strip()):
        if token not in stopper and token != '':
            stemmed_token = stem(token)
            if stemmed_token not in tokens:
                bag_of_all_terms[stemmed_token][corpora] += 1
            tokens.append(stemmed_token)
    return tokens

def save_corpora_term_score_mi():
    for corpora_index in range(3):
        with open('./'+corpora_names[corpora_index]+'_mi_term_score.csv', 'w', newline='') as csvfile:
            mi_writer = csv.writer(csvfile, delimiter=',')
            for term, score in sorted(corpora_token_score_mi[corpora_names[corpora_index]].items(), key=lambda item:item[1], reverse=True):
                line = []
                line.append(term)
                line.append("{:3f}".format(score))
                mi_writer.writerow(line)

def MI():
    N = len(corpora2documents[corpora_names[0]]) + len(corpora2documents[corpora_names[1]]) + len(corpora2documents[corpora_names[2]])
    for term in bag_of_all_terms.keys():
        # for each term
        for corpora_id in range(3):
            # for each term-corpora(target) pair
            # first 0/1 (not) contain term, second 0/1 (not) in target corpora
            N11 = bag_of_all_terms[term][corpora_names[corpora_id]]
            N01 = len(corpora2documents[corpora_names[corpora_id]]) - N11
            N10 = sum(bag_of_all_terms[term].values()) - N11
            N00 = len(corpora2documents[corpora_names[(corpora_id+1)%3]]) + len(corpora2documents[corpora_names[(corpora_id+2)%3]]) - N10
            N1n = N11+N10
            N0n = N01+N00
            Nn1 = N01+N11
            Nn0 = N00+N10
            score11, score01, score10, score00 = 0, 0, 0, 0
            if N11 != 0:
                score11 = math.log2(N*N11/(N1n*Nn1)) * N11/N
            if N01 != 0:
                score01 = math.log2(N*N01/(N0n*Nn1)) * N01/N
            if N10 != 0:
                score10 = math.log2(N*N10/(N1n*Nn0)) * N10/N
            if N00 != 0:
                score00 = math.log2(N*N00/(N0n*Nn0)) * N00/N
            score = score11+score01+score10+score00
            corpora_token_score_mi[corpora_names[corpora_id]][term] = score
    save_corpora_term_score_mi()

def save_corpora_term_score_chi():
    for corpora_index in range(3):
        with open('./'+corpora_names[corpora_index]+'_chi_term_score.csv', 'w', newline='') as csvfile:
            mi_writer = csv.writer(csvfile, delimiter=',')
            for term, score in sorted(corpora_token_score_chi[corpora_names[corpora_index]].items(), key=lambda item:item[1], reverse=True):
                line = []
                line.append(term)
                line.append("{:3f}".format(score))
                mi_writer.writerow(line)

def Chi_squared():
    N = len(corpora2documents[corpora_names[0]]) + len(corpora2documents[corpora_names[1]]) + len(corpora2documents[corpora_names[2]])
    for term in bag_of_all_terms.keys():
        # for each term
        for corpora_id in range(3):
            # for each term-corpora(target) pair
            # first 0/1 (not) contain term, second 0/1 (not) in target corpora
            N11 = bag_of_all_terms[term][corpora_names[corpora_id]]
            N01 = len(corpora2documents[corpora_names[corpora_id]]) - N11
            N10 = sum(bag_of_all_terms[term].values()) - N11
            N00 = len(corpora2documents[corpora_names[(corpora_id+1)%3]]) + len(corpora2documents[corpora_names[(corpora_id+2)%3]]) - N10
            N1n = N11+N10
            N0n = N01+N00
            Nn1 = N01+N11
            Nn0 = N00+N10
            corpora_token_score_chi[corpora_names[corpora_id]][term] = N*((N11*N00-N10*N01)**2)/(Nn1*N1n*Nn0*N0n)
    save_corpora_term_score_chi()
    
def useLDA():
    N = len(corpora2documents[corpora_names[0]]) + len(corpora2documents[corpora_names[1]]) + len(corpora2documents[corpora_names[2]])
    # transform bag_of_all_terms to document frequency sparse matrix
    '''deprecated
#    matrix = csr_matrix((3, len(bag_of_all_terms.keys())), dtype=np.int)
#    for term_id, term in enumerate(bag_of_all_terms.keys()):
#        for corpora_index in range(3):
#            matrix[corpora_index, term_id] = bag_of_all_terms[term][corpora_names[corpora_index]]
    
#    for term_id, term in enumerate(bag_of_all_terms.keys()):
#        matrix[]
#        for corpora_index in range(3):
#            matrix[corpora_index, term_id] = bag_of_all_terms[term][corpora_names[corpora_index]]
    # memory error and more
    rows, cols, vals = [], [], []
    matrix_depth = 0
    for corpora_index in range(3):
        for doc_depth in range(len(corpora2documents[corpora_names[corpora_index]])):
            for term_id, term in enumerate(bag_of_all_terms.keys()):
                rows.append(matrix_depth)
                cols.append(term_id)
                if term in corpora2documents[corpora_names[corpora_index]][doc_depth]:
                    vals.append(1)
                else:
                    vals.append(0)
            matrix_depth+=1
    matrix = csr_matrix((vals, (rows, cols)))
    '''
    
    '''lack efficiency'''
    matrix = csr_matrix((N, len(bag_of_all_terms.keys())), dtype=np.int)
    matrix_depth = 0
    for corpora_index in range(3):
        for doc_depth in range(len(corpora2documents[corpora_names[corpora_index]])):
            # the doc_depth th document in corpora[index]
            for term_id, term in enumerate(bag_of_all_terms.keys()):
                if term in corpora2documents[corpora_names[corpora_index]][doc_depth]:
                    matrix[matrix_depth, term_id] = 1
            matrix_depth += 1
    
    lda.fit(matrix)
    matrix_Quran = matrix[:len(corpora2documents[corpora_names[0]])]
    matrix_OT = matrix[len(corpora2documents[corpora_names[0]]):len(corpora2documents[corpora_names[0]])+len(corpora2documents[corpora_names[1]])]
    matrix_NT = matrix[len(corpora2documents[corpora_names[0]])+len(corpora2documents[corpora_names[1]]):len(corpora2documents[corpora_names[0]])+len(corpora2documents[corpora_names[1]])+len(corpora2documents[corpora_names[2]])]
    Quran_document_topic_score = lda.transform(matrix_Quran)
    OT_document_topic_score = lda.transform(matrix_OT)
    NT_document_topic_score = lda.transform(matrix_NT)
    Quran_average_topic_score = [sum(x)/len(corpora2documents[corpora_names[0]]) for x in zip(*Quran_document_topic_score)]
    OT_average_topic_score = [sum(x)/len(corpora2documents[corpora_names[1]]) for x in zip(*OT_document_topic_score)]
    NT_average_topic_score = [sum(x)/len(corpora2documents[corpora_names[2]]) for x in zip(*NT_document_topic_score)]
    
    Quran_best_topic_index, Quran_best_topic_score = max(enumerate(Quran_average_topic_score), key=operator.itemgetter(1))
    OT_best_topic_index, OT_best_topic_score = max(enumerate(OT_average_topic_score), key=operator.itemgetter(1))
    NT_best_topic_index, NT_best_topic_score = max(enumerate(NT_average_topic_score), key=operator.itemgetter(1))

    topic_term_probability_matrix = lda.components_ / lda.components_.sum(axis=0)
    
    with open('./topic_analysis_table_stats.txt', 'w') as table_stat_writer:
        table_stat_writer.write('For Quran - topic ' + str(Quran_best_topic_index) +' with average score of ' + str(Quran_best_topic_score) +':\n')
        for feature_index in topic_term_probability_matrix[Quran_best_topic_index].argsort()[-10:][::-1]:
            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[Quran_best_topic_index][feature_index]) + '\n')
        table_stat_writer.write('For OT - topic ' + str(OT_best_topic_index) +' with average score of ' + str(OT_best_topic_score) +':\n')
        for feature_index in topic_term_probability_matrix[OT_best_topic_index].argsort()[-10:][::-1]:
            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[OT_best_topic_index][feature_index]) + '\n')
        table_stat_writer.write('For NT - topic ' + str(NT_best_topic_index) +' with average score of ' + str(NT_best_topic_score) +':\n')
        for feature_index in topic_term_probability_matrix[NT_best_topic_index].argsort()[-10:][::-1]:
            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[NT_best_topic_index][feature_index]) + '\n')
    
#    with open('./topic_analysis_table_stats.txt', 'w') as table_stat_writer:
#        table_stat_writer.write('For Quran:\n')
#        for feature_index in lda.components_[Quran_best_topic_index].argsort()[-10:][::-1]:
#            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[Quran_best_topic_index][feature_index]) + '\n')
#        table_stat_writer.write('For OT:\n')
#        for feature_index in lda.components_[OT_best_topic_index].argsort()[-10:][::-1]:
#            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[OT_best_topic_index][feature_index]) + '\n')
#        table_stat_writer.write('For NT:\n')
#        for feature_index in lda.components_[NT_best_topic_index].argsort()[-10:][::-1]:
#            table_stat_writer.write(str(list(bag_of_all_terms.keys())[feature_index]) + ' ' +  str(topic_term_probability_matrix[NT_best_topic_index][feature_index]) + '\n')

#    print('For Quran:')
#    for feature_index in lda.components_[Quran_best_topic_index].argsort()[-10:]:
#        print(list(bag_of_all_terms.keys())[feature_index], lda.components_[Quran_best_topic_index][feature_index])
#    print('For OT:')
#    for feature_index in lda.components_[OT_best_topic_index].argsort()[-10:]:
#        print(list(bag_of_all_terms.keys())[feature_index], lda.components_[OT_best_topic_index][feature_index])
#    print('For NT:')
#    for feature_index in lda.components_[NT_best_topic_index].argsort()[-10:]:
#        print(list(bag_of_all_terms.keys())[feature_index], lda.components_[NT_best_topic_index][feature_index])

def ANALYSIS():
    with open(input_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        for row in corpora_reader:
            corpora2documents[row[0]].append(text_preprocess(row[1], row[0]))
    # doc size: 5612, 16720, 5242
    # 10025 terms in total
    # compute MI and X^2 scores for all tokens (after preprocessing) for each corpora.
    '''Compute the Mutual Information and Χ2 scores for all tokens (after preprocessing) 
    for each of the three corpora. 
    Generate a ranked list of the results, in the format token,score.
    '''
    MI()
    Chi_squared()
    '''
    Run LDA on the entire set of verses from ALL corpora together. Set k=20 topics and inspect the results.
    - For each corpus, compute the average score for each topic 
    by summing the document-topic probability for each document in that corpus 
    and dividing by the total number of documents in the corpus.
    - Then, you should identify the topic that has the highest average score for the three corpora (3 topics). 
    For each of those three topics, find the top 10 tokens with highest probabilty of belonging to that topic.
    '''
    useLDA()

docuid2label2tokens = defaultdict(list)
corpusname2id = {'Quran':0, 'OT':1, 'NT':2}
corpusid2name = {0:'Quran', 1:'OT', 2:'NT'}
corpustoken2id = defaultdict(int)
corpusid2token = defaultdict(str)
testdocuid2tokens = defaultdict(list)

train_id_set, devel_id_set, train_correct_cato, devel_correct_cato = [], [], [], []
svc_model = SVC(C=1000, random_state=2020314)
svc_model_improved = SVC(C=50, gamma=0.007, random_state=2020314)#, tol=0.01 decision_function_shape='ovo', 
train_predict_cato = []
devel_predict_cato = []

test_correct_cato, test_predict_cato = [], []

def simple_preprocess(text):
    '''
    token
    '''
    tokens = []
    for token in r.split(text.lower().strip()):
        tokens.append(token)
        if token not in corpustoken2id:
            l = len(corpustoken2id)
            corpustoken2id[token] = l
            corpusid2token[l] = token
    return tokens

def evaluate_prediction(predict_cato, correct_cato):
    '''for each class
        precision, recall, f1-score
        across three classes
        macro-averaged precision, recall, f1-score
    '''
    if len(predict_cato) != len(correct_cato):
        sys.exit('wrong evalution size')
    N = len(predict_cato)
    predict_senario_count_matrix = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    for doc_index in range(N):
        predict_senario_count_matrix[predict_cato[doc_index]][correct_cato[doc_index]] += 1
    # [[5076, 0, 0], [0, 14997, 0], [0, 0, 4743]]
    # [[506, 16, 16], [26, 1667, 73], [4, 40, 410]]
    precision_corpus, recall_corpus, f1_corpus = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for corpus_index in range(3):
        randp = predict_senario_count_matrix[corpus_index][corpus_index]
        recall_pool = sum(predict_senario_count_matrix[cor][corpus_index] for cor in range(3))
        precis_pool = sum(predict_senario_count_matrix[corpus_index][cor] for cor in range(3))
        precision_corpus[corpus_index] = randp / precis_pool
        recall_corpus[corpus_index] = randp / recall_pool
        f1_corpus[corpus_index] = 2*randp / (precis_pool+recall_pool)
    return [precision_corpus, recall_corpus, f1_corpus]

def create_classification_eval_csv(systemname, train_evaluation, devel_evaluation, test_evaluation):
    header = ['system','split','p-quran','r-quran','f-quran','p-ot','r-ot','f-ot','p-nt','r-nt','f-nt','p-macro','r-macro','f-macro',]
    with open(output_classification_eval_csv, 'w', newline='') as csvfile:
        cl_eval_writer = csv.writer(csvfile, delimiter=',')
        cl_eval_writer.writerow(header)
        
        line = []
        line.append(systemname)
        line.append('train')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(train_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(train_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)
        
        line = []
        line.append(systemname)
        line.append('dev')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(devel_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(devel_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)
        
        line = []
        line.append(systemname)
        line.append('test')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(test_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(test_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)

def simpler_preprocess(text):
    '''
    token 
    '''
    tokens = []
    for token in r.split(text.lower().strip()):
        if token != '':
            tokens.append(token)
    return tokens

def CLASSIFICATION_BASELINE():
    '''given a verse, predict which corpus that verse belongs to.'''
    '''Shuffle the order of the data and split the dataset into a training set and a separate development set. 
    You can split the data however you like. For example, 
    you can use 90% of the documents for training and 10% for testing.
    '''
    with open(input_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        line_counter = 0
        for row in corpora_reader:
            tokens = simple_preprocess(row[1])
            docuid2label2tokens[line_counter] = [corpusname2id[row[0]], tokens]
            line_counter+=1
    count_matrix = dok_matrix((len(docuid2label2tokens), len(corpusid2token)), dtype=int)
    # count matrix
    for docuid in docuid2label2tokens.keys():
        tokens = docuid2label2tokens[docuid][1]
        for token in tokens:
            count_matrix[docuid, corpustoken2id[token]] += 1
    whole_id_set = list(docuid2label2tokens.keys())
    random.Random(2020314).shuffle(whole_id_set)
    train_id_set.extend(whole_id_set[:int(0.9*len(docuid2label2tokens.keys()))])
    devel_id_set.extend(whole_id_set[int(0.9*len(docuid2label2tokens.keys())):])
    [train_correct_cato.append(docuid2label2tokens[doc_idx][0]) for doc_idx in train_id_set]
    [devel_correct_cato.append(docuid2label2tokens[doc_idx][0]) for doc_idx in devel_id_set]
    print('training and development set ready')
    train_matrix = count_matrix[train_id_set, :]
    devel_matrix = count_matrix[devel_id_set, :]
    print('fitting model')
    svc_model.fit(train_matrix, train_correct_cato)
    print('making prediction')
    train_predict_cato.extend(svc_model.predict(train_matrix))
    devel_predict_cato.extend(svc_model.predict(devel_matrix))
    train_evaluation, devel_evaluation = evaluate_prediction(train_predict_cato, train_correct_cato), evaluate_prediction(devel_predict_cato, devel_correct_cato)
    
    test_doc_depth = 0
    with open(test_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        for row in corpora_reader:
            tokens = simpler_preprocess(row[1])
            test_correct_cato.append(corpusname2id[row[0]])
            testdocuid2tokens[test_doc_depth] = tokens
            test_doc_depth += 1
    count_test_matrix = dok_matrix((len(testdocuid2tokens), len(corpusid2token)), dtype=int)
    for docuid in testdocuid2tokens.keys():
        tokens = testdocuid2tokens[docuid]
        for token in tokens:
            if token in corpustoken2id.keys():
                count_test_matrix[docuid, corpustoken2id[token]] += 1
    test_predict_cato.extend(svc_model.predict(count_test_matrix))
    test_evaluation = evaluate_prediction(test_predict_cato, test_correct_cato)
    
    create_classification_eval_csv('baseline', train_evaluation, devel_evaluation, test_evaluation)
    
    '''
        first three wrong prediction in devel
0
14 - docid=23066
2
1 - NT to OT
docuid2label2tokens[23066][1]
['but', 'there', 'was', 'certain', 'of', 'the', 'scribes',
       'sitting', 'there', 'and', 'reasoning', 'in', 'their', 'hearts',
       '2', '7', 'why', 'doth', 'this', 'man', 'thus', 'speak',
       'blasphemies', 'who', 'can', 'forgive', 'sins', 'but', 'god',
       'only', '2', '8', 'and', 'immediately', 'when', 'jesus',
       'perceived', 'in', 'his', 'spirit', 'that', 'they', 'so',
       'reasoned', 'within', 'themselves', 'he', 'said', 'unto', 'them',
       'why', 'reason', 'ye', 'these', 'things', 'in', 'your', 'hearts',
       '2', '9', 'whether', 'is', 'it', 'easier', 'to', 'say', 'to',
       'the', 'sick', 'of', 'the', 'palsy', 'thy', 'sins', 'be',
       'forgiven', 'thee', 'or', 'to', 'say', 'arise', 'and', 'take',
       'up', 'thy', 'bed', 'and', 'walk', '2', '10', 'but', 'that', 'ye',
       'may', 'know', 'that', 'the', 'son', 'of', 'man', 'hath', 'power',
       'on', 'earth', 'to', 'forgive', 'sins', 'he', 'saith', 'to', 'the',
       'sick', 'of', 'the', 'palsy', '2', '11', 'i', 'say', 'unto',
       'thee', 'arise', 'and', 'take', 'up', 'thy', 'bed', 'and', 'go',
       'thy', 'way', 'into', 'thine', 'house', '']
1
19 - docid=1419
0
1 - Quran to OT
['as', 'for', 'the', 'happy', 'they', 'shall', 'live', 'in',
       'paradise', 'for', 'ever', 'so', 'as', 'long', 'as', 'the',
       'heavens', 'and', 'the', 'earth', 'endure', 'and', 'as', 'your',
       'lord', 'wills', 'an', 'unbroken', 'gift', '']

2
24 - docid=23145
2
1 - NT to OT
['and', 'he', 'said', 'unto', 'her', 'daughter', 'thy', 'faith',
       'hath', 'made', 'thee', 'whole', 'go', 'in', 'peace', 'and', 'be',
       'whole', 'of', 'thy', 'plague', '']

problem spotted:
    1) change preprocessing: add stemming and stopping. #too many uninformative words, need stopping, like 'the', numbers. ignore ''.
    2) feature selection: use top N features with the highest MI scores.
    3) change SVM parameters.
target:
    evaluation on development set should be better than baseline.
    better not to overfit. final goal, better test performance.
    '''

def heavy_preprocess(text):
    tokens = []
    
#    heavier version
#    with open(check_stopword_txt, 'r') as stopperReader:
#        stopper = stopperReader.read().splitlines()
#        stopperReader.close()
#    for token in r.split(text.lower().strip()):
##        if token not in stopper and token != '' and re.match('^[0-9]+$', token) == None:
#        if token not in stopper and token != '':
##        if token != '':
#            stemmed_token = stem(token)
#            tokens.append(stemmed_token)
#            if stemmed_token not in corpustoken2id and stemmed_token in reduced_features_namelist:
#                l = len(corpustoken2id)
#                corpustoken2id[stemmed_token] = l
#                corpusid2token[l] = stemmed_token
    
#    lighter version
    for token in r.split(text.lower().strip()):
        if token != '':
            tokens.append(token)
            if token not in corpustoken2id and token in reduced_features_namelist:
                l = len(corpustoken2id)
                corpustoken2id[token] = l
                corpusid2token[l] = token
    return tokens

def add_classfication_eval_csv(systemname, train_evaluation, devel_evaluation, test_evaluation):
    with open(output_classification_eval_csv, 'a', newline='') as csvfile:
        cl_eval_writer = csv.writer(csvfile, delimiter=',')
        
        line = []
        line.append(systemname)
        line.append('train')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(train_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(train_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)
        
        line = []
        line.append(systemname)
        line.append('dev')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(devel_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(devel_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)
        
        line = []
        line.append(systemname)
        line.append('test')
        for corpus_index in range(3):
            for prf_index in range(3):
                line.append("{:3f}".format(test_evaluation[prf_index][corpus_index]))
        for prf_index in range(3):
            line.append("{:3f}".format(sum(test_evaluation[prf_index])/3))
        cl_eval_writer.writerow(line)

reduced_features_namelist = []    
top_N_features = 3000 
#top_N_features = float('inf')

def CLASSIFICATION_IMPROVED():
#    read corpora_mi_term_score.csv  
    for corpus_index in range(3):
        top_n_index = 0
#        mi_file_name = './' + corpusid2name[corpus_index] + '_mi_term_score.csv'
        mi_file_name = './' + corpusid2name[corpus_index] + '_mi_term_score_lightPRE.csv'
        with open(mi_file_name, newline='') as csvfile:
            mi_reader = csv.reader(csvfile, delimiter=',')
            for row in mi_reader:
                if row[0] not in reduced_features_namelist:
                    reduced_features_namelist.append(row[0])
                top_n_index+=1
                if top_n_index == top_N_features:
                    break
    
    with open(input_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        line_counter = 0
        for row in corpora_reader:
            tokens = heavy_preprocess(row[1])
            docuid2label2tokens[line_counter] = [corpusname2id[row[0]], tokens]
            line_counter+=1
            
    count_matrix = dok_matrix((len(docuid2label2tokens), len(reduced_features_namelist)), dtype=int)
    
    # count matrix
    for docuid in docuid2label2tokens.keys():
        tokens = docuid2label2tokens[docuid][1]
        for token in tokens:
            if token in reduced_features_namelist:
                count_matrix[docuid, corpustoken2id[token]] += 1
                
    whole_id_set = list(docuid2label2tokens.keys())
    random.Random(2020314).shuffle(whole_id_set)
    train_id_set.extend(whole_id_set[:int(0.9*len(docuid2label2tokens.keys()))])
    devel_id_set.extend(whole_id_set[int(0.9*len(docuid2label2tokens.keys())):])
    [train_correct_cato.append(docuid2label2tokens[doc_idx][0]) for doc_idx in train_id_set]
    [devel_correct_cato.append(docuid2label2tokens[doc_idx][0]) for doc_idx in devel_id_set]
    print('training and development set ready')
    train_matrix = count_matrix[train_id_set, :]
    devel_matrix = count_matrix[devel_id_set, :]
    print('fitting model')
    svc_model_improved.fit(train_matrix, train_correct_cato)
    print('making prediction')
    train_predict_cato.extend(svc_model_improved.predict(train_matrix))
    devel_predict_cato.extend(svc_model_improved.predict(devel_matrix))
    train_evaluation, devel_evaluation = evaluate_prediction(train_predict_cato, train_correct_cato), evaluate_prediction(devel_predict_cato, devel_correct_cato)
    
    test_doc_depth = 0
    with open(test_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        for row in corpora_reader:
            tokens = simpler_preprocess(row[1])
            test_correct_cato.append(corpusname2id[row[0]])
            testdocuid2tokens[test_doc_depth] = tokens
            test_doc_depth += 1
    count_test_matrix = dok_matrix((len(testdocuid2tokens), len(corpusid2token)), dtype=int)
    for docuid in testdocuid2tokens.keys():
        tokens = testdocuid2tokens[docuid]
        for token in tokens:
            if token in corpustoken2id.keys():
                count_test_matrix[docuid, corpustoken2id[token]] += 1
    test_predict_cato.extend(svc_model_improved.predict(count_test_matrix))
    test_evaluation = evaluate_prediction(test_predict_cato, test_correct_cato)
    
    add_classfication_eval_csv('improved', train_evaluation, devel_evaluation, test_evaluation)

def NEWsave_corpora_term_score_mi():
    for corpora_index in range(3):
        with open('./'+corpora_names[corpora_index]+'_mi_term_score_lightPRE.csv', 'w', newline='') as csvfile:
            mi_writer = csv.writer(csvfile, delimiter=',')
            for term, score in sorted(corpora_token_score_mi[corpora_names[corpora_index]].items(), key=lambda item:item[1], reverse=True):
                line = []
                line.append(term)
                line.append("{:3f}".format(score))
                mi_writer.writerow(line)

def NEWMI():
    N = len(corpora2documents[corpora_names[0]]) + len(corpora2documents[corpora_names[1]]) + len(corpora2documents[corpora_names[2]])
    for term in bag_of_all_terms.keys():
        # for each term
        for corpora_id in range(3):
            # for each term-corpora(target) pair
            # first 0/1 (not) contain term, second 0/1 (not) in target corpora
            N11 = bag_of_all_terms[term][corpora_names[corpora_id]]
            N01 = len(corpora2documents[corpora_names[corpora_id]]) - N11
            N10 = sum(bag_of_all_terms[term].values()) - N11
            N00 = len(corpora2documents[corpora_names[(corpora_id+1)%3]]) + len(corpora2documents[corpora_names[(corpora_id+2)%3]]) - N10
            N1n = N11+N10
            N0n = N01+N00
            Nn1 = N01+N11
            Nn0 = N00+N10
            score11, score01, score10, score00 = 0, 0, 0, 0
            if N11 != 0:
                score11 = math.log2(N*N11/(N1n*Nn1)) * N11/N
            if N01 != 0:
                score01 = math.log2(N*N01/(N0n*Nn1)) * N01/N
            if N10 != 0:
                score10 = math.log2(N*N10/(N1n*Nn0)) * N10/N
            if N00 != 0:
                score00 = math.log2(N*N00/(N0n*Nn0)) * N00/N
            score = score11+score01+score10+score00
            corpora_token_score_mi[corpora_names[corpora_id]][term] = score
    NEWsave_corpora_term_score_mi()

def light_preprocess_MI(text, corpora):
    '''
    tokenization
    '''
    tokens = []
    for token in r.split(text.lower().strip()):
        if token != ''and  token not in tokens:
            bag_of_all_terms[token][corpora] += 1
        tokens.append(token)
    return tokens

def NEWANALYSIS():
    with open(input_corpora_tsv, newline='') as tsvfile:
        corpora_reader = csv.reader(tsvfile, delimiter='\t')
        for row in corpora_reader:
            corpora2documents[row[0]].append(light_preprocess_MI(row[1], row[0]))
    NEWMI()

def main():
#    EVAL()
#    ANALYSIS()
#    CLASSIFICATION_BASELINE()
#    NEWANALYSIS()
    CLASSIFICATION_IMPROVED()

if __name__ == '__main__':
    main()










