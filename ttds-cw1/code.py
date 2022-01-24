#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:27:53 2020

@author: s2020314
"""

import xml.etree.ElementTree as ET
import string
import re
import numpy as np
import mmap
import shlex
import math
from enum import Enum
from stemming.porter2 import stem
from collections import defaultdict

# set enum type of possible boolean search options
class Binary(Enum):
    AND = 1
    OR = 2
    NULL = 0

# set regular expression ready for skipping all space and punctuations
r = re.compile(r'[\s{}]+'.format(re.escape(string.punctuation)))

# all input files are stored in directory /input/
collection = ET.parse('./input/trec.5000.xml')
stopWordTxt = './input/englishST.txt'
queriesTxt = './input/queries.boolean.txt'
rankedQueriesTxt = './input/queries.ranked.txt'

# use nested dictionary to store positional inverted index
termIndex = defaultdict(lambda: defaultdict(list))
readTermIndex = defaultdict(lambda: defaultdict(list))

# use another dictionary to store document frequency of term
df = defaultdict(int)
readDocumentFre = defaultdict(int)

# all document id stored in list
D = []
readD = []

# in ranked search, set the upper limit of document returned
topN = 150

# all output files are stored in directory /output/
indexTxt = './output/index.txt'
DTxt = './output/D.txt'
resultTxt = './output/results.boolean.txt'
rankedResultTxt = './output/results.ranked.txt'

def preprocess(text):
    ''' input the whole text (headline and text) of a document
        output a list of tokens after tokenisation, removing stopword and Porter stemming
    '''
    tokens = []
    with open(stopWordTxt, 'r') as stopperReader:
        stopper = stopperReader.read().splitlines()
        stopperReader.close()
    for token in r.split(text.strip().lower()):
        if token not in stopper:
             tokens.append(stem(token))
    return tokens
    
def createInvertedIndex(tokens, docno):
    ''' input a list of tokens and its corresponding document id
        save token-docid-position in termIndex
    '''
    index = 0
    for token in tokens:
        termIndex[token][docno].append(str(index))
        index += 1

def countDocumentFrequency():
    ''' count the number of documents each token appeared in
        save token-df in df
    '''
    for token in termIndex:
        df[token] = len(termIndex[token])
        
def outputIndexTxt():
    ''' output inverted index into "index.txt"
    '''
    with open(indexTxt, 'w+') as f:
        f.seek(0)
        for token in df:
            print(token+':'+str(df[token]), end='\n', file=f)
            for doc in termIndex[token]:
                print('\t'+str(doc)+':'+','.join(termIndex[token][doc]), file=f)

def readIndexTxt():
    ''' read inverted index from "index.txt"
    '''
    with open(indexTxt, 'rb') as f:
        # use mmap to map the file into the proces space and thus have direct access to the content via memory
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmapf:
            indextext = mmapf.read()
            # skip is used to check the end of inverted index of each term
            skip = 0
            term = ''
            for lineno, value in enumerate(indextext.decode('utf8').splitlines()):
                if skip:
                    dno, posstr = value.strip().split(':')
                    readTermIndex[term][int(dno)].extend(list(map(int, posstr.split(','))))
                    skip -= 1
                    continue
                term, df = value.split(':', 1)
                readDocumentFre[term] = int(df)
                skip = int(df)
                  
def saveallD():
    ''' save all document id into "D.txt"
    '''
    with open(DTxt, 'w+') as f:
        f.seek(0)
        for doc in D:
            print(doc, end='\n', file=f)

def readallD():
    ''' read all document id from "D.txt"
    '''
    with open(DTxt, 'rb') as f:
        readD.extend(list(map(int, f.read().decode('utf8').splitlines())))
        
def processProximityQuery(q):
    ''' input query context
        return possible doc list
    '''
    proxiDocList = []
    smallert = 2
    # choose the term with smaller relevant document set to start for loop
    try:
        if readDocumentFre[q[1]] < readDocumentFre[q[2]]:   # use smaller loop
            smallert = 1
    except IndexError:
        print('no such term')
    else:
        for docchoice in readTermIndex[q[smallert]]:
            if docchoice not in readTermIndex[q[3-smallert]]:
                continue
            # if dc is shared, check distance
            # choose the term in doc with smaller relevant position set
            smallerp = 2
            if readTermIndex[q[1]][docchoice] < readTermIndex[q[2]][docchoice]:
                smallerp = 1
            # both p represent the current index of position in the position set
            p1 = p2 = 0
            # both p shouldn't overflow the total length of position set
            while p1 < len(readTermIndex[q[smallerp]][docchoice]) and p2 < len(readTermIndex[q[3-smallerp]][docchoice]):
                # check position distance smaller or equal to distance limit
                if abs(readTermIndex[q[3-smallerp]][docchoice][p2]-readTermIndex[q[smallerp]][docchoice][p1]) <= int(q[0]):
                    proxiDocList.append(docchoice)
                    break
                # check both position indexed p
                # if here, distance is definitely not within limit
                # should increment the p with lower position
                if readTermIndex[q[3-smallerp]][docchoice][p2] < readTermIndex[q[smallerp]][docchoice][p1]:
                    p2 += 1
                else:
                    p1 += 1
    return proxiDocList

def phraseSearch(ph):
    ''' input component of phrase
        return phrase search document
    '''
    phDocList = []
    finalDocList = []
    minLengthIndex = 0
    minLength = math.inf
    # find the term with least relevant document in phrase and later for loop according to it
    for index, value in enumerate(ph):
        if len(value) < minLength:
            minLengthIndex = index
            minLength = len(value)
    head = readTermIndex[ph[minLengthIndex]]
    for doc in head:
        # check doc existence first, then check position
        yesFlag = 1
        for word in ph:
            if doc not in readTermIndex[word]:
                yesFlag = 0
                break
        # document which contains all words in phrase
        if yesFlag:
            phDocList.append(doc)
    # check position
    for doc in phDocList:
        yesDoc = 0
        for start in readTermIndex[ph[0]][doc]:
            yesFlag = 1
            for index, word in enumerate(ph):
                if start+index not in readTermIndex[word][doc]:
                    yesFlag = 0
                    break
            if yesFlag:
            # this starting position can do, phrase exists
                yesDoc = 1
                break
        if yesDoc:
            finalDocList.append(doc)
    return finalDocList
    
def notbutinD(notinlist):
    ''' input list of document in which term should not exist
        return doc in all documents other than the list
    '''
    complementaryList = []
    for doc in readD:
        if doc not in notinlist:
            complementaryList.append(doc)
    return complementaryList

def boolSearch(choice, former, latter):
    ''' boolean choice known AND, OR
        return corresponding document id list
    '''
    boolDocList = []
    if choice.name == 'AND':
        for doc in former:
            if doc in latter:
                boolDocList.append(doc)
    elif choice.name == 'OR':
        boolDocList.extend(former)
        for doc in latter:
            if doc not in former:
                boolDocList.append(doc)
    else:
        print('Clash!')
    return boolDocList
    
def writeIntoResults(file, query, answer):
    ''' query(id) has answer(list)
        write into "results.boolean.txt"
    '''
    with open(file, 'a+') as f:
        for doc in answer:
            print(query+','+str(doc), end='\n', file=f)

def processQueries():
    ''' for normal queries
        should preprocess the same e.g. after dissecting NOT, AND, OR
        need to recognize proximity search first, then phrase search, lastly boolean search 
    '''
    with open(queriesTxt, 'r') as queriesReader:
        queries = queriesReader.read().splitlines() 
        queriesReader.close()
    for query in queries:
        # seperate query number and query context
        qno, q = query.split(' ', 1)
        finalDocumentListForQ = []
        if q[0] == '#':
            # if # comes first, it's proximity search
            q = list(filter(None, preprocess(q)))
            finalDocumentListForQ = processProximityQuery(q)
        else:
            # use shlex to protect possible phrase component
            sepq = shlex.split(q)
            # queries could be boolean 
            # use mid to denote the position of AND, OR
            # use booleansChoice to denote which one it is
            # queries could contain NOT
            # use neg to denote position of NOT
            mid = 0
            neg = []
            booleanChoice = Binary.NULL
            # use a list to save already seperated query with AND, OR, NOT replaced by []
            preprocessedList = []
            for index, xseq in enumerate(sepq):
                value = []
                if xseq == 'AND':
                    booleanChoice = Binary.AND
                    mid = index
                elif xseq == 'OR':
                    booleanChoice = Binary.OR
                    mid = index
                elif xseq == 'NOT':
                    neg.append(index)
                else:
                    value = list(filter(None, preprocess(xseq)))
                preprocessedList.append(value)
            allComponentDocList = []
            for component in preprocessedList:
                # check each meaningful component 
                # if it includes more than one word, it's a phrase search
                if component != []:
                    thisList = []
                    if len(component) > 1:
                        thisList = phraseSearch(component)
                    else:
                        thisList = list(readTermIndex[component[0]].keys())
                    allComponentDocList.append(thisList)
            if mid == 0:
                # no boolean
                if neg == []:
                    # no negative, pure one word, check allComponentDocList should be length 1
                    if len(allComponentDocList) != 1:
                        print("Clash!")
                        continue
                    else:
                        finalDocumentListForQ = allComponentDocList[0]
                else:
                    # one negative, one word
                    if len(neg) != 1 or neg[0] != 0:
                        print("Clash!")
                        continue
                    else:
                        finalDocumentListForQ = notbutinD(allComponentDocList[0])
            else:
                # yes boolean
                if len(neg) == 2:
                    if neg[0] != mid-2 or neg[1] != mid+1:
                        print('Clash!')
                        continue
                    else:
                        # handle both neg
                        if sepq[mid-1] == [] or sepq[mid+1] == [] :
                            print('Clash!')
                            continue
                        else:
                            finalDocumentListForQ = boolSearch(booleanChoice, notbutinD(allComponentDocList[0]), notbutinD(allComponentDocList[1]))
                elif len(neg) == 1:
                    if neg[0] == mid-2:
                        # former one neg
                        if sepq[mid-1] == []:
                            print('Clash!')
                            continue
                        else:
                            finalDocumentListForQ = boolSearch(booleanChoice, notbutinD(allComponentDocList[0]), allComponentDocList[1])
                    elif neg[0] == mid+1:
                        # latter one neg
                        if sepq[mid+2] == []:
                            print('Clash!')
                            continue
                        else:
                            finalDocumentListForQ = boolSearch(booleanChoice, allComponentDocList[0], notbutinD(allComponentDocList[1]))
                    else:
                        print('Clash!')
                        continue
                elif len(neg) == 0:
                    # no neg
                    if sepq[mid-1] == [] or sepq[mid+1] == []:
                        print('Clash!')
                        continue
                    else:
                        finalDocumentListForQ = boolSearch(booleanChoice, allComponentDocList[0], allComponentDocList[1])
                else:
                    print('Clash! Can not handle.')
                    continue
        writeIntoResults(resultTxt, qno, sorted(finalDocumentListForQ))
            
def outputRankedResults(qno, rankedResults):
    ''' input query number and ranked results
        write into "results.ranked.txt"
    '''
    with open(rankedResultTxt, 'a') as f:
        f.seek(0)
        index = topN
        for result in rankedResults:
            print(qno+','+str(result[0])+','+'{:.4f}'.format(result[1]), end='\n', file=f)
            index -= 1
            if index == 0:
                break

def processRankedQueries():
    ''' read all ranked queries
        get query number and query context
        same preprocessing
        call f to output worted ranked results to file
    '''
    with open(rankedQueriesTxt, 'r') as reader:
        queries = reader.read().splitlines()
        reader.close()
    for query in queries:
        qno, qcomponent = query.split(' ', 1)
        qwords = preprocess(qcomponent)
        score = defaultdict(float) # dno, score
        # for each word, calculate score for every document and sum up by document
        for word in qwords:
            for doc in readD:
                if doc in readTermIndex[word]:
                    score[doc] += (1+math.log10(len(readTermIndex[word][doc]))) * math.log10( len(readD) / readDocumentFre[word])
                else:
                    continue
        rankedResults = sorted(score.items(), key=lambda x: x[1], reverse=True)
        outputRankedResults(qno, rankedResults)

def main():
    # using xml.etree.ElementTree to parse original document.xml
    root = collection.getroot()
    for doc in root:
        # find content labeled DOCNO, HEADLINE and TEXT
        docID = doc.find('DOCNO').text
        D.append(docID)
        docText = doc.find('HEADLINE').text + ' ' + doc.find('TEXT').text
        # call f to preprocess the whole text
        docTokens = preprocess(docText)
        # call f to add to inverted index for each token
        createInvertedIndex(docTokens, docID)
        # call f to calculate df for each token
        countDocumentFrequency()
        print(docID)
    # call f to output inverted index and document id to file
    outputIndexTxt()
    saveallD()
    # reading inverted index and document id
    readallD()
    readIndexTxt()
    # processing normal queries
    processQueries()
    # processing tfidf ranked queries
    processRankedQueries()
       
if __name__ == "__main__":
    main()
