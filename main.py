import io, os
import re as re
import zipfile as zipfile
import math
from functools import reduce

mytextzip = ''
docList=[]
idx_ID=1
with zipfile.ZipFile('30Columnists.zip') as z:
    for zipinfo in z.infolist():
        mytextzip = ''
        if zipinfo.filename.endswith('.txt') and re.search('raw_texts', zipinfo.filename):
            with z.open(zipinfo) as f:
                textfile = io.TextIOWrapper(f, encoding='cp1254', newline='')
                for line in textfile:
                    if len(line.strip()): 
                        mytextzip += ' ' + line.strip()
                document = {
                    'id': str(idx_ID),
                    'text': mytextzip
                }
                docList.append(document)
                idx_ID+=1


# TOKENIZATION

# Non-breaking to normal space
NON_BREAKING = re.compile(u"\s+"), " "
# Multiple dot
MULTIPLE_DOT = re.compile(u"\.+"), " "
# Merge multiple spaces.
ONE_SPACE = re.compile(r' {2,}'), ' '
# 2.5 -> 2.5 - asd. -> asd . 
DOT_WITHOUT_FLOAT = re.compile("((?<![0-9])[\.])"), r' '
# 2,5 -> 2,5 - asd, -> asd , 
COMMA_WITHOUT_FLOAT = re.compile("((?<![0-9])[,])"), r' '
# doesn't -> doesn't  -  'Something' -> ' Something '
QUOTE_FOR_NOT_S = re.compile("[\']"), r' '
AFTER_QUOTE_SINGLE_S = re.compile("\s+[s]\s+"), r' '
# Extra punctuations "!()
NORMALIZE = re.compile("([\–])"), r'-'
EXTRAS_PUNK = re.compile("([^\'\.\,\w\s\-\–])"), r' '

STOP_WORDS_LIST=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
STOP_WORDS=re.compile(r'\b(?:%s)\b[^\-]' % '|'.join(STOP_WORDS_LIST)),r" "

REGEXES = [
    STOP_WORDS,
    NON_BREAKING,
    MULTIPLE_DOT,
    DOT_WITHOUT_FLOAT,
    COMMA_WITHOUT_FLOAT,
    QUOTE_FOR_NOT_S,
    AFTER_QUOTE_SINGLE_S,
    NORMALIZE,
    EXTRAS_PUNK,
    ONE_SPACE
]

def normalize(sentence):
    sentence = sentence.lower()
    for regexp, subsitution in REGEXES:
        sentence = regexp.sub(subsitution, sentence)   
    return sentence

topWord = 0
for document in docList:
    text = document['text']
    tokenizedText = normalize(text)
    tokens = tokenizedText.split(' ')
    del tokens[0]
    del tokens[len(tokens)-1]
    topWord += len(tokens)
    document['tokens']=tokens

avgDl = topWord / len(docList)

invertedIndex=dict()

def docReducer(i, j):
    for k in j: 
        if type(i.get(k, 0)) != int:
            if i.get(k, 0)[0][1] == j.get(k, 0)[0][1]:
                i[k] = [(i.get(k, 0)[0][0] + j.get(k, 0)[0][0], i.get(k, 0)[0][1])]
        else:
            i[k] = [(j.get(k, 0)[0][0], j.get(k, 0)[0][1])]
    return i

def merge(dict1,dict2):
    for elm2 in dict2:
        if elm2 in dict1:
            dict1[elm2].extend(dict2[elm2])
        else:
            dict1[elm2] = dict2[elm2]
    return dict1

import time
start_time = time.time()

for document in docList:
    docMap = map(lambda char: dict([[char, [(1,document['id'])]]]), document['tokens'])
    singleReducer = reduce(docReducer, docMap)
    invertedIndex = merge(invertedIndex,singleReducer)

elapsed_time = time.time() - start_time

print("Inverted Index: "+str(elapsed_time))



def highlight_term(id, relevance):
    return "--- document {id}\n Relevance Score: {rs} \n ".format(id=id,rs=relevance)

def searchTFIDF(query):
    #Calculate TF*IDF Relevance Score
    results={}
    for word in query:
        if word in invertedIndex.keys():
            occurances=invertedIndex[word]
            idf=math.log2(len(docList)/len(occurances))
            for tf,docId in occurances:
                tf = tf/len(docList[int(docId)-1]['tokens'])
                if docId in results.keys():
                    results[docId]+=idf*tf
                else:
                    results[docId] = idf*tf
    return results

def searchBM25(query):
    #Calculate BM25 Relevance Score
    k = 1.6
    b = 0.75
    results={}
    for word in query:
        if word in invertedIndex.keys():
            occurances=invertedIndex[word]
            idf=math.log(len(docList)/len(occurances))
            for tf,docId in occurances:
                divTop = (tf*(k+1))
                dl = len(docList[int(docId)-1]['tokens'])
                divBot = (tf + (k * (1 - b + (b*(dl/avgDl)))))
                if docId in results.keys():
                    results[docId] += idf*(divTop/divBot)
                else:
                    results[docId] = idf*(divTop/divBot)
    return results

def searchDFI(query):
    #Calculate DFI Relevance Score
    results={}
    for word in query:
        if word in invertedIndex.keys():
            occurances=invertedIndex[word]
            TF = 0
            for tf,docId in occurances:
                TF += tf
            N = topWord
            for tf,docId in occurances:
                dl = len(docList[int(docId)-1]['tokens'])
                e = (TF * dl) / N
                if docId in results.keys():
                    dfi = ( ( tf - e ) / ( math.sqrt(e) ) ) + 1
                    results[docId]+= math.log2( dfi )
                else:
                    dfi = ( ( tf - e ) / ( math.sqrt(e) ) ) + 1
                    results[docId] = math.log2( dfi )
    return results

def searchQuery(query,scoring='tfidf'):
    query = normalize(query)
    query=query.split(' ')
    if scoring=='dfi':
        results = searchDFI(query)
    elif scoring=='bm25':
        results = searchBM25(query)
    else:
        results = searchTFIDF(query)

    #Sorting results for relevance score
    sortedResults={k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    results={}
    resultNum=1
    for key,value in sortedResults.items():
        results[resultNum]={}
        results[resultNum]['id']=resultNum
        results[resultNum]['relevanceScore']=value
        results[resultNum]['docId']=key
        resultNum+=1
    return results

def search(query, scoring='tfidf'):
    print("\n %s Searching..." % (query) )
    start_time = time.time()
    responses=searchQuery(query,scoring)
    elapsed_time = time.time() - start_time
    print(" %06.4fs \n Total Result: %d \n" % (elapsed_time,len(responses)))

    for response in responses.values():
        output=highlight_term(response['docId'],response['relevanceScore'])
        print(output)
        if response['id'] == 10:
            break
    return list(responses.values())[:10]


queryList=["Edinburgh nice place?",
            "elections of edinburgh?",
            "Situation of postwar",
            "Great Britain vs Italy",
            "Healthcare industry changes",
            "Football news",
            "Effects of bank",
            "the best in the sports world",
            "what is COBRA laws",
            "economics in Scotland"]

queryTrues={
    1:['1','576','1145'],
    2:['27','560','567','569','574','577','587','593'],
    3:['51','1391'],
    4:['50','337','472'],
    5:['805','856','881','882','884','895','898'],
    6:['281','1035','1412','1419','1420','1427','1433','1444'],
    7:['437','453','467','499','504','531','627','636','658','686'],
    8:['5','207','307','322','325','346','1142','1440'],
    9:['873','878','880','884','899'],
    10:['7','24','50','551','1126','1202']
}

responses = dict()
for queryID in range(len(queryList)):
    responses[queryID]=dict()
    responses[queryID]['query'] = queryList[queryID]
    responses[queryID]['tfidf'] = search(queryList[queryID],'tfidf')
    responses[queryID]['bm25'] = search(queryList[queryID],'bm25')
    responses[queryID]['dfi'] = search(queryList[queryID],'dfi')

mapDict = {
    'tfidf':0,
    'bm25':0,
    'dfi':0
}

for queryID, results in responses.items():
    for model,result in results.items():
        if model != 'query':
            resCount = 0
            findCount = 0
            sumCount = 0
            for item in result:
                resCount+=1
                if item['docId'] in queryTrues[queryID+1]:
                    findCount+=1
                sumCount+= findCount/resCount
            mapDict[model] += sumCount / 10


print("Model Name   =>   MAP Score \n")
for model in mapDict.keys():
    mapDict[model] = mapDict[model] / 10
    print("{0}  => {1} \n".format(model,mapDict[model]))

mapDict={k: v for k, v in sorted(mapDict.items(), key=lambda item: item[1], reverse=True)}
print("Best Weighting Model => " + next(iter(mapDict)) )

