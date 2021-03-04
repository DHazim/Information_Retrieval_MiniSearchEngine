from itertools import chain
import numpy
import pickle
import sys
import copy

with open("tokdoc.dict",'rb') as f:
	tokdoc = pickle.load(f)

with open("pageRank.dict",'rb') as f:
	pageRankDict = pickle.load(f)



Ntokens = sum(map(len,tokdoc.values()))
docList = list(set(chain(*tokdoc.values())))
Ndocs = len(docList)


print("hey")
tokInfo = dict()
tf = dict()
tfidf = dict()
# Compute the token information and count the token occurrences

for tok in tokdoc:
	tokInfo[tok] = -numpy.log(len(tokdoc[tok])/Ndocs)
	for doc in tokdoc[tok]:
		if not doc in tf:
			tf[doc] = dict()
		tf[doc][tok] = tf[doc].get(tok,0) + 1

# Normalize token occurrences to token frequencies
for doc in tf:
	Ntok = sum(tf[doc].values())
	for tok in tf[doc]:
		tf[doc][tok] /= Ntok

# Compute the TF-IDF
for tok in tokdoc:
	for doc in tokdoc[tok]:
		if not doc in tfidf:
			tfidf[doc] = dict()
		tfidf[doc][tok] = tokInfo[tok]*tf[doc][tok]

#norm2 vector
def normvector(v):
	s=0
	for i in v.values():
		s+=i*i
	return numpy.sqrt(s)



# Scalar product
def scal(query,doc,tfidf):
	s = float()
	for tok in query:
		s +=  tfidf[doc].get(tok,0)*tokInfo[tok] 
	return s


# Ranked by token relevance (vector model)
def getBestResults(queryStr, topN):
	query = queryStr.split(" ")
	searchRes = list(map(lambda d:scal(query,d,tfidf)/(normvector(tfidf[d])*normvector(tokInfo)),docList))
	bestPages = list(reversed([ docList[i] for i in numpy.argsort(searchRes)[-topN:] ]))
	return bestPages

# Page ranking of results
def rankResults(results):
	ranks = [ pageRankDict.get(page,0) for page in results ]
	rankedResults = list(reversed([ results[i] for i in numpy.argsort(ranks) ]))
	return rankedResults


def printResults(rankedResults):
	for idx,page in enumerate(rankedResults):
		print(str(idx) + ". " + page)



query = "evolution of bacteria"
top = 15
results = getBestResults(query,top)
#printResults(results)
rankedResults = rankResults(results)
#printResults(rankedResults)
print("\n\n now \n\n")
for i,page in enumerate(rankedResults):
	if "Bacterial evolution" in page:
		print(page," avec un Rang de: ",i," avec PageRank")
for i,page in enumerate(results):
	if "Bacterial evolution" in page:
		print(page," avec un Rang de: ",i," avec tfidf")

# 7 Page Ranks of DNA & RNA
for i,page in enumerate(rankedResults):
	if "DNA" in page:
		print(page," contenant DNA avec un Rang de: ",i," avec PageRank")
	if "RNA" in page:
		print(page," contenant RNA avec un Rang de: ",i," avec PageRank")
	



