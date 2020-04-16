import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from django.shortcuts import render
 
def home(request):
	return render(request, 'home.html')
 
def new_page(request):
	text = request.GET['fulltextarea']
	stopWords = set(stopwords.words("english"))
	words = word_tokenize(text)
	wordnet_lemmatizer = WordNetLemmatizer()
	st=".,"
	freqTable = dict()
	for word in words:
		word=word.lower()
		word=wordnet_lemmatizer.lemmatize(word)
		if word in stopWords:
			continue
		if word in st:
			continue
		if word in freqTable:
			freqTable[word] += 1
		else:
			freqTable[word] = 1
	mk=max(freqTable, key=freqTable.get)
	maxwt=freqTable[mk]
	for k in freqTable:
		freqTable[k]/=maxwt
	sentences = sent_tokenize(text)
	sent_score=[]
	for s in sentences:
		sum=0
		sen_wor=word_tokenize(s)
		for w in sen_wor:
			w=w.lower()
			w=wordnet_lemmatizer.lemmatize(w)
			if w in stopWords:
				continue
			if w in st:
				continue
			sum+=freqTable[w]
		sent_score.append(sum)
	si=0
	for i in sent_score:
		si+=i
	Avg_wt=si/len(sent_score)
	Summary=[]
	for i in range(len(sent_score)):
		if(sent_score[i]>=Avg_wt):
			Summary.append(sentences[i])
	sum_sen=[]
	for i in Summary:
		if i not in sum_sen:
			sum_sen.append(i)
	stcSt="Number of sentences in Source text:"
	stcSt+=str(len(sentences))
	stcSu="Number of sentences in Summary:"
	stcSu+=str(len(sum_sen))
	stri=""
	stro='\n'
	for s in sum_sen:
		stri+=s
	strop='\n'.join([stri,stro,stcSt,stcSu])
	clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
	clean_sentences = [s.lower() for s in clean_sentences]
	stop_words = stopwords.words('english')
	def remove_stopwords(sen):
		sen_new = " ".join([i for i in sen if i not in stop_words])
		return sen_new
	clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
	vectorizer = TfidfVectorizer(norm = False, smooth_idf = False)
	sentence_vectors = vectorizer.fit_transform(clean_sentences)
	m=sentence_vectors.toarray().shape
	wc=m[1]
	sim_mat = np.zeros([len(clean_sentences), len(clean_sentences)])
	for i in range(len(clean_sentences)):
		for j in range(len(clean_sentences)):
			if i != j:
				sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,wc), sentence_vectors[j].reshape(1,wc))[0,0]
	nx_graph = nx.from_numpy_array(sim_mat)
	scores = nx.pagerank(nx_graph)
	ranked_sentences = [(scores[i],s) for i,s in enumerate(sentences)]
	stro=''
	s=0
	for i in ranked_sentences:
		s+=i[0]
	avg=s/m[0]
	suml=[]
	for i in ranked_sentences:
		if i[0]>=avg and i[1] not in suml:
			suml.append(i[1])
	stcSu1="Number of sentences in Summary:"
	stcSu1+=str(len(suml))
	stri1=""
	stro1='\n'
	for s in suml:
		stri1+=s
	strop1='\n'.join([stri1,stro1,stcSt,stcSu1])
	sumsr='SENTENCE RANKING:'
	sumtr='TEXTRANK:'
	strOP='\n'.join([sumsr,stro1,strop,stro1,stro1,sumtr,stro1,strop1])
	return render(request, 'newpage.html', {'data':strOP})