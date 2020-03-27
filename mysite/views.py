import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from django.shortcuts import render
 
def home(request):
	return render(request, 'home.html')
 
def new_page(request):
	text = request.GET['fulltextarea']
	stopWords = set(stopwords.words("english"))
	words = word_tokenize(text)
	ps = PorterStemmer()
	st=".,"
	freqTable = dict()
	for word in words:
		word=word.lower()
		word=ps.stem(word)
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
			w=ps.stem(w)
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
	strio='\n'
	stri=""
	for s in sum_sen:
		stri+=s
	strop='\n'.join([stri,strio,stcSt,stcSu])
	return render(request, 'newpage.html', {'data':strop})
