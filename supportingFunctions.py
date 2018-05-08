from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob

def lowerCase(x,logger):
	logger.info('case conversion to lower')
	try:
		return(x.lower())
	except:
		return(x)

def stopWords(x,logger):
	logger.info('Removal of stopwords')
	stopWords = stopwords.words('english')
	try:
		x =[i for i in x.split() if i not in stopWords]
		return(" ".join(x))
	except:
		return(x)

def stemming(x,logger):
	logger.info('Stemminization')
	try:
		st=PorterStemmer()
		x = [st.stem(word) for word in x.split()]
		return(" ".join(x))
	except:
		return(x)

def lemmatization(x,logger):
	logger.info('Lemmatization')
	try:
		x = [Word(word).lemmatize() for word in x.split()]
		return(" ".join(x))
	except:
		return(x)

def sentiments(x,logger):
	logger.info('sentiment analysis')
	try:
		return(TextBlob(x).sentiment)
	except:
		return('Unknown')	