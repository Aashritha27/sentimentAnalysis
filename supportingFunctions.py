from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from textblob import Word
from textblob import TextBlob
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
		return(TextBlob(x).sentiment[0])
	except:
		return('Unknown')	

def sentimentReviews(x,logger):
	logger.info('reviews')
	try:
		if x > 0.5:
			return('Positive')
		elif x < -0.5:
			return('negative')
		else:
			return('Unknown')
	except:
		return(x)
	
def wordCount(x,logger):
	logger.info('counting the words')
	try:
		x = x.split(" ")
		return(len(x))
	except:
		return(0)
	
def averageWord(x,logger):
	logger.info('avarage of the words')
	try:
         words = x.split()
         return(sum(len(word) for word in words)/len(words))
	except:
		return(0)
       
def specialCharacters(x,logger):
	logger.info('special characters')
	try:
		x = [i for i in x.split(" ") if i.startswith('#')]
		return(len(x))
	except:
		return(0)
    
def upperCase(x,logger):
	logger.info('upper case')
	try:
		x = [i for i in x.split() if i.isupper()]
		return(len(x))
	except:
		return(0)
    
def infoProduct(x,logger):
	logger.info('product')
	try:
		wordToken = nltk.word_tokenize(x)
		posTag = nltk.pos_tag(wordToken)
		productInfo = [key for key,value in posTag if value =='NN']
		return(",".join(productInfo))
	except Exception as e:
		logger.info('enter product Exception {}'.format(e))
		return('No procuct')
    
	

	
