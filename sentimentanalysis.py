import pandas as pd
import yaml
import nltk
import time
import logging.config
from supportingFunctions import lowerCase,stopWords,stemming,lemmatization,sentiments

nltk.download('wordnet')

class sentimentAnalysis(object):
	def __init__(self,Data,logger):
		self.Data = pd.read_csv(Data,low_memory=False)
		self.logger = logger

	def basicProcessing(self):
		self.logger.info('Processing of Data for cleaning and exploration')
		try:
			self.Data['contents'] = self.Data['contents'].apply(lowerCase,args=(self.logger,))
			self.Data['contents'] = self.Data['contents'].str.replace('[^\w\s]','')
			self.Data['contents'] = self.Data['contents'].apply(stopWords,args=(self.logger,))
			self.Data['contents'] = self.Data['contents'].apply(stemming,args=(self.logger,))
			self.Data['contents'] = self.Data['contents'].apply(lemmatization,args=(self.logger,))
		
		except Exception as e:
			self.logger.error('Cannot preprocess the file {}'.format(e),exc_info=True)
	
	def sentAnalysis(self):
		self.logger.info('sentiment Analysis')
		try:
			self.Data['sentimentPercentage'] = self.Data['contents'].apply(sentiments,args=(self.logger,))
			self.Data['sentiments'] = self.Data['sentimentPercentage].apply(sentimentReviews,args=(self.logger,))
			self.Data.to_csv('Sentiment.csv',sep=',')
		except Exception as e:
			self.logger.error('cannot define sentiment {}'.format(e),exc_info=True)
		


if __name__ == '__main__':
	with open('config.yml','rt') as f:
		config = yaml.safe_load(f.read())
	logging.config.dictConfig(config)
	logger = logging.getLogger('sentimentAnalysis')
	analysis =sentimentAnalysis('merged_twitter_data_final.csv',logger)
	startTime = time.time()
	analysis.basicProcessing()
	analysis.sentAnalysis()
	endTime = time.time()
