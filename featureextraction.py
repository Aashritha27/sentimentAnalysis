import pandas as pd
import yaml
import logging.config
from supportingFunctions import wordCount,averageWord,specialCharacters,upperCase,infoProduct



class featureExtraction:
    def __init__(self,data,logger):
        self.Data =  pd.read_csv(data,low_memory=False)
        self.logger = logger
    
    def numberWordCount(self):
        try:
            self.logger.info('number of words')
            self.Data['wordcount'] = self.Data['contents'].apply(wordCount,args=(self.logger,)) 
            self.Data['avgword'] = self.Data['contents'].apply(averageWord,args=(self.logger,))
            self.Data['specialcharacter'] = self.Data['contents'].apply(specialCharacters,args=(self.logger,))
            self.Data['uppercase'] = self.Data['contents'].apply(upperCase,args=(self.logger,))
            self.Data['product'] = self.Data['contents'].apply(infoProduct,args=(self.logger,))
            self.Data.to_csv('feature.csv',sep=',')
            
        except Exception as e:
            self.logger.error('Error {}'.format(e),exc_info=True)
	   
       
if __name__ == '__main__':
    with open('config.yml','rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger('sentimentAnalysis')
    feature = featureExtraction('merged_twitter_data_final.csv',logger)
    feature.numberWordCount()