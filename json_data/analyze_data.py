import json
import pandas as pd
import numpy as np
import os 
import re
import ijson
import sys
from itertools import combinations


def getPairs(df, sourceLang, targetLang):
	source = df[(df['Language'] == sourceLang)]
	source.columns = ['ID', 'Source_Language', 'Source_Value']
	target = df[(df['Language'] == targetLang)]
	target.columns = ['ID', 'Target_Language', 'Target_Value']

	joined = source.join(target.set_index('ID'), on='ID')
	joined.to_csv('{0}2{1}.csv'.format(sourceLang, targetLang), index=False)

if __name__=="__main__":
	inputFile = sys.argv[1]
	df = pd.read_csv(inputFile, names = ['ID', 'language', 'value'])
	allLanguages = df.language.unique()
	
	comb = combinations(allLanguages, 2)
	for source, target in list(comb):
		
		print(source, target)
	
	#getPairs(inputFile, 'en', 'es')



