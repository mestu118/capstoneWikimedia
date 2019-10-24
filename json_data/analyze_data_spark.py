import json
import pandas as pd
import numpy as np
import os 
import re
import ijson
import sys 
from pyspark.sql import SparkSession
from itertools import combinations 
from pyspark.sql import Row




def getPairs(df, sourceLang, targetLang):
	errors = open('errors_pairs.csv', 'w')
	try:
		source = df.filter(df.language == sourceLang)
		target = df.filter(df.language == targetLang)
		source = source.toDF(*['ID', 'source_language', 'source_value'])
		target = target.toDF(*['ID', 'target_language', 'target_value'])
		
		result = source.join(target, ['ID'])
		result.repartition(1)\
			.write\
			.format("com.databricks.spark.csv")\
			.option("header", "true")\
			.csv("allPairs/{0}2{1}.csv".format(sourceLang, targetLang))
		print("Created file for source: {0}, target: {1}".format(sourceLang, targetLang))

	except:
		errors.write("{0},{1}\n".format(sourceLang, targetLang))
		print("error creating file for source: {0} target: {1}".format(sourceLang, targetLang))
 

def converToList(allLangs):
	"""
	Code assumes allLangs is a list of Row objects
	"""
	retVal = []
	for row in allLangs:
		retVal.append(row.language)
	return retVal


if __name__=="__main__":
	spark = SparkSession.builder.appName('getPairs').getOrCreate()
	inputFile = sys.argv[1]
	df = spark.read.csv(inputFile, schema ='id STRING, language STRING, value STRING')
	df.createOrReplaceTempView('data')
	allLanguages = df.select('language').distinct().collect()
	allLanguages = converToList(allLanguages)
	
	comb = combinations(allLanguages, 2)
	for source, target in list(comb):
		print(source, target)
		getPairs(df, source, target) 
