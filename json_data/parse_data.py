import json
import ijson


def read_data_ijson():
	outputFile = open('parsed_data_all.csv', 'w')
	f = open('wikidata-20190930-all.json', 'r')
	objects = ijson.items(f, 'item')
	columns = ('id', 'descriptions')
	#counter = 0
	fileNum = 0 
	for objectCurrent in objects:
		#counter += 1
		#if counter >= 100:
		#	break 

		if 'id' in objectCurrent and 'descriptions' in objectCurrent:
			ID = objectCurrent['id']
			descriptions = objectCurrent['descriptions']
			for description_lang in descriptions:
				INFO = descriptions[description_lang]
				try:
					LANG = INFO['language']
					VAL = INFO['value']
					outputFile.write(ID + ',' + LANG + ',' + '"' + VAL + '"' + '\n')
				except:
					continue

read_data_ijson()

#obj = json.loads(data)

#print(obj.keys)
