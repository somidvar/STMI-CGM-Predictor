import pandas as pd
from pandas import ExcelFile
import math

#delete duplicate blood files + delete all nan except 'time'


xls = ExcelFile('all_record_38A(oldest_original).xlsx')

df = xls.parse(xls.sheet_names[0])

df_copy = df

df_copy = df_copy.drop_duplicates()

blood_files = []

for idx, row in df.iterrows():
	

	if type(row['blood_analysis']) == unicode:

		#print row['blood_analysis']
	
	
		#first blood file
		if row['blood_analysis'] not in blood_files:
			
			blood_files.append(row['blood_analysis'])
		
			df_copy.loc[idx, 'blood_analysis'] = df_copy.loc[idx, 'blood_analysis'][:-4] + 'jpg'
			
		#later blood file
		else: 
			
			df_copy.loc[idx, 'blood_analysis'] = blank
			
			df_copy.loc[idx, 'Participant'] = blank
			
			df_copy.loc[idx, 'time'] = blank
			
	else:
		blank = row['blood_analysis']

			
			
df_copy_droped = df_copy.drop_duplicates(subset=['Participant', 'time', 'Sensor_ID', 'Meal', 'Photo_annotation', 'medication', 'blood_analysis'])
			
			
ew = pd.ExcelWriter('all_record_38A(deleted).xlsx', options={'encoding':'utf-8'})

df_copy_droped.to_excel(ew, index=False)

ew.save()