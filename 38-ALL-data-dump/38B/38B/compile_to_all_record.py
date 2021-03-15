import os
import datetime
from pandas import ExcelFile
from shutil import copyfile
from pyexcel.cookbook import merge_all_to_a_book
import glob
import pandas as pd



dirpath = os.getcwd()

Participant_ID = dirpath[-3:]

files = os.listdir(dirpath)

f1=open('./overall_spread_sheet_%s.csv' %dirpath[-3:], 'w')

#headers
f1.write('Participant,time,BG,Sensor_ID,Meal,Photo,Photo_annotation,medication,med_annotation,blood_analysis\n')


##############################################
#get BG from TXT
for file in files:
    
    if file.endswith('txt'):
        
        with open(file, 'r') as f0:
            
            for line in f0:
                
                one_reading=line.split()
                
				#read every thing but the first line
                if len(one_reading) == 5:
                    
                    #patient
                    f1.write('%s,'%(Participant_ID))
                    
                    #date time
                    if one_reading[1].startswith('2018'):
                    
                        f1.write(datetime.datetime.strptime(one_reading[1]+'  '+one_reading[2], '%Y/%m/%d %H:%M').strftime('%Y/%m/%d  %H:%M:%S').replace('06','6').replace('07','7').replace('/0','/').replace(' 0',' ')+',')

                    #BG
                    f1.write(one_reading[-1]+',')
                    
                    #Sensor ID
                    f1.write(file[-5])
                    
                    #the rest
                    f1.write((','*6)+'\n')
                   
 
############################################## 
'''
#get photo from whatsapp      

datetime_pic_dict={}
            
for file in files:
    
    if file.endswith('backup'):
        
        with open(file+'\WhatsApp Chat with 38B.txt', 'r') as f2:
        
            for line in f2:
            
                line = line.split(' - ')
                
                #if pic path
                if len(line) > 1 and line[-1].endswith('(file attached)\n'):
                
                    date_time = line[0].replace(',', '').replace('/18 ', '/2018 ')
                
                    if date_time in datetime_pic_dict:
                    
                        #minute + 1
                        date_time = datetime.datetime.strptime(date_time, '%m/%d/%Y %I:%M %p') + datetime.timedelta(seconds=60)
                        
                        date_time = date_time.strftime('%m/%d/%Y  %I:%M %p').replace('06','6').replace('07','7').replace('/0','/').replace(' 0',' ')
                        
                        datetime_pic_dict[date_time] = ['/' + file + '/' + line[1].replace('38B: ', '').replace(' (file attached)\n', '')]

                    else:
                        
                        datetime_pic_dict[date_time] = ['/' + file + '/'  + line[1].replace('38B: ', '').replace(' (file attached)\n', '')]
                
                #else is annotation
                else:
                
                    try:
                        datetime_pic_dict[date_time].append(line[0].replace('\n', '').replace(',', ''))
                        
                    except NameError:
                        
                        print line
                        
                    
for date_time in datetime_pic_dict:
    
    #patient
    f1.write('38B,')
    
    #date time
    f1.write(date_time)
    
    #the rest
    f1.write(','*4)
    
    #food pic
    f1.write(datetime_pic_dict[date_time][0] + ',')
    
    #ann
    try:
        f1.write(datetime_pic_dict[date_time][1] + ',')
    except:
        f1.write(',')
    
    #the rest
    f1.write(',,\n')
'''
##############################################    
#photo of pre/post will be distinguished
non_pre_meal=['IMG-20180620-WA0000.jpg', 'IMG-20180620-WA0001.jpg', 'IMG-20180620-WA0002.jpg', 'IMG-20180620-WA0003.jpg', 'IMG-20180621-WA0001.jpg', 'IMG-20180621-WA0008.jpg', 'IMG-20180629-WA0000.jpg', '20180714-WA0003.jpg', 'IMG-20180620-WA0008.jpg', 'IMG-20180620-WA0009.jpg', 'IMG-20180621-WA0009.jpg', 'IMG-20180622-WA0002.jpg', 'IMG-20180629-WA0003.jpg']

#get photo 

datetime_pic_dict={}
            
for file in files:
    
    if file.endswith('backup') or file.endswith('complete thread'):
        
        with open(file+'\WhatsApp Chat with %s.txt' %(Participant_ID), 'r') as f2:
        
			for line in f2:

				line = line.split(' - ')
				
				#if pic path
				if len(line) > 1 and line[-1].endswith('(file attached)\n'):
				
					f1.write('\n')
					
					#patient 
					f1.write('%s,' %(Participant_ID))

					#date time
					date_time = line[0].replace(',', '').replace('/18 ', '/2018 ')
					date_time = datetime.datetime.strptime(date_time, '%m/%d/%Y %I:%M %p')
					date_time = date_time.strftime('%m/%d/%Y  %I:%M %p').replace('06','6').replace('07','7').replace('/0','/').replace(' 0',' ')
					f1.write(date_time)
					
					#the rest
					f1.write(','*3)

					#pre/post meal
					if line[1].replace('%s: ' %(Participant_ID), '').replace(' (file attached)\n', '') in non_pre_meal:
						f1.write('2,')
					else:
						f1.write(',')
					
					#food pic
					f1.write('/' + file + '/' + line[1].replace('%s: ' %(Participant_ID), '').replace(' (file attached)\n', '') + ',') 
					
				#else is annotation (two types)
				else:
				
					#annotation that don't know where it belongs
					if len(line) > 1:
					
						f1.write('\n')
                    
						#patient 
						f1.write('%s,' %(Participant_ID))
		
						#date time
						date_time = line[0].replace(',', '').replace('/18 ', '/2018 ')
						date_time = datetime.datetime.strptime(date_time, '%m/%d/%Y %I:%M %p')
						date_time = date_time.strftime('%m/%d/%Y  %I:%M %p').replace('06','6').replace('07','7').replace('/0','/').replace(' 0',' ')
						f1.write(date_time)
						
						#the rest
						f1.write(','*5)
					
					
						f1.write(line[1].replace('\n', '').replace(',', ''))
                    
						f1.write(',,,')
					
					#annotation that belongs to last image
					else:
						f1.write(line[0].replace('\n', '').replace(',', ''))
                    
						f1.write(',,,')
                    
f1.write('\n')
                    
                    

##############################################
#get blood analysis

for file in files:

	if file.startswith('38-Macro-Result') and file.endswith('xlsx'):

		#read excel file to dataframe
		xls = ExcelFile(file)
		df = xls.parse(xls.sheet_names[0])
		
		for idx, row in df.iterrows():
		
			#patient
			f1.write('%s,'%(Participant_ID))
			
			#date time
			f1.write(str(row['Date'])[:-8].replace('-', '/')+' '+str(row['RealTime']))
			
			#the rest
			f1.write(','*8)
			
			#blood analysis
			f1.write(file[:-4]+'jpg')
			
			f1.write('\n')
			
			
			break


f1.close()
	
	
# delete BG duplicates (same time, different values) + duplicate photos (different folder, same photo)
df = pd.read_csv('./overall_spread_sheet_%s.csv' %dirpath[-3:])

df=df.drop_duplicates(subset=['Participant', 'time', 'Sensor_ID', 'Meal', 'Photo_annotation', 'medication', 'blood_analysis'])


df.to_csv('./overall_spread_sheet_%s.csv' %dirpath[-3:], index=False)


'''#still need to mannually copy'''
#merge_all_to_a_book(glob.glob('./overall_spread_sheet_%s.csv' %dirpath[-3:]), 'all_record_%s.xlsx' %dirpath[-3:])
			
			
			
			
			
