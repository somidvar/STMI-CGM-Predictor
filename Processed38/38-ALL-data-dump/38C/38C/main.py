from pandas import *
from bokeh.plotting import figure, output_file, show
from bokeh.models import *
from bokeh.models.widgets import TextInput, Div, DataTable, TableColumn
from bokeh.plotting import *
import time
#from PIL import Image
import os
import math
from bokeh.io import curdoc
from bokeh.layouts import gridplot, widgetbox, column, row
import numpy as np
from os.path import dirname, join  
from xlrd import open_workbook
import sys
reload(sys)
sys.setdefaultencoding('UTF8')


Participant_ID = dirname(__file__)[-3:]


#read excel file to dataframe
xls = ExcelFile('%s/all_record_%s.xlsx' %(Participant_ID, Participant_ID))
df_original = xls.parse(xls.sheet_names[0])
df = df_original.drop_duplicates(subset=['Participant', 'time', 'Sensor_ID', 'Meal', 'BG', 'Photo_annotation', 'blood_analysis'])


'''
#new df only keep the last of duplicate photo
all_photo = []
deleted_photo_indices = []
for idx, item in enumerate(reversed(df['Photo'])):
	print idx, item
	if (item in all_photo) and (type(item)=='str'):
		deleted_photo_indices.append(idx)
	all_photo.append(item)
df = df.drop(deleted_photo_indices)
'''


#sort by time
df = df.sort_values(by=['time'])


#print len(df)
#if pd.Timestamp(2017, 1, 1, 12, 30)  >  df.iloc[200]['time']:
#	print df.iloc[200]['time']






#tooltips for hovertool
TOOLTIPS = """
    <div min-height: 100px; overflow: hidden;>
        <div>
            <img
                src="@imgs" height="200" alt="@imgs" width="200" height:auto;
                style="float: none; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
		<div>
			<p>index: @indices_food, timestamp:  $x{%F %T}</p>
		</div>
		<div>
			<p>@imgs_ann</p>
		</div>
    </div>
    
<script>
// When the user clicks on div, open the popup
function myFunction() {
    var popup = document.getElementById("myPopup");
    popup.classList.toggle("show");
}
</script>
    
"""


TOOLTIPS_2 = """
    <div min-height: 100px; overflow: hidden;>
        <div>
            <img
                src="@excelfile" height="150" alt="@excelfile" width="800" height:auto;
                style="float: none; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
		<div>
			<p>index: @indices_blood, timestamp:  $x{%F %T}</p>
		</div>
    </div>
"""





#create data vectors for figure

# x/x_food means the timestamp
x=[]
y=[]
x_food=[]
indices_food=[]
indices_blood=[]
imgs=[]
x_food_ann=[]
imgs_ann=[]
x_blood=[]
excelfile=[]
x_food_realtime=[]

for idx, row in df.iterrows():
    if row['Participant'] == Participant_ID:
        
		if idx == 0:
			last_time = row['time'] - - pd.Timedelta('0 days 00:15:00')

		if not math.isnan(row['BG']):

			if ((row['time'] - last_time) <= pd.Timedelta('0 days 00:15:00')):

				x.append(row['time'])
				y.append(row['BG'])
			
			else:
				x.append(row['time'] - pd.Timedelta('0 days 00:15:00'))
				y.append(float('nan'))

			last_time = row['time']
			
	
		if isinstance(row['Photo'], unicode) and row['Photo'].endswith('jpg') and row['Meal'] != 0 and row['Meal'] != 2:
			x_food.append(row['time'])
			x_food_realtime.append(str(row['time']))
			indices_food.append(idx)
			#imgs.append('.'+str(row['Photo']))
			imgs.append('%s/static' %(Participant_ID) +str(row['Photo']))
			if isinstance(row['Photo_annotation'], unicode):
				try:
					imgs_ann.append(row['Photo_annotation'])
				except UnicodeEncodeError:
					print row['Photo_annotation']
			else:
				imgs_ann.append('')

		if isinstance(row['blood_analysis'], unicode) and row['blood_analysis'].endswith('jpg'):
			indices_blood.append(idx)
			x_blood.append(row['time'])
			excelfile.append('./%s/static/' %(Participant_ID) + row['blood_analysis'])
            
            
            
            
        
        
            #print x_food[-1]
            #img=Image.open('.'+str(row['Photo']))
            #img.show()
            #x_food.append(time.mktime(row['time'].timetuple())*1000)
        
            
		'''
		if isinstance(row['Photo_annotation'], unicode) and row['Photo_annotation'].endswith('jpg'):
			x_food.append(row['time'])
			imgs.append('.'+row['Photo_annotation'])
		'''    


output_file("CGM.html")






#configure figure 
p = figure(x_axis_type="datetime", plot_width=1200, plot_height=400, title='participant %s' %(Participant_ID))


p.add_tools(TapTool())

welcome_message = 'You have selected: (none)'
text_banner = Paragraph(text=welcome_message, width=1000, height=100)



def callback_print(text_banner=text_banner):
	user_input = str(cb_obj.x)
	user_input = int(user_input[:10])
	welcome_message = 'You have selected: ' + (str(user_input + 18000)).strftime("%Y-%m-%d %H:%M:%S")
	#+ time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(str(user_input + 18000)))
	
	#text_banner.text = welcome_message

	
def callback(div, attributes=[], style='float:left;clear:left;font_size=10pt'):
	return CustomJS(args=dict(div=div), code="""
			
			var datetimestamp = new Date(cb_obj.x + 18000000);
			
			var formatted = datetimestamp.getFullYear()+"-"+(datetimestamp.getMonth()+1)+"-"+datetimestamp.getDate()+" "+datetimestamp.getHours()+":"+datetimestamp.getMinutes();
			
			div.text="timestamp of mouse click: " + formatted;
			
	""")

div = Div(text='timestamp of mouse click: (none)', width=1000)
p.js_on_event('tap', callback(div))











#create different types of glyph 
p.line(x, y, line_dash='solid', legend='glucose value')


source = ColumnDataSource(data=dict(x_food=x_food, y_food=[100]*len(x_food), imgs=imgs, imgs_ann=imgs_ann, indices_food=indices_food))
source_2 = ColumnDataSource(data=dict(x_blood=x_blood, y_blood=[200]*len(x_blood), excelfile=excelfile, indices_blood=indices_blood))

glyph_scatter = p.circle(x='x_food', y='y_food', size=5, color='red', source=source, fill_alpha=0.001, line_alpha=0.9, legend='food pic')



def handler(attr, old, new):
	text_banner.text = 'You have selected n-th food picture n=' + str(new.indices)


glyph_scatter.data_source.on_change('selected', handler)



glyph_blood = p.diamond(x='x_blood', y='y_blood', color='green', source=source_2, legend='blood')


V_bar = p.vbar(x='x_food', top=220, bottom=0, width=0.1, fill_color="red", line_color='red', line_dash='dashed', legend='food', line_alpha=0.15, source=source)









#add tools to figure 
HT = HoverTool(tooltips=TOOLTIPS, renderers=[glyph_scatter], mode='vline', formatters={'$x': 'datetime'})
HT_2 = HoverTool(tooltips=TOOLTIPS_2, renderers=[glyph_blood], formatters={'$x': 'datetime'})


drag_tool = BoxEditTool(renderers=[glyph_scatter], empty_value=1)
p.add_tools(drag_tool)
p.toolbar.active_drag = drag_tool

p.tools.append(HT)
p.tools.append(HT_2)
p.legend.click_policy="hide"







#add widgets to figure

source_table = ColumnDataSource(data=dict())
def update_show():

	row_todelete = text_input_show.value
	
	current = df[(df.index.values >= int(row_todelete)) & (df.time <= (df.iloc[int(row_todelete)]['time'] + pd.Timedelta('0 days 08:10:00')))]
	
	string_dates = []
	for i in current['time']: string_dates.append(i.strftime("%Y-%m-%d %H:%M:%S"))
	current['time'] = string_dates
	
	source_table.data = {
		'Participant'             : current.Participant,
		'time'           : current.time,
		'BG' : current.BG,
		'Sensor_ID' : current.Sensor_ID,
		'Meal'  :  current.Meal,
		'Photo'  : current.Photo,
		'Photo_annotation'  : current.Photo_annotation,
		'medication' : current.medication,
		'med_annotation'   : current.med_annotation,
		'blood_analysis'  : current.blood_analysis
		}
		
text_input_show = TextInput(value="", title="Enter index:")
text_input_show.on_change('value', lambda attr, old, new: update_show())


datefmt = DateFormatter(format="%m/%d/%Y %H:%M:%S")

columns = [
    TableColumn(field="Participant", title="Participant"),
    TableColumn(field="time", title="time"),
    TableColumn(field="BG", title="BG"),
	TableColumn(field="Sensor_ID", title="Sensor_ID"),
	TableColumn(field="Meal", title="Meal"),
	TableColumn(field="Photo", title="Photo"),
	TableColumn(field="Photo_annotation", title="Photo_annotation"),
	TableColumn(field="medication", title="medication"),
	TableColumn(field="med_annotation", title="med_annotation"),
	TableColumn(field="blood_analysis", title="blood_analysis")
		
]

data_table = DataTable(source=source_table, columns=columns, width=1600, height=650, editable=True)
table = widgetbox(data_table)


def add_To_excel():
	'''
	added_row=df.iloc[int(text_input_show.value)]
	added_row.iat[0, 'time'] = text_input_addToexcel.value
	
	df_droppedOld = df_added.drop([int(text_input_show.value)])
	
	df_added = df.append(added_row)
	'''
	
	df_copy = df
	df_copy.at[int(text_input_show.value), 'time'] = pd.Timestamp(text_input_addToexcel.value)
	ew = pd.ExcelWriter("%s/all_record_%s(new).xlsx"%(os.path.dirname(__file__), Participant_ID), options={'encoding':'utf-8'})
	df_copy.to_excel(ew, index=False)
	ew.save()
	
text_input_addToexcel = TextInput(value="", title="Enter the timestamp followed the format shown above (when hit Enter, it will instantly change the timestamp and render a new Excel)")
#text_input_addToexcel.on_change('value', lambda attr, old, new: add_To_excel())


def extract_8hrs_data():
	df_8hrs = df[(df.time >= df.iloc[row_todelete]['time'])]
	ew_2 = pd.ExcelWriter("%s/all_record_%s(%s_8hours).xlsx"%(os.path.dirname(__file__), Participant_ID, row_todelete), options={'encoding':'utf-8'})
	df_8hrs.to_excel(ew_2, index=False)
	ew_2.save

change_timestamp_button = Button(label="Enter", button_type="success")
change_timestamp_button.on_click(add_To_excel)

extract_8hrs_button =  Button(label="Extract 8 hours data after this index", button_type="success")
extract_8hrs_button.callback = CustomJS(args=dict(source=source_table), code=open(join(dirname(__file__), "download.js")).read())






#show(p)
widg = widgetbox(text_banner, text_input_show, extract_8hrs_button, text_input_addToexcel, change_timestamp_button)
#grid=gridplot([[p],[widg]])
#show(grid)


curdoc().add_root(column([p, div, widg, table]))