from pandas import *
from bokeh.plotting import figure, output_file, show
from bokeh.models import *
from bokeh.models.widgets import TextInput, Div, DataTable, TableColumn
from bokeh.plotting import *
import time
from PIL import Image
import os
import math
from bokeh.io import curdoc
from bokeh.layouts import gridplot, widgetbox, column
import numpy as np
from os.path import dirname, join

print dirname(__file__)

#read excel file to dataframe
xls = ExcelFile('all_record_38B.xlsx')
df_original = xls.parse(xls.sheet_names[0])
df = df_original.drop_duplicates(subset=['Participant', 'time', 'meal', 'BG', 'Photo_annotation'])
if pd.Timestamp(2017, 1, 1, 12, 30)  >  df.iloc[200]['time']:
	print df.iloc[200]['time']




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
                src="@excelfile" height="400" alt="@excelfile" width="800" height:auto;
                style="float: none; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
    </div>
"""


#only do 38B
# x/x_food means the timestamp
x=[]
y=[]
x_food=[]
indices_food=[]
imgs=[]
x_food_ann=[]
imgs_ann=[]
x_blood=[]
excelfile=[]
x_food_realtime=[]

for idx, row in df.iterrows():
    if row['Participant'] == '38B':
        
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
			imgs.append('38B/static'+str(row['Photo']))
			if isinstance(row['Photo_annotation'], unicode):
				try:
					imgs_ann.append(row['Photo_annotation'])
				except UnicodeEncodeError:
					print row['Photo_annotation']
			else:
				imgs_ann.append('')

		if isinstance(row['blood_analysis'], unicode) and row['blood_analysis'].endswith('jpg'):
			x_blood.append(row['time'])
			excelfile.append('./'+row['blood_analysis'])
            
            
            
            
        
        
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



p = figure(x_axis_type="datetime", plot_width=1200, plot_height=400, title='participant 38B')






p.add_tools(TapTool())
welcome_message = 'You have selected: (none)'
text_banner = Paragraph(text=welcome_message, width=800, height=100)











def callback_print(text_banner=text_banner):
	user_input = str(cb_obj.x)
	user_input = int(user_input[:10])
	welcome_message = 'You have selected: ' + str(user_input + 18000)
	
	text_banner.text = welcome_message
	
	
	
	
	
	
	
	
	
	
	
	
	
callback = CustomJS.from_py_func(callback_print)

p.js_on_event('tap', callback)
	
	
	
'''
taptool = p.select(type=TapTool)
taptool.callback = CustomJS.from_py_func(callback_print)
'''
















p.line(x, y, line_dash='solid', legend='glucose value')

#only show food when hover, so create a 
source = ColumnDataSource(data=dict(x_food=x_food, y_food=[100]*len(x_food), imgs=imgs, imgs_ann=imgs_ann, indices_food=indices_food))
source_2 = ColumnDataSource(data=dict(x_blood=x_blood, y_blood=[200]*len(x_blood), excelfile=excelfile))

glyph_scatter = p.circle(x='x_food', y='y_food', size=5, color='red', source=source, fill_alpha=0.001, line_alpha=1, legend='food pic')

glyph_blood = p.diamond(x='x_blood', y='y_blood', color='green', source=source_2, legend='blood')


V_bar = p.vbar(x='x_food', top=220, bottom=0, width=0.1, fill_color="red", line_color='red', line_dash='dashed', legend='food', line_alpha=0.2, source=source)


HT = HoverTool(tooltips=TOOLTIPS, renderers=[glyph_scatter], mode='vline', formatters={'$x': 'datetime'})
HT_2 = HoverTool(tooltips=TOOLTIPS_2, renderers=[glyph_blood])





drag_tool = BoxEditTool(renderers=[glyph_scatter], empty_value=1)
p.add_tools(drag_tool)
p.toolbar.active_drag = drag_tool



'''
url='https://www.google.com'
taptool = p.select(type=TapTool, renderers=[glyph_blood])
#taptool.renderers.append(glyph_blood)
taptool.callback = OpenURL(url=url)
'''

p.tools.append(HT)
p.tools.append(HT_2)
p.legend.click_policy="hide"


#show(p)
widg = widgetbox(text_banner)
#grid=gridplot([[p],[widg]])
#show(grid)












#add a text input box
'''
# PREP DATA
welcome_message = 'You have selected: (none)'

# TAKE ONLY OUTPUT
text_banner = Paragraph(text=welcome_message, width=200, height=100)

# CALLBACKS
def callback_print(text_banner=text_banner):
	user_input = str(cb_obj.value)
	welcome_message = 'You have selected: ' + user_input

	text_banner.text = welcome_message

# USER INTERACTIONS
text_input = TextInput(value="", title="Enter row number:",             
callback=CustomJS.from_py_func(callback_print))

# LAYOUT
widg = widgetbox(text_input, text_banner)
#show(widg)

    
grid=gridplot([[p],[widg]])
show(grid)
'''



curdoc().add_root(column(p, widg))