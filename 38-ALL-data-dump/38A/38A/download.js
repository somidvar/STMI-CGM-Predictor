console.log("Hello world!");

var data = source.data;
var filetext = 'Participant,time,BG,Sensor_ID,Meal,Photo,Photo_annotation,medication,med_annotation,blood_analysis\n';
for (var i = 0; i < data['Participant'].length; i++) {
    var currRow = [data['Participant'][i].toString(),
				   data['time'][i].toString(),
                   data['BG'][i].toString(),
				   data['Sensor_ID'][i].toString(),
				   data['Meal'][i].toString(),
				   data['Photo'][i].toString(),
				   data['Photo_annotation'][i].toString(),
				   data['medication'][i].toString(),
				   data['med_annotation'][i].toString(),
                   data['blood_analysis'][i].toString().concat('\n')];

    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = '8hours_after_index.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}