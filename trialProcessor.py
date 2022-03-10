# %%
from datetime import datetime, timedelta, timezone
from dateutil.tz import tzutc
from dateutil.relativedelta import relativedelta
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from scipy.stats import pearsonr, mode, skew, kurtosis, linregress
import xgboost as xgb
from joblib import Parallel, delayed
import sklearn
import pickle
from sklearn.metrics import mean_squared_error, plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import zipfile
import warnings
import seaborn as sns
import sys
from copy import deepcopy

# sns.set_style("white")


plt.rcParams["text.usetex"] = True
font = {"family": "normal", "weight": "bold", "size": 22}

plt.rc("font", **font)

CM_LAG_CORRECTION = [
    ("p1", timedelta(minutes=2 * 60 + 36)),
    ("p3", timedelta(minutes=2 * 60)),
    ("p5", timedelta(minutes=-360)),
    ("p6", timedelta(minutes=-360)),
    ("p7", timedelta(minutes=-360)),
    ("p8", timedelta(minutes=-360)),
    # ("p8", datetime.strptime("02 02 2022-00:00:00", "%m %d %Y-%H:%M:%S"),datetime.strptime("02 06 2022-07:20:00", "%m %d %Y-%H:%M:%S"),timedelta(minutes=-165)),
    # ("p8", datetime.strptime("02 06 2022-07:20:00", "%m %d %Y-%H:%M:%S"),datetime.strptime("02 09 2022-00:00:00", "%m %d %Y-%H:%M:%S"),timedelta(minutes=-165-190)),
    # ("p8", datetime.strptime("02 09 2022-07:20:00", "%m %d %Y-%H:%M:%S"),datetime.strptime("02 14 2022-00:00:00", "%m %d %Y-%H:%M:%S"),timedelta(minutes=93)),
]

CGM_LAG_IMPOSING = timedelta(minutes=int(sys.argv[1]))# ONLY TO IMPOSE A TIME LAG BETWEEN CGM READINGS AND CORE MOTION DATA!!!! WATCH OUT AND USE IT CAUTIOUSLY
OUTTER_WINDOW_LENGTH = timedelta(minutes=int(sys.argv[2]))
print("Staring a new round##################",OUTTER_WINDOW_LENGTH)
# CGM_LAG_IMPOSING = timedelta(minutes=30)# ONLY TO IMPOSE A TIME LAG BETWEEN CGM READINGS AND CORE MOTION DATA!!!! WATCH OUT AND USE IT CAUTIOUSLY
# OUTTER_WINDOW_LENGTH = timedelta(minutes=45)
FASTING_LENGTH = timedelta(minutes=30)
BIG_MEAL_CALORIE = 200
FOLD_NUMBER = 5
INNER_WINDOW_LENGTH = timedelta(seconds=60)
MINIMUM_POINT = INNER_WINDOW_LENGTH.total_seconds()
COMPLEX_MEAL_DURATION = timedelta(minutes=60)


START_OF_TRIAL = [datetime.strptime("11 06 2021-04:00:00", "%m %d %Y-%H:%M:%S"), datetime.strptime("02 03 2022-00:00:00", "%m %d %Y-%H:%M:%S")]
END_OF_TRIAL = [datetime.strptime("11 15 2021-00:00:00", "%m %d %Y-%H:%M:%S"), datetime.strptime("02 13 2022-00:00:00", "%m %d %Y-%H:%M:%S")]
DAY_LIGHT_SAVING = datetime.strptime("11 06 2021-02:00:00", "%m %d %Y-%H:%M:%S")
coreNumber = 24

addDataPrefix = "/Users/sorush/My Drive/Documents/Educational/TAMU/Research/TAMU/"
if not os.path.exists(addDataPrefix):
    addDataPrefix = "/home/grads/s/sorush.omidvar/CGMDataset/TAMU/"
if not os.path.exists(addDataPrefix):
    addDataPrefix = "C:\\GDrive\\Documents\\Educational\\TAMU\\Research\\Trial\\Data\\11-5-21-11-15-21"

addUserInput = os.path.join(addDataPrefix, "User inputted")
addHKCM = os.path.join(addDataPrefix, "hk+cm")
addCGM = os.path.join(addDataPrefix, "CGM")
addE4 = os.path.join(addDataPrefix, "E4")
addResults = os.path.join(addDataPrefix, "Results"+sys.argv[1])
if not os.path.exists(addResults):
    os.mkdir(addResults)

exempts = ["p2", "p4"]

pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use({"figure.facecolor": "white"})
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU


warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 500)

# %%
# participants=list(set(dfMeal['Participant'].to_list()))
# participants.sort()
# # dfMeal.insert(0,'Duration',0)
# # dfMeal['Duration']=dfMeal['FinishTime']-dfMeal['StartTime']
# for participant in participants:
#     dfTemp=dfMeal[dfMeal['Participant']==participant]
#     print(participant,'&',len(dfTemp),'&',np.round(dfTemp['Duration'].dt.total_seconds().mean()/60,1),'(',np.round(dfTemp['Duration'].dt.total_seconds().std()/60,1),')','&',np.round(dfTemp['Calories'].mean(),1),"(",np.round(dfTemp['Calories'].std(),1),")",'&',np.round(dfTemp['Carbs'].mean(),1),"(",np.round(dfTemp['Carbs'].std(),1),")",'&',np.round(dfTemp['Fat'].mean(),1),"(",np.round(dfTemp['Fat'].std(),1),")",'&',np.round(dfTemp['Protein'].mean(),1),"(",np.round(dfTemp['Protein'].std(),1),")")


# %%
# files = os.listdir(addResults)
# for item in files:
#     if ".xlsx" in item or '.jpg' in item or 'All-Features' in item:
#         os.remove(os.path.join(addResults, item))


# %%
# T1 = datetime.strptime("02 03 2022-00:00:00", "%m %d %Y-%H:%M:%S")
# T2=T1+timedelta(hours=120)
# a=dfE4[(dfE4['Time']>=T1) &(dfE4['Time']<=T2)]

# plt.figure(figsize=(40,10))
# x=a['Time'].to_list()
# x=np.asarray(x)
# y=a['Data1'].to_list()
# plt.scatter(x,y)
# plt.grid(which='both',color='r', linestyle='-', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([T1,T2])


# plt.figure(figsize=(40,10))
# b=dfCM[(dfCM['Time']>=T1) &(dfCM['Time']<=T2)]
# x=b['Time'].to_list()
# x=np.asarray(x)
# y=b['Yaw'].to_list()
# y=np.asarray(y)/10+70
# plt.scatter(x,y)
# plt.grid(which='both',color='r', linestyle='-', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([T1,T2])


# %%
# def unzipperE4(participantFolder):
#     for root, dirs, files in os.walk(participantFolder):
#         for file in files:
#             if not '.zip' in file:
#                 continue
#             with zipfile.ZipFile(os.path.join(root,file), 'r') as zip_ref:
#                 destFile=file[:file.find('.zip')]
#                 destFile=os.path.join(root,destFile)
#                 if not os.path.exists(destFile):
#                     os.mkdir(destFile)
#                 zip_ref.extractall(destFile)
# def zipCleanerE4(E4Folder):
#     for root, dirs, files in os.walk(E4Folder):
#         for file in files:
#             if '.zip' in file:
#                 os.remove(os.path.join(root,file))

# unzipperE4('/Users/sorush/Desktop/Round2E4/p5')
# unzipperE4('/Users/sorush/Desktop/Round2E4/p6')
# unzipperE4('/Users/sorush/Desktop/Round2E4/p7')
# unzipperE4('/Users/sorush/Desktop/Round2E4/p8')
# zipCleanerE4('/Users/sorush/Desktop/Round2E4')


# %%
def timeZoneFixer(df, LocalizeFlag, columnName):
    if LocalizeFlag:
        df[columnName] -= timedelta(hours=5)
    tempColumn = df[columnName]
    tempColumn[tempColumn >= DAY_LIGHT_SAVING] -= timedelta(hours=1)
    df[columnName] = tempColumn
    return df


def trialTimeLimitter(df, columnName):
    participants = list(set(df["Participant"].to_list()))
    dfTotal = []
    for participant in participants:
        if participant == "p1" or participant == "p2" or participant == "p3" or participant == "p4":
            startOfTrial = START_OF_TRIAL[0]
            endOfTrial = END_OF_TRIAL[0]
        elif participant == "p5" or participant == "p6" or participant == "p7" or participant == "p8":
            startOfTrial = START_OF_TRIAL[1]
            endOfTrial = END_OF_TRIAL[1]
        else:
            print("Mayday in trialTimeLimitter")
            print(participant)
            raise
        dfTemp = df[df["Participant"] == participant]
        dfTemp = dfTemp[(dfTemp[columnName] >= startOfTrial) & (dfTemp[columnName] <= endOfTrial)]
        if len(dfTotal) == 0:
            dfTotal = dfTemp
        else:
            frames = [dfTotal, dfTemp]
            dfTotal = pd.concat(frames)
    return dfTotal


def mealMarker(df):
    df.insert(len(df.columns), "BigMeal", False)
    for counter in range(0, len(df)):
        if df["Calories"].iloc[counter] >= BIG_MEAL_CALORIE:
            df["BigMeal"].iloc[counter] = True

    df.insert(len(df.columns), "ComplexMeal", False)
    participants = df["Participant"].to_list()
    participants = list(set(participants))
    for participant in participants:
        dfTemp = df[df["Participant"] == participant]
        for counter in range(1, len(dfTemp)):
            # bothComplexFlag = dfTemp["BigMeal"].iloc[counter - 1] and dfTemp["BigMeal"].iloc[counter]
            # if dfTemp["StartTime"].iloc[counter - 1] + OUTTER_WINDOW_LENGTH >= dfTemp["StartTime"].iloc[counter] and bothComplexFlag:
            if dfTemp["StartTime"].iloc[counter - 1] + COMPLEX_MEAL_DURATION >= dfTemp["StartTime"].iloc[counter]:
                dfTemp["ComplexMeal"].iloc[counter] = True
                dfTemp["ComplexMeal"].iloc[counter - 1] = True
        indexs = dfTemp.index[dfTemp["ComplexMeal"] == True]
        df["ComplexMeal"][indexs] = True
    return df


if os.path.exists(os.path.join(addResults, "All_meals.pkl")):
    os.remove(os.path.join(addResults, "All_meals.pkl"))
os.chdir(addUserInput)
if not os.path.exists(os.path.join(addResults, "All_meals.pkl")):
    dfMeal = []
    for root, dirs, files in os.walk(addUserInput):
        print(root)
        for file in files:
            if ".csv" in file.lower():
                if "meals" in file.lower() and "modified" not in file.lower():
                    participantName = file[: file.find("Meals")]
                    if participantName in exempts:
                        print("Exemption...", file)
                        continue
                    print("Reading ...", file)
                    dfTemp = pd.read_csv(file)
                    dfTemp.insert(0, "Participant", participantName)
                    dfTemp.rename(columns={"startTime": "StartTime"}, inplace=True)
                    dfTemp["StartTime"] = pd.to_datetime(dfTemp["StartTime"])
                    dfTemp["FinishTime"] = pd.to_datetime(dfTemp["FinishTime"])
                    dfTemp.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
                    dfTemp.reset_index(drop=True, inplace=True)

                    if len(dfMeal) != 0:
                        frames = [dfTemp, dfMeal]
                        dfMeal = pd.concat(frames)
                    else:
                        dfMeal = dfTemp
    print("reading is done")
    dfMeal = trialTimeLimitter(dfMeal, "StartTime")
    dfMeal = trialTimeLimitter(dfMeal, "FinishTime")
    dfMeal.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
    dfMeal.reset_index(drop=True, inplace=True)
    # dfMeal.insert(4, "MealDuration", -1)
    # dfMeal["MealDuration"] = dfMeal["FinishTime"] - dfMeal["StartTime"]
    # dfMeal["MealDuration"] = dfMeal["MealDuration"].dt.total_seconds()
    print("Meal database is limited to the trial period")
    dfMeal.to_pickle(os.path.join(addResults, "All_meals.pkl"))
else:
    dfMeal = pd.read_pickle(os.path.join(addResults, "All_meals.pkl"))
dfMeal = mealMarker(dfMeal)


# %%
def pdInterpolation(dfTemp):
    index = dfTemp["Time"]
    seriesParticipant = pd.Series(dfTemp["Abbot"].to_list(), index=index)
    seriesParticipant = seriesParticipant.resample("1T").asfreq()
    seriesParticipant.interpolate(method="polynomial", order=3, inplace=True)
    tempTime = seriesParticipant.index
    tempVal = seriesParticipant.values
    dfTemp = pd.DataFrame(zip(tempTime, tempVal), columns=["Time", "Abbot"])
    return dfTemp


def cmLagCorrector(df):
    participants = df["Participant"].to_list()
    participants = list(set(participants))
    dfTotal = []

    for element in CM_LAG_CORRECTION:
        participant = element[0]
        timeLag = element[1]
        dfParticipant = df[df["Participant"] == participant]
        if len(dfParticipant) == 0:
            continue
        dfParticipant["Time"] += timeLag
        if len(dfTotal) == 0:
            dfTotal = dfParticipant
        else:
            frames = [dfTotal, dfParticipant]
            dfTotal = pd.concat(frames)

    return dfTotal


def cmSmoother(df):
    columnLabels = df.columns
    for columnLabel in columnLabels:
        if columnLabel == "Time":
            continue
        tempSerie = df[columnLabel]
        tempSerie = tempSerie.ewm(span=10).mean()  # Considering the frequency of 10 Hz
        df[columnLabel] = tempSerie
    return df


def CGMLagImposer(df):
    df["Time"] += CGM_LAG_IMPOSING
    return df


# if os.path.exists(os.path.join(addResults, "All_cgm.pkl")):
#     os.remove(os.path.join(addResults, "All_cgm.pkl"))
if not os.path.exists(os.path.join(addResults, "All_cgm.pkl")):
    os.chdir(addCGM)
    dfCGM = []
    for root, dirs, files in os.walk(addCGM):
        for file in files:
            if ".txt" in file.lower():
                if "_libre" in file.lower():
                    participantName = file[: file.find("_libre")]
                    if participantName in exempts:
                        print("Exemption...", file)
                        continue
                    print("Reading ...", file)
                    dfTemp = pd.read_csv(file, sep="\t", skiprows=1)
                    if len(dfTemp.columns) != 4:
                        print("MAYDAY. Error in reading csv")
                        break
                    dfTemp.columns.values[0] = "ID"
                    dfTemp.columns.values[1] = "Time"
                    dfTemp.columns.values[2] = "Record"
                    dfTemp.columns.values[3] = "Abbot"
                    dfTemp.drop(columns=["ID", "Record"], inplace=True)
                    dfTemp["Time"] = pd.to_datetime(dfTemp["Time"])
                    if participantName == "p1" or participantName == "p2" or participantName == "p3" or participantName == "p4":
                        dfTemp["Time"] += timedelta(hours=-1)  # This fixes the daylight saving for the first round
                    dfTemp["Abbot"] = pd.to_numeric(dfTemp["Abbot"])
                    dfTemp.sort_values(["Time"], ascending=(True), inplace=True)
                    dfTemp = pdInterpolation(dfTemp)
                    dfTemp.insert(0, "Participant", participantName)
                    if len(dfTemp.columns) != 3:
                        print("MAYDAY. Error in processing csv")
                        break
                    if len(dfCGM) != 0:
                        frames = [dfTemp, dfCGM]
                        dfCGM = pd.concat(frames)
                    else:
                        dfCGM = dfTemp
    print("reading is done")
    dfCGM = CGMLagImposer(dfCGM)
    dfCGM = trialTimeLimitter(dfCGM, "Time")
    dfCGM.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfCGM.reset_index(drop=True, inplace=True)
    print("CGM database is limited to the trial period")
    dfCGM.to_pickle(os.path.join(addResults, "All_cgm.pkl"))
else:
    dfCGM = pd.read_pickle(os.path.join(addResults, "All_cgm.pkl"))


# if os.path.exists(os.path.join(addResults, "All_cm.pkl")):
#     os.remove(os.path.join(addResults, "All_cm.pkl"))
os.chdir(addHKCM)
if not os.path.exists(os.path.join(addResults, "All_cm.pkl")):
    dfCM = []
    for root, dirs, files in os.walk(addHKCM):
        for file in files:
            if ".csv" in file.lower():
                if "corrected_cm_all" in file.lower():
                    participantName = file[: file.find("_corrected")]
                    if participantName in exempts:
                        print("Exemption...", file)
                        continue
                    print("Reading ...", file)
                    dfTemp = pd.read_csv(file)
                    print("File is read")
                    dfTemp["UnixTime"] = pd.to_datetime(dfTemp["UnixTime"], unit="s")

                    dfTemp.rename(columns={"UnixTime": "Time"}, inplace=True)
                    dfTemp.drop(columns=["UID", "Date"], inplace=True)
                    dfTemp.sort_values(["Time"], ascending=(True), inplace=True)

                    dfTemp = cmSmoother(dfTemp)
                    dfTemp["Yaw"] *= 180 / 3.1415
                    dfTemp["Pitch"] *= 180 / 3.1415
                    dfTemp["Roll"] *= 180 / 3.1415

                    dfTemp.insert(0, "Participant", participantName)
                    # this is to avoid 0 later on for feature calculation
                    dfTemp.insert(len(dfTemp.columns), "|Ax|+|Ay|+|Az|", dfTemp["Ax"].abs() + dfTemp["Ay"].abs() + dfTemp["Az"].abs() + 0.001)
                    dfTemp.insert(len(dfTemp.columns), "|Yaw|+|Roll|+|Pitch|", dfTemp["Yaw"].abs() + dfTemp["Roll"].abs() + dfTemp["Pitch"].abs())
                    dfTemp.insert(len(dfTemp.columns), "|Rx|+|Ry|+|Rz|To|Ax|+|Ay|+|Az|", dfTemp["Rx"].abs() + dfTemp["Ry"].abs() + dfTemp["Rz"].abs())
                    dfTemp["|Rx|+|Ry|+|Rz|To|Ax|+|Ay|+|Az|"] = dfTemp["|Rx|+|Ry|+|Rz|To|Ax|+|Ay|+|Az|"] / dfTemp["|Ax|+|Ay|+|Az|"]
                    dfTemp.insert(len(dfTemp.columns), "RotationalToLinear", dfTemp["|Yaw|+|Roll|+|Pitch|"] / dfTemp["|Ax|+|Ay|+|Az|"])
                    print("modified")

                    if len(dfTemp.columns) != 15:
                        print("MAYDAY. Error in reading csv")
                        print(dfTemp.columns)
                        break
                    if len(dfCM) != 0:
                        frames = [dfTemp, dfCM]
                        dfCM = pd.concat(frames)
                    else:
                        dfCM = dfTemp
    print("Processing is done")
    dfCM = cmLagCorrector(dfCM)
    dfCM.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfCM.reset_index(drop=True, inplace=True)
    print("CM database is limited to the trial period")
    dfCM.to_pickle(os.path.join(addResults, "All_cm.pkl"))
else:
    dfCM = pd.read_pickle(os.path.join(addResults, "All_cm.pkl"))


# if os.path.exists(os.path.join(addResults, "All_E4.pkl")):
#     os.remove(os.path.join(addResults, "All_E4.pkl"))
os.chdir(addE4)
# fields=['ACC','BVP','EDA','HR','IBI','TEMP']
fields = ["HR", "TEMP", "EDA"]
if not os.path.exists(os.path.join(addResults, "All_E4.pkl")):
    dfE4 = []
    for root, dirs, files in os.walk(addE4):
        for file in files:
            if ".csv" in file.lower():
                participantName = root[root.find("E4") + 3 :]
                participantName = participantName[:2]
                field = file[: file.find(".csv")]
                if field not in fields:
                    print("File name does not comply with analyzed fields", file)
                    continue
                print(participantName, field)
                if participantName in exempts:
                    print("Exemption...", file)
                    continue
                print("Reading ...", file)
                os.chdir(root)
                dfTemp = pd.read_csv(file, header=None)
                # if field=='ACC':
                #     assert len(dfTemp.columns)==3
                #     timeBase=dfTemp.iloc[0,0]
                #     timeStep=1/dfTemp.iloc[1,0]
                #     dfTemp.drop([0,1],inplace=True)
                #     dfTemp.rename(columns={0:'Data1',1:'Data2',2:'Data3'}, inplace=True)#x,y,z for data1,data2,data3
                #     timeTemp=[]
                #     for counter in range(len(dfTemp)):
                #         timeTemp.append(timeBase+counter*timeStep)
                #     dfTemp.insert(0,'Time',timeTemp)
                #     dfTemp.insert(0,'Field',"Acceleration")
                #     dfTemp['Time'] = pd.to_datetime(dfTemp['Time'],unit='s')
                # if field == "BVP":
                #     assert len(dfTemp.columns) == 1
                #     timeBase = dfTemp.iloc[0, 0]
                #     timeStep = 1 / dfTemp.iloc[1, 0]
                #     dfTemp.drop([0, 1], inplace=True)
                #     dfTemp.rename(columns={0: "Data1"}, inplace=True)
                #     dfTemp["Data2"] = ""
                #     dfTemp["Data3"] = ""
                #     timeTemp = []
                #     for counter in range(len(dfTemp)):
                #         timeTemp.append(timeBase + counter * timeStep)
                #     dfTemp.insert(0, "Time", timeTemp)
                #     dfTemp.insert(0, "Field", "BVP")
                #     dfTemp["Time"] = pd.to_datetime(dfTemp["Time"], unit="s")
                if field == "HR":
                    assert len(dfTemp.columns) == 1
                    timeBase = dfTemp.iloc[0, 0]
                    timeStep = 1 / dfTemp.iloc[1, 0]
                    dfTemp.drop([0, 1], inplace=True)
                    dfTemp.rename(columns={0: "Data1"}, inplace=True)
                    dfTemp["Data2"] = ""
                    dfTemp["Data3"] = ""
                    timeTemp = []
                    for counter in range(len(dfTemp)):
                        timeTemp.append(timeBase + counter * timeStep)
                    dfTemp.insert(0, "Time", timeTemp)
                    dfTemp.insert(0, "Field", "HR")
                    dfTemp["Time"] = pd.to_datetime(dfTemp["Time"], unit="s")
                elif field == "EDA":
                    assert len(dfTemp.columns) == 1
                    timeBase = dfTemp.iloc[0, 0]
                    timeStep = 1 / dfTemp.iloc[1, 0]
                    dfTemp.drop([0, 1], inplace=True)
                    dfTemp.rename(columns={0: "Data1"}, inplace=True)
                    dfTemp["Data2"] = ""
                    dfTemp["Data3"] = ""
                    timeTemp = []
                    for counter in range(len(dfTemp)):
                        timeTemp.append(timeBase + counter * timeStep)
                    dfTemp.insert(0, "Time", timeTemp)
                    dfTemp.insert(0, "Field", "EDA")
                    dfTemp["Time"] = pd.to_datetime(dfTemp["Time"], unit="s")
                # elif field=='IBI':
                #     assert len(dfTemp.columns)==2
                #     timeBase=dfTemp.iloc[0,0]
                #     dfTemp.drop([0],inplace=True)
                #     dfTemp.rename(columns={0:'Time',1:'Data1'}, inplace=True)
                #     dfTemp["Data2"]=""
                #     dfTemp["Data3"]=""
                #     timeTemp=[]
                #     dfTemp['Time']+=timeBase
                #     dfTemp.insert(0,'Field',"IBI")
                #     dfTemp['Time'] = pd.to_datetime(dfTemp['Time'],unit='s')
                elif field == "TEMP":
                    assert len(dfTemp.columns) == 1
                    timeBase = dfTemp.iloc[0, 0]
                    timeStep = 1 / dfTemp.iloc[1, 0]
                    dfTemp.drop([0, 1], inplace=True)
                    dfTemp.rename(columns={0: "Data1"}, inplace=True)
                    dfTemp["Data2"] = ""
                    dfTemp["Data3"] = ""
                    timeTemp = []
                    for counter in range(len(dfTemp)):
                        timeTemp.append(timeBase + counter * timeStep)
                    dfTemp.insert(0, "Time", timeTemp)
                    dfTemp.insert(0, "Field", "Temperature")
                    dfTemp["Time"] = pd.to_datetime(dfTemp["Time"], unit="s")
                dfTemp.insert(0, "Participant", participantName)
                dfTemp.sort_values(["Participant", "Field", "Time"], ascending=(True, True, True), inplace=True)
                if len(dfTemp.columns) != 6:
                    print("MAYDAY. Error in reading csv")
                    break
                if len(dfE4) != 0:
                    frames = [dfTemp, dfE4]
                    dfE4 = pd.concat(frames)
                else:
                    dfE4 = dfTemp
    print("reading is done")
    dfE4 = timeZoneFixer(dfE4, True, "Time")
    dfE4.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfE4 = trialTimeLimitter(dfE4, "Time")
    dfE4.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfE4.reset_index(drop=True, inplace=True)
    print("E4 database is limited to the trial period")
    dfE4.to_pickle(os.path.join(addResults, "All_E4.pkl"))
else:
    dfE4 = pd.read_pickle(os.path.join(addResults, "All_E4.pkl"))


# %%
# T1 = datetime.strptime("02 03 2022-00:00:00", "%m %d %Y-%H:%M:%S")
# T2 = T1 + timedelta(hours=240)
# myParticipant = "p7"
# a = dfE4[dfE4["Participant"] == myParticipant]
# a = a[(a["Time"] >= T1) & (a["Time"] <= T2)]

# plt.figure(figsize=(40, 10))
# x = a["Time"].to_list()
# x = np.asarray(x)
# y = a["Data1"].to_list()
# plt.scatter(x, y)
# plt.grid(which="both", color="r", linestyle="-", linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([T1, T2])


# plt.figure(figsize=(40, 10))
# b = dfCM[dfCM["Participant"] == myParticipant]
# b = b[(b["Time"] >= T1) & (b["Time"] <= T2)]
# x = b["Time"].to_list()
# x = np.asarray(x)
# y = b["Yaw"].to_list()
# y = np.asarray(y) / 10 + 70
# plt.scatter(x, y)
# plt.grid(which="both", color="r", linestyle="-", linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([T1, T2])



# %%
# fig=plt.figure(figsize=(18,24))
# plt.subplot(6,1,1)
# participant='p8'
# colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# T1 = datetime.strptime("02 03 2022-12:00:00", "%m %d %Y-%H:%M:%S")  # to handle the daylight saving issue in apple watches
# T2 = T1 + timedelta(hours=5)
# dfTempCGM=dfCGM[dfCGM['Participant']==participant]
# dfTempCGM=dfTempCGM[(dfTempCGM['Time']>=T1) & (dfTempCGM['Time']<T2)]

# x=dfTempCGM['Time'].to_list()
# y=dfTempCGM['Abbot'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# x=x[1:len(x):15]
# y=y[1:len(y):15]
# myAx=plt.plot(x,y, '-*',c=colors[0])
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.axes.set_xticklabels([])
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([12,17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)
# plt.ylabel('BG [mg/dL]')
# plt.scatter(x=12+43/60,y=120,color='red')


# plt.subplot(6,1,2)
# dfTempE4=dfE4[dfE4['Participant']==participant]
# dfTempE4=dfTempE4[(dfTempE4['Time']>=T1) & (dfTempE4['Time']<T2) &(dfTempE4['Field']=='HR')]

# x=dfTempE4['Time'].to_list()
# y=dfTempE4['Data1'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# myAx=plt.plot(x,y ,'-',c=colors[1])
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.axes.set_xticklabels([])
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([11,17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)
# plt.ylabel('Heart Rate [BPM]')

# plt.subplot(6,1,3)
# dfTempE4=dfE4[dfE4['Participant']==participant]
# dfTempE4=dfTempE4[(dfTempE4['Time']>=T1) & (dfTempE4['Time']<T2) &(dfTempE4['Field']=='Temperature')]

# x=dfTempE4['Time'].to_list()
# y=dfTempE4['Data1'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# myAx=plt.plot(x,y ,'-',c=colors[3])
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.axes.set_xticklabels([])
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([12,17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)
# plt.ylabel('Temperature [$^\circ$C]')


# plt.subplot(6,1,4)
# dfTempE4=dfE4[dfE4['Participant']==participant]
# dfTempE4=dfTempE4[(dfTempE4['Time']>=T1) & (dfTempE4['Time']<T2) &(dfTempE4['Field']=='EDA')]

# x=dfTempE4['Time'].to_list()
# y=dfTempE4['Data1'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# myAx=plt.plot(x,y ,'-',c=colors[4])
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.axes.set_xticklabels([])
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([12,17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)
# plt.ylabel('EDA [$^\mu$S]')


# plt.subplot(6,1,5)
# dfCMTemp=dfCM[dfCM['Participant']==participant]
# dfCMTemp=dfCMTemp[(dfCMTemp['Time']>=T1) & (dfCMTemp['Time']<T2)]

# x=dfCMTemp['Time'].to_list()
# y=dfCMTemp['Ax'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# myAx=plt.plot(x,y ,'-',c=colors[5])
# frame1 = plt.gca()
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.axes.set_xticklabels([])
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([12,17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)
# plt.ylabel('Acceleration [$m^2$/s]')


# plt.subplot(6,1,6)
# dfCMTemp=dfCM[dfCM['Participant']==participant]
# dfCMTemp=dfCMTemp[(dfCMTemp['Time']>=T1) & (dfCMTemp['Time']<T2)]

# x=dfCMTemp['Time'].to_list()
# y=dfCMTemp['Yaw'].to_list()
# y=np.asarray(y).astype(float)
# for counter in range(len(x)):
#     x[counter]=x[counter].to_pydatetime()
#     x[counter]=x[counter].time().hour+x[counter].time().minute/60+x[counter].time().second/3600
# myAx=plt.plot(x,y ,'-',c=colors[6])
# frame1 = plt.gca()
# frame1.axes.set_xticks([12,14,16,18])
# frame1.axes.minorticks_on()
# frame1.tick_params('both', length=6, width=1, which='major')
# frame1.tick_params('both', length=2, width=1, which='minor')
# frame1.axes.set_xlim([12, 17])
# frame1.axes.yaxis.set_label_coords(-0.1,0.5)

# plt.ylabel('Yaw [$^\circ$]')
# plt.xlabel('Time [hr]')
# plt.show()


# %%
def e4Reporter(df):
    # topics = ["BVP", "EDA", "HR", "Temperature"]
    topics = ["EDA", "HR", "Temperature"]
    report = []
    for topic in topics:
        dfTemp = df[df["Field"] == topic]
        # if topic == "BVP":
        #     MIN_POINT = MINIMUM_POINT * 64 * 0.3
        if topic == "EDA":
            MIN_POINT = MINIMUM_POINT * 4 * 0.3
        elif topic == "HR":
            MIN_POINT = MINIMUM_POINT / 10 * 0.3
        elif topic == "Temperature":
            MIN_POINT = MINIMUM_POINT * 4 * 0.3
        else:
            print(topic)
            print("MAYDAY at sensor reader")
            os._exit()
        if len(dfTemp) < MIN_POINT:
            report.append("Nan")
        else:
            val = dfTemp["Data1"].mean()
            report.append(val)
    return report


def motionCalculator(df):
    f1 = df["RotationalToLinear"]
    f2 = df["|Ax|+|Ay|+|Az|"]
    return [f1.mean(), f1.std(), f1.max() - f1.min(), f2.mean(), f2.std(), f2.max() - f2.min()]


def statFeatures(dataList):
    dataList = np.asarray(dataList).astype(float)
    result = []
    dataDim = dataList.ndim
    if dataDim > 1:
        for counter in range(dataList.shape[1]):
            if not np.isnan(dataList[:, counter]).all():
                meanVal = np.nanmean(dataList[:, counter], axis=0)
                stdVal = np.nanstd(dataList[:, counter], axis=0)
                minVal = np.nanmin(dataList[:, counter], axis=0)
                maxVal = np.nanmax(dataList[:, counter], axis=0)
                rangeVal = maxVal - minVal
                skewnessVal = skew(dataList[:, counter], nan_policy="omit", axis=0)
                kurtosisVal = kurtosis(dataList[:, counter], nan_policy="omit", axis=0)
                result.extend([rangeVal, meanVal, stdVal, minVal, maxVal, skewnessVal, kurtosisVal])
            else:
                result.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    else:
        if not np.isnan(dataList).all():
            meanVal = np.nanmean(dataList)
            stdVal = np.nanstd(dataList)
            minVal = np.nanmin(dataList)
            maxVal = np.nanmax(dataList)
            rangeVal = maxVal - minVal
            skewnessVal = skew(dataList, nan_policy="omit")
            kurtosisVal = kurtosis(dataList, nan_policy="omit")
            result.extend([rangeVal, meanVal, stdVal, minVal, maxVal, skewnessVal, kurtosisVal])
        else:
            result.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    return result


def innerWindowExtractor(outterWindowStart, innerWindowNumber, dfParticipantCM, dfParticipantE4):
    tempListCM = []
    tempListE4 = []
    for counterInner in range(0, innerWindowNumber, 1):
        innerWindowStart = outterWindowStart + counterInner * INNER_WINDOW_LENGTH
        innerWindowEnd = innerWindowStart + INNER_WINDOW_LENGTH
        dfTempCM = dfParticipantCM[(dfParticipantCM["Time"] >= innerWindowStart) & (dfParticipantCM["Time"] < innerWindowEnd)]

        if len(dfTempCM) < MINIMUM_POINT * 10 * 0.3:
            tempListCM.append(["Nan", "Nan", "Nan", "Nan", "Nan", "Nan"])
        else:
            tempListCM.append(motionCalculator(dfTempCM))

        dfTempE4 = dfParticipantE4[(dfParticipantE4["Time"] >= innerWindowStart) & (dfParticipantE4["Time"] < innerWindowEnd)]
        tempListE4.append(e4Reporter(dfTempE4))

    return tempListCM, tempListE4


def parallelCall(windowData, dfParticipantCM, dfParticipantE4, dfParticipantCGM):
    tempList = []
    outterWindowStart = windowData[0]
    outterWindowEnd = windowData[1]
    innerWindowNumber = windowData[2]
    carbs = windowData[3]
    fat = windowData[4]
    protein = windowData[5]
    mealFlag = windowData[6]
    participant = windowData[7]

    dfTempCM = dfParticipantCM[(dfParticipantCM["Time"] >= outterWindowStart) & (dfParticipantCM["Time"] < outterWindowEnd)]
    dfTempE4 = dfParticipantE4[(dfParticipantE4["Time"] >= outterWindowStart) & (dfParticipantE4["Time"] < outterWindowEnd)]
    tempListCM, tempListE4 = innerWindowExtractor(outterWindowStart, innerWindowNumber, dfTempCM, dfTempE4)

    # tempListCM = statFeatures(tempListCM)
    tempList.append(tempListCM)  # 1

    tempListE4 = statFeatures(tempListE4)
    tempList.extend(tempListE4)  # 21

    dfTempCGM = dfParticipantCGM[(dfParticipantCGM["Time"] >= outterWindowStart) & (dfParticipantCGM["Time"] < outterWindowEnd)]
    tempListCGM = dfTempCGM["Abbot"].to_list()
    tempListCGM = statFeatures(tempListCGM)
    tempList.extend(tempListCGM)  # 7

    tempList.append(outterWindowStart)  # 1
    tempList.append(outterWindowEnd)  # 1
    tempList.append(participant)  # 1

    tempList.append(carbs)  # 1
    tempList.append(fat)  # 1
    tempList.append(protein)  # 1

    tempList.append(mealFlag)  # mealFlag

    assert len(tempList) == 1 + 21 + 7 + 3 + 3 + 1

    return tempList


def outterNegWindowExtractor(dfParticipantMeal, dfParticipantCM, dfParticipantE4, dfParticipantCGM, participant):
    print("Negative windows:")
    participantDataList = []
    gaps = []
    for counterOuter in range(1, len(dfParticipantMeal)):
        if dfParticipantMeal["StartTime"].iloc[counterOuter] - dfParticipantMeal["StartTime"].iloc[counterOuter - 1] >= FASTING_LENGTH:
            # if not dfParticipantMeal["ComplexMeal"].iloc[counterOuter]:
            counter = 0
            while True:
                endQuerry = dfParticipantMeal["StartTime"].iloc[counterOuter] - counter * OUTTER_WINDOW_LENGTH
                startQuerry = endQuerry - OUTTER_WINDOW_LENGTH
                if startQuerry > dfParticipantMeal["StartTime"].iloc[counterOuter - 1] + FASTING_LENGTH:
                    gaps.append([startQuerry, endQuerry])
                else:
                    break
                if counter == 50:  # Each positive window can have 10 negative winodws at most
                    break
                counter += 1
    windowDatas = []
    for counterOuter in range(len(gaps)):
        element = gaps[counterOuter]
        outterWindowStart = element[0]
        outterWindowEnd = element[1]
        innerWindowNumber = int(OUTTER_WINDOW_LENGTH.total_seconds() / INNER_WINDOW_LENGTH.total_seconds())

        carbs = 0
        fat = 0
        protein = 0

        windowDatas.append([outterWindowStart, outterWindowEnd, innerWindowNumber, carbs, fat, protein, 0, participant])
    for counterOuter in tqdm(range(len(windowDatas))):
        windowData = windowDatas[counterOuter]
        participantDataList.append(parallelCall(windowData, dfParticipantCM, dfParticipantE4, dfParticipantCGM))
    # participantDataList = Parallel(n_jobs=coreNumber)(delayed(parallelCall)(windowData, dfParticipantCM, dfParticipantE4, dfParticipantCGM) for windowData in tqdm(windowDatas))
    return participantDataList


def outterPosWindowExtractor(dfParticipantMeal, dfParticipantCM, dfParticipantE4, dfParticipantCGM, participant):
    print("Positive windows:")
    windowDatas = []
    participantDataList = []
    for counterOuter in range(len(dfParticipantMeal)):
        # if not dfParticipantMeal["BigMeal"].iloc[counterOuter]:
        #     continue
        # if dfParticipantMeal["ComplexMeal"].iloc[counterOuter]:
        #     continue
        outterWindowStart = dfParticipantMeal["StartTime"].iloc[counterOuter]
        outterWindowEnd = outterWindowStart + OUTTER_WINDOW_LENGTH
        innerWindowNumber = int(OUTTER_WINDOW_LENGTH.total_seconds() / INNER_WINDOW_LENGTH.total_seconds())

        carbs = dfParticipantMeal["Carbs"].iloc[counterOuter]
        fat = dfParticipantMeal["Fat"].iloc[counterOuter]
        protein = dfParticipantMeal["Protein"].iloc[counterOuter]

        windowDatas.append([outterWindowStart, outterWindowEnd, innerWindowNumber, carbs, fat, protein, 1, participant])
    for counterOuter in tqdm(range(len(windowDatas))):
        windowData = windowDatas[counterOuter]
        participantDataList.append(parallelCall(windowData, dfParticipantCM, dfParticipantE4, dfParticipantCGM))
    # participantDataList = Parallel(n_jobs=coreNumber)(delayed(parallelCall)(windowData, dfParticipantCM, dfParticipantE4, dfParticipantCGM) for windowData in tqdm(windowDatas))
    return participantDataList


def main():
    allDataList = []
    participants = dfMeal["Participant"].to_list()
    participants = list(set(participants))
    participants.sort()
    columnHeaderList = ["CM"]
    sensors = ["EDA", "HR", "Temperature", "CGM"]
    statFeatureNames = ["-Mean", "-Std", "-Min", "-Max", "-Range", "-Skewness", "-Kurtosis"]
    for sensor in sensors:
        for statFeatureName in statFeatureNames:
            columnHeaderList.append(sensor + statFeatureName)
    columnHeaderList.extend(["StartTime", "FinishTime", "Participant", "Carb", "Fat", "Protein", "MealLabel"])
    for participant in participants:
        print("Participant:", participant)
        if participant in exempts:
            continue
        dfParticipantMeal = dfMeal[dfMeal["Participant"] == participant]
        dfParticipantCM = dfCM[dfCM["Participant"] == participant]
        dfParticipantE4 = dfE4[dfE4["Participant"] == participant]
        dfParticipantCGM = dfCGM[dfCGM["Participant"] == participant]
        participantDataList = outterPosWindowExtractor(dfParticipantMeal, dfParticipantCM, dfParticipantE4, dfParticipantCGM, participant)
        participantDataList.extend(outterNegWindowExtractor(dfParticipantMeal, dfParticipantCM, dfParticipantE4, dfParticipantCGM, participant))

        participantDataList = pd.DataFrame(participantDataList, columns=columnHeaderList)
        if len(allDataList) == 0:
            allDataList = participantDataList
        else:
            frames = [allDataList, participantDataList]
            allDataList = pd.concat(frames)

    allDataList.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
    allDataList.reset_index(drop=True, inplace=True)
    # Dropping rows without CoreMotion
    CMData = allDataList["CM"].to_list()
    CMData = np.asarray(CMData).astype(float)
    nanList = []
    for counter in range(CMData.shape[0]):
        if np.isnan(CMData[counter, :, 0]).all():
            nanList.append(counter)
    allDataList.drop(allDataList.index[nanList], inplace=True)

    allDataList.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
    allDataList.reset_index(drop=True, inplace=True)
    allDataList.to_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl"),))
    return allDataList


# if os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl"))):
#     os.remove(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl")))
if not os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl"),)):
    dfAllFeatures = main()

else:
    dfAllFeatures = pd.read_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl"),))


# %%
def meanSTDFinder(df):
    cmData = []
    for counter in range(len(df)):
        elements = df["CM"].iloc[counter]
        for element in elements:
            cmData.append(element)
    cmData = np.asarray(cmData).astype(float)
    cmDataMean = np.nanmean(cmData, axis=0)
    cmDataStd = np.nanstd(cmData, axis=0)
    return cmDataMean, cmDataStd


def cmNormalizerPredictor(element, cmDataMean, cmDataStd, hooverModel):
    element = np.asarray(element).astype(float)
    element -= cmDataMean
    element /= cmDataStd
    element = np.expand_dims(element, axis=0)
    hooverPrediction = hooverModel.predict_proba(element)
    hooverPrediction = hooverPrediction[0, 1]

    return hooverPrediction


def maxProbWinodFinder(tempPredictions):
    tempMax = -1
    for innerCounter in range(len(tempPredictions) - 5):
        tempArray = np.asarray(tempPredictions[innerCounter : innerCounter + 5])
        tempArray = np.mean(tempArray)
        if tempArray > tempMax:
            tempMax = tempArray
    assert tempMax >= 0
    return tempMax


def hooverPredictor(dfAllFeatures):
    hooverModelAdd = "/home/grads/s/sorush.omidvar/CGMDataset/Hoover/HooverModel-0.8.sav"
    hooverModel = pickle.load(open(hooverModelAdd, "rb"))
    hooverModel.n_jobs = coreNumber
    participants = dfAllFeatures["Participant"].to_list()
    participants = list(set(participants))
    participants.sort()
    dfTotal = []
    fig = plt.figure(figsize=(10, 15))
    subplotCounter = 1
    for participant in participants:
        dfTemp = dfAllFeatures[dfAllFeatures["Participant"] == participant]
        cmDataMean, cmDataStd = meanSTDFinder(dfTemp)

        predictions = []
        groundTruth = []
        for counter in tqdm(range(len(dfTemp))):
            elements = dfTemp["CM"].iloc[counter]
            tempPredictions = []
            for element in elements:
                tempPredictions.append(cmNormalizerPredictor(element, cmDataMean, cmDataStd, hooverModel))
            # predictions.append(maxProbWinodFinder(tempPredictions))
            dfTemp["CM"].iloc[counter] = maxProbWinodFinder(tempPredictions)
            # groundTruth.append(dfTemp["MealLabel"].iloc[counter])
        if len(dfTotal) != 0:
            frames = [dfTotal, dfTemp]
            dfTotal = pd.concat(frames)
        else:
            dfTotal = dfTemp
        predictions = dfTemp["CM"].to_list()
        groundTruth = dfTemp["MealLabel"].to_list()
        print("Total:", len(groundTruth), "Positive windows:", np.sum(groundTruth))
        fpr, tpr, thresholds = roc_curve(groundTruth, predictions, pos_label=1)
        print(roc_auc_score(groundTruth, predictions))
        plt.subplot(3, 2, subplotCounter)
        if subplotCounter % 2 == 1:
            plt.ylabel("TPR")
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        else:
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [])
        if subplotCounter >= 5:
            plt.xlabel("FPR")
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        else:
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [])

        plt.plot(fpr, tpr, label=participant.capitalize() + " AUC=" + str(np.round(roc_auc_score(groundTruth, predictions), 3)))
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r:")
        subplotCounter += 1
    plt.suptitle("Outer Window=" + str(OUTTER_WINDOW_LENGTH.total_seconds() / 60) + " min")
    fig.tight_layout()
    fig.savefig(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-ROC.jpg")), dpi=600)
    plt.show()
    return dfTotal


dfAllFeatures = pd.read_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features.pkl")))
dfAllFeaturesHoover = hooverPredictor(dfAllFeatures)
dfAllFeaturesHoover.to_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features-AfterHoover.xlsx")), index=False)


# %%
def xgClassifier(xTrain, xTest, yTrain, yTest):
    clf = xgb.XGBClassifier(scale_pos_weight=len(yTrain) / np.sum(yTrain), n_jobs=coreNumber, n_estimators=250, max_depth=4, objective="binary:logistic", eval_metric="error",)
    clf.fit(xTrain, yTrain)

    predictionsTest = clf.predict_proba(xTest)
    predictionsTest = predictionsTest[:, 1]

    # fpr, tpr, thresholds = roc_curve(yTest, predictionsTest, pos_label=1)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC-AUC:" + str(np.round(roc_auc_score(yTest, predictionsTest), 3)))
    # plt.scatter(fpr, tpr)
    # plt.plot([0, 0], [1, 1], color="red")
    # plt.show()

    predictionsTest[predictionsTest >= 0.5] = 1
    predictionsTest[predictionsTest < 0.5] = 0

    accuracy = sklearn.metrics.accuracy_score(yTest, predictionsTest)
    recall = sklearn.metrics.recall_score(yTest, predictionsTest)
    precision = sklearn.metrics.precision_score(yTest, predictionsTest)
    f1Score = sklearn.metrics.f1_score(yTest, predictionsTest, average="weighted")
    rocAuc = roc_auc_score(yTest, predictionsTest)

    # print(np.round(recall, 3), "\t", np.round(precision, 3), "\t", np.round(f1Score, 3), "\t", np.round(rocAuc, 3), "\t", len(yTest) - np.sum(yTest), "\t", np.sum(yTest), "\t", np.round(accuracy, 3))
    return [rocAuc, accuracy, recall, precision, f1Score, len(yTest) - np.sum(yTest), np.sum(yTest)]


def testTrainSplit(dfParticipant, participant, combination, normalFlag):
    dfParticipant.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)

    yDataBinary = dfParticipant["MealLabel"].to_list()
    yDataBinary = np.asarray(yDataBinary).astype(int)

    dfParticipant.drop(columns=["StartTime", "FinishTime", "MealLabel", "Participant", "Carb", "Fat", "Protein"], inplace=True)
    xDataBinary = dfParticipant.values
    xDataBinary = np.asarray(xDataBinary).astype(float)

    classifierReport = []
    if normalFlag:
        xDataBinary -= np.mean(xDataBinary, axis=0)
        xDataBinary /= np.std(xDataBinary, axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    setNumber = 0
    for trainIndex, testIndex in skf.split(xDataBinary, yDataBinary):
        xTrain, xTest = xDataBinary[trainIndex, :], xDataBinary[testIndex, :]
        yTrain, yTest = yDataBinary[trainIndex], yDataBinary[testIndex]
        tempListReport = xgClassifier(xTrain, xTest, yTrain, yTest)
        tempListReport.extend([participant, combination, setNumber])
        classifierReport.append(tempListReport)
        setNumber += 1

    return classifierReport


def predictionMain(dfCombination, randomSeed, normalFlag, combination):
    participants = list(set(dfCombination["Participant"].to_list()))
    classifierReports = []
    # for participantCounter in tqdm(range(len(participants) + 1)):
    for participantCounter in range(len(participants)):
        if participantCounter == len(participants):  # General Model (one model for all participants)
            dfParticipant = dfCombination
            participant = "All"
        else:  # Personal Model (each participant have a his/her own model)
            participant = participants[participantCounter]
            dfParticipant = dfCombination[dfCombination["Participant"] == participant]
        dfParticipant.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
        print("*************************", "Participant:", participant)
        classifierReports.extend(testTrainSplit(dfParticipant, participant, combination, normalFlag))
    return classifierReports


def foldSummarizerBinary(df):
    combinations = list(set(df["Combination"].to_list()))
    participants = list(set(df["Participant"].to_list()))
    dfSummarizeds = []
    headers = ["Participant", "Combination", "ROC-AUC", "Accuracy", "Recall", "Precision", "F1", "TestPositive", "TestNegative", "SetNumber"]
    for participant in participants:
        for combination in combinations:
            dfSummarized = [participant, combination]
            dfTemp = df[(df["Participant"] == participant) & (df["Combination"] == combination)]
            assert len(dfTemp) == FOLD_NUMBER
            dfSummarized.extend(dfTemp.mean())
            dfSummarizeds.append(dfSummarized)
    dfSummarizeds = pd.DataFrame(dfSummarizeds, columns=headers)
    dfSummarizeds.drop(columns=["SetNumber"], inplace=True)
    dfSummarizeds.sort_values(["Participant", "Combination"], ascending=(True, True), inplace=True)
    dfSummarizeds.to_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier-Summary.xlsx"),), index=False)

    print("Outter:", str(OUTTER_WINDOW_LENGTH), "Fasting:", str(FASTING_LENGTH))
    print(("---------------------------------------------------------"))
    print(dfSummarizeds)


if os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier.xlsx"))):
    os.remove(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier.xlsx")))
if not os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier.xlsx"))):
    dfAllFeaturesHoover = pd.read_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-All-Features-AfterHoover.xlsx")))
    combinations = [["CGM"], ["CM"], ["CGM", "CM"], ["CGM", "CM", "Temperature"], ["CGM", "CM", "Temperature", "HR", "EDA"]]
    columns = dfAllFeaturesHoover.columns
    headersClassifier = ["ROC-AUC", "Accuracy", "Recall", "Precision", "F1", "TestPositive", "TestNegative", "Participant", "Combination", "SetNumber"]
    dfClassifier = []
    for combination in combinations:
        columnList = ["StartTime", "FinishTime", "Participant", "Carb", "Fat", "Protein", "MealLabel"]
        for topic in combination:
            for column in columns:
                if topic in column:
                    columnList.append(column)

        dfCombination = dfAllFeaturesHoover[dfAllFeaturesHoover.columns.intersection(columnList)]
        randomSeed = 60
        print("----------------------")
        print("Combination:", "+".join(combination))
        NORMALIZED_FLAG = True
        classifierReport = predictionMain(dfCombination, randomSeed, NORMALIZED_FLAG, "+".join(combination))
        dfTempClassifier = pd.DataFrame(classifierReport, columns=headersClassifier)
        if len(dfClassifier) > 0:
            frames = [dfTempClassifier, dfClassifier]
            dfClassifier = pd.concat(frames)
        else:
            dfClassifier = dfTempClassifier

    dfClassifier.reset_index(drop=True, inplace=True)
    dfClassifier.to_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier.xlsx")), index=False)

else:
    dfClassifier = pd.read_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(FASTING_LENGTH) + "-Final-Classifier.xlsx")))
foldSummarizerBinary(dfClassifier)



# %%
os.chdir(os.path.join(addResults))


def summaryPlotter(participant, metricType):
    metricCMList = []
    metricCGMList = []
    metricCGMCMList = []
    metricCGMCMTempList = []
    metricCGMCMTempHREDAList = []
    windowLenList = []
    for root, dirs, files in os.walk(os.path.join(addResults)):
        for file in sorted(files):
            if ".xlsx" in file.lower() and "summary" in file.lower() and "classifier" in file.lower():
                windowLen = file[: file.find("-")]

                dfTemp = pd.read_excel(file)

                metricVal = dfTemp[(dfTemp["Participant"] == participant) & (dfTemp["Combination"] == "CM")]
                metricVal = metricVal[metricType].to_list()
                assert len(metricVal) == 1
                metricVal = metricVal[0]
                metricCMList.append(metricVal)

                metricVal = dfTemp[(dfTemp["Participant"] == participant) & (dfTemp["Combination"] == "CGM")]
                metricVal = metricVal[metricType].to_list()
                assert len(metricVal) == 1
                metricVal = metricVal[0]
                metricCGMList.append(metricVal)

                metricVal = dfTemp[(dfTemp["Participant"] == participant) & (dfTemp["Combination"] == "CGM+CM")]
                metricVal = metricVal[metricType].to_list()
                assert len(metricVal) == 1
                metricVal = metricVal[0]
                metricCGMCMList.append(metricVal)

                metricVal = dfTemp[(dfTemp["Participant"] == participant) & (dfTemp["Combination"] == "CGM+CM+Temperature")]
                metricVal = metricVal[metricType].to_list()
                assert len(metricVal) == 1
                metricVal = metricVal[0]
                metricCGMCMTempList.append(metricVal)

                metricVal = dfTemp[(dfTemp["Participant"] == participant) & (dfTemp["Combination"] == "CGM+CM+Temperature+HR+EDA")]
                metricVal = metricVal[metricType].to_list()
                assert len(metricVal) == 1
                metricVal = metricVal[0]
                metricCGMCMTempHREDAList.append(metricVal)
                windowLenList.append(windowLen)

    for counter in range(len(windowLenList)):
        tempVal = datetime.strptime(windowLenList[counter], "%H:%M:%S")
        tempVal = tempVal.time().hour * 60 + tempVal.time().minute
        windowLenList[counter] = tempVal
    return metricCMList, metricCGMList, metricCGMCMList, metricCGMCMTempList, metricCGMCMTempHREDAList, windowLenList


participants = ["p1", "p3", "p5", "p6", "p7", "p8"]
subplotCounter = 1
fig = plt.figure(figsize=(10, 15))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
for participant in participants:
    metricName = "Precision"
    metricCMList, metricCGMList, metricCGMCMList, metricCGMCMTempList, metricCGMCMTempHREDAList, windowLenList = summaryPlotter(participant, metricName)
    slopeCGM, interceptCGM, r_valueCGM, p_valueCGM, std_errCGM = linregress(windowLenList, metricCGMList)
    slopeCM, interceptCM, r_valueCM, p_valueCM, std_errCM = linregress(windowLenList, metricCMList)
    slopeCGMCM, interceptCGMCM, r_valueCGMCM, p_valueCGMCM, std_errCGMCM = linregress(windowLenList, metricCGMCMList)

    print(participant, (30 * slopeCGM + interceptCGM - interceptCGMCM) / slopeCGMCM)
    # print(participant,slopeCGM,slopeCGMCM)
    # print(participant,interceptCGM,interceptCGMCM)

    plt.subplot(3, 2, subplotCounter)
    sns.regplot(x=windowLenList, y=metricCMList, marker="+", color=colors[0], label="CM")
    sns.regplot(x=windowLenList, y=metricCGMList, marker="s", color=colors[1], label="CGM")
    sns.regplot(x=windowLenList, y=metricCGMCMList, marker="d", color=colors[2], label="CGM+CM")
    # plt.plot(windowLenList, metricCGMCMTempList, "--o", color=colors[3], label="CGM+CM+Temperature")
    # plt.plot(windowLenList, metricCGMCMTempHREDAList, ":s", color=colors[4], label="CGM+CM+Temperature+HR+EDA")
    plt.text(20, 0.9, participant.capitalize())
    plt.ylim([0, 1])
    if subplotCounter == 3:
        plt.ylabel(metricName, labelpad=30)
    if subplotCounter == 5:
        # plt.xlabel("Window Length [min]",labelpad=30)
        frame1 = plt.gca()
        frame1.axes.set_xlabel("Window Length [min]", labelpad=30, x=1)
    if subplotCounter == 2:
        plt.legend(loc="upper right")
    if subplotCounter % 2 == 1:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    else:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [])
    if subplotCounter >= 5:
        plt.xticks([15, 30, 45, 60, 75, 90], ["15", "30", "45", "60", "75", "90"])
    else:
        plt.xticks([15, 30, 45, 60, 75, 90], [])

    subplotCounter += 1


fig.tight_layout()
fig.savefig(os.path.join(addResults, "Eating-ROC-AUC Summary-" + metricName + "-" + str(FASTING_LENGTH) + ".jpg"), dpi=600)
plt.show()



