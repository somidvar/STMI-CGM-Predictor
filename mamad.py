# %%
from datetime import datetime, timedelta, timezone
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, mode, skew, kurtosis, linregress
import xgboost as xgb
from joblib import Parallel, delayed
import sklearn
import pickle
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, recall_score, precision_score, accuracy_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error as MSE
import zipfile
import warnings
import seaborn as sns
import joblib
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from matplotlib import patches
import sys

# sns.set_style("white")


# plt.rcParams["text.usetex"] = True
# font = {"family": "normal", "weight": "bold", "size": 22}

# plt.rc("font", **font)

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
# CGM_LAG_IMPOSING_STR = sys.argv[1]# ONLY TO IMPOSE A TIME LAG BETWEEN CGM READINGS AND CORE MOTION DATA!!!! WATCH OUT AND USE IT CAUTIOUSLY
CGM_LAG_IMPOSING_STR = "0"

OUTTER_WINDOW_STEP = timedelta(seconds=int(sys.argv[1]))
INNER_WINDOW_LENGTH = timedelta(seconds=int(sys.argv[1]))

print("Outter Step:",str(OUTTER_WINDOW_STEP),"Inner Length:",str(INNER_WINDOW_LENGTH))


CGM_LAG_IMPOSING = timedelta(minutes=int(CGM_LAG_IMPOSING_STR))  # ONLY TO IMPOSE A TIME LAG BETWEEN CGM READINGS AND CORE MOTION DATA!!!! WATCH OUT AND USE IT CAUTIOUSLY
AVERAGING_BLOCK = timedelta(minutes=5)
OUTTER_WINDOW_LENGTH = timedelta(minutes=30)
# OUTTER_WINDOW_STEP = timedelta(seconds=30)
# INNER_WINDOW_LENGTH = timedelta(seconds=30)


FASTING_LENGTH = timedelta(minutes=30)
BIG_MEAL_CALORIE = 200
FOLD_NUMBER = 5
MINIMUM_POINT = INNER_WINDOW_LENGTH.total_seconds()
COMPLEX_MEAL_DURATION = timedelta(minutes=60)


START_OF_TRIAL = [datetime.strptime("11 06 2021-04:00:00", "%m %d %Y-%H:%M:%S"), datetime.strptime("02 03 2022-00:00:00", "%m %d %Y-%H:%M:%S")]
END_OF_TRIAL = [datetime.strptime("11 15 2021-00:00:00", "%m %d %Y-%H:%M:%S"), datetime.strptime("02 13 2022-00:00:00", "%m %d %Y-%H:%M:%S")]
DAY_LIGHT_SAVING = datetime.strptime("11 06 2021-02:00:00", "%m %d %Y-%H:%M:%S")
coreNumber = 100

addDataPrefix = "/Users/sorush/My Drive/Documents/Educational/TAMU/Research/TAMU/"
if not os.path.exists(addDataPrefix):
    addDataPrefix = "/home/grads/s/sorush.omidvar/CGMDataset/TAMU/"
if not os.path.exists(addDataPrefix):
    addDataPrefix = "C:\\GDrive\\Documents\\Educational\\TAMU\\Research\\Trial\\Data\\11-5-21-11-15-21"

addUserInput = os.path.join(addDataPrefix, "User inputted")
addHKCM = os.path.join(addDataPrefix, "hk+cm")
addCGM = os.path.join(addDataPrefix, "CGM")
addE4 = os.path.join(addDataPrefix, "E4")
addResults = os.path.join(addDataPrefix, "Results" + str(CGM_LAG_IMPOSING_STR))
if not os.path.exists(addResults):
    os.mkdir(addResults)

exempts = ["p2", "p4"]
# exempts = ["p1", "p2", "p3", "p4", "p5", "p6"]

pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use({"figure.facecolor": "white"})
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no GPU


warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 100)


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


def CGMNormalizer(df):
    participants = list(set(df["Participant"].to_list()))
    participants.sort()
    dfResult = []
    for participant in participants:
        dfParticipant = df[df["Participant"] == participant]
        dates = list(set(dfParticipant["Time"].dt.date.to_list()))
        dates.sort()
        for date in dates:
            dfDate = dfParticipant[dfParticipant["Time"].dt.date == date]
            if len(dfDate) <= 10:
                continue
            minBG = dfDate["Abbot"].min()
            maxBG = dfDate["Abbot"].max()

            meanBG = dfDate["Abbot"].mean()
            stdBG = dfDate["Abbot"].std()
            dfDate["Abbot"] -= minBG
            dfDate["Abbot"] /= maxBG - minBG
            # dfDate["Abbot"] -= meanBG
            # dfDate["Abbot"] /= stdBG

            assert not np.isnan(minBG)
            assert not np.isnan(maxBG)
            assert not np.isnan(meanBG)
            assert not np.isnan(stdBG)
            if len(dfResult) == 0:
                dfResult = dfDate.copy()
            else:
                frames = [dfResult, dfDate]
                dfResult = pd.concat(frames)
    return dfResult


def CGMLowPass(df):
    participants = list(set(df["Participant"].to_list()))
    participants.sort()
    dfResult = []
    for participant in participants:
        dfParticipant = df[df["Participant"] == participant]
        cgmVals = dfParticipant["Abbot"].to_list()
        cgmVals = np.asarray(cgmVals)
        lowPassFilter = signal.butter(3, 12, "lp", fs=60 * 24, output="sos")  # high pass of period of 2 hours (12 per day)
        cgmVals = signal.sosfilt(lowPassFilter, cgmVals)
        dfParticipant["Abbot"] = cgmVals
        if len(dfResult) == 0:
            dfResult = dfParticipant.copy()
        else:
            frames = [dfResult, dfParticipant]
            dfResult = pd.concat(frames)
    return dfResult


if os.path.exists(os.path.join(addResults, "All_cgm.pkl")):
    os.remove(os.path.join(addResults, "All_cgm.pkl"))
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
                    dfTemp.reset_index(drop=True, inplace=True)
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
    # dfCGM = CGMLowPass(dfCGM)
    # dfCGM = CGMNormalizer(dfCGM)
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
                    dfTemp.reset_index(drop=True, inplace=True)

                    dfTemp = cmSmoother(dfTemp)
                    dfTemp["Yaw"] *= 180 / np.pi
                    dfTemp["Pitch"] *= 180 / np.pi
                    dfTemp["Roll"] *= 180 / np.pi

                    dfTemp.insert(0, "Participant", participantName)
                    # this is to avoid 0 later on for feature calculation
                    dfTemp.insert(len(dfTemp.columns), "|Ax|+|Ay|+|Az|", dfTemp["Ax"].abs() + dfTemp["Ay"].abs() + dfTemp["Az"].abs() + 0.0001)
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
    dfCM = trialTimeLimitter(dfCM, "Time")
    dfCM.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfCM.reset_index(drop=True, inplace=True)
    print("CM database is limited to the trial period")
    dfCM.to_pickle(os.path.join(addResults, "All_cm.pkl"))
else:
    dfCM = pd.read_pickle(os.path.join(addResults, "All_cm.pkl"))


def E4Smoother(df):
    dfE4EDA = df[df["Field"] == "EDA"]
    dfE4HR = df[df["Field"] == "HR"]
    dfE4Temperature = df[df["Field"] == "Temperature"]

    tempSerie = dfE4EDA["Data1"]
    tempSerie = tempSerie.ewm(span=5 * 4).mean()  # Considering the frequency of 4 Hz
    dfE4EDA["Data1"] = tempSerie

    tempSerie = dfE4HR["Data1"]
    tempSerie = tempSerie.ewm(span=5).mean()  # Considering the frequency of 1 Hz
    dfE4HR["Data1"] = tempSerie

    tempSerie = dfE4Temperature["Data1"]
    tempSerie = tempSerie.ewm(span=5 * 4).mean()  # Considering the frequency of 4 Hz
    dfE4Temperature["Data1"] = tempSerie
    frames = [dfE4EDA, dfE4HR, dfE4Temperature]
    df = pd.concat(frames)

    return df


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
                dfTemp.reset_index(drop=True, inplace=True)
                dfTemp = E4Smoother(dfTemp)
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
    dfE4.reset_index(drop=True, inplace=True)
    dfE4 = trialTimeLimitter(dfE4, "Time")
    dfE4.sort_values(["Participant", "Time"], ascending=(True, True), inplace=True)
    dfE4.reset_index(drop=True, inplace=True)
    print("E4 database is limited to the trial period")
    dfE4.to_pickle(os.path.join(addResults, "All_E4.pkl"))
else:
    dfE4 = pd.read_pickle(os.path.join(addResults, "All_E4.pkl"))


# %%
def motionCalculator(df):
    f1 = np.asarray(df["RotationalToLinear"].to_list())
    f2 = np.asarray(df["|Ax|+|Ay|+|Az|"].to_list())

    aX = np.asarray(df["Ax"].to_list())
    aY = np.asarray(df["Ay"].to_list())
    aZ = np.asarray(df["Az"].to_list())

    yaw = np.asarray(df["Yaw"].to_list())
    pitch = np.asarray(df["Pitch"].to_list())
    roll = np.asarray(df["Roll"].to_list())

    # featureData = [f1.mean(skipna=True), f1.std(skipna=True), f1.max(skipna=True) - f1.min(skipna=True), f2.mean(skipna=True), f2.std(skipna=True), f2.max(skipna=True) - f2.min(skipna=True)]
    featureData = [
        np.nanmean(f1),
        np.nanstd(f1),
        np.nanmax(f1),
        np.nanmin(f1),
        np.nanmax(f1) - np.nanmin(f1),
        np.nanmean(f2),
        np.nanstd(f2),
        np.nanmax(f2),
        np.nanmin(f2),
        np.nanmax(f1) - np.nanmin(f1),
        np.nanmean(aX),
        np.nanstd(aX),
        np.nanmax(aX),
        np.nanmin(aX),
        np.nanmax(aX) - np.nanmin(aX),
        np.nanmean(aY),
        np.nanstd(aY),
        np.nanmax(aY),
        np.nanmin(aY),
        np.nanmax(aY) - np.nanmin(aY),
        np.nanmean(aZ),
        np.nanstd(aZ),
        np.nanmax(aZ),
        np.nanmin(aZ),
        np.nanmax(aZ) - np.nanmin(aZ),
        np.nanmean(yaw),
        np.nanstd(yaw),
        np.nanmax(yaw),
        np.nanmin(yaw),
        np.nanmax(yaw) - np.nanmin(yaw),
        np.nanmean(pitch),
        np.nanstd(pitch),
        np.nanmax(pitch),
        np.nanmin(pitch),
        np.nanmax(pitch) - np.nanmin(pitch),
        np.nanmean(roll),
        np.nanstd(roll),
        np.nanmax(roll),
        np.nanmin(roll),
        np.nanmax(roll) - np.nanmin(roll),
    ]
    assert len(featureData) == 40
    return featureData


def CGMStatFeatures(dataList):
    # nanList = []
    # for counter in range(24):
    #     nanList.extend([-1000])
    # return nanList
    # assert len(dataList) >= int(OUTTER_WINDOW_LENGTH.seconds / INNER_WINDOW_LENGTH.seconds) - 1
    assert len(dataList) >= 10

    dataList = np.asarray(dataList).astype(float)
    result = []
    dataDim = dataList.ndim
    assert dataDim == 1
    assert len(dataList) == len(dataList[~np.isnan(dataList)])
    dataList = dataList[~np.isnan(dataList)]

    meanVal = np.nanmean(dataList)
    stdVal = np.nanstd(dataList)
    minVal = np.nanmin(dataList)
    maxVal = np.nanmax(dataList)
    rangeVal = maxVal - minVal
    skewnessVal = skew(dataList, nan_policy="omit")
    kurtosisVal = kurtosis(dataList, nan_policy="omit")

    tempSize = int(len(dataList) / 4)
    firstFourthSlopeVal = np.mean(dataList[0:tempSize])
    secondFourthSlopeVal = np.mean(dataList[tempSize : 2 * tempSize])
    thirdFourthSlopeVal = np.mean(dataList[2 * tempSize : 3 * tempSize])
    forthFourthSlopeVal = np.mean(dataList[3 * tempSize :])
    firstHalfSlopeVal = np.mean(dataList[0 : 2 * tempSize])
    secondHalfSlopeVal = np.mean(dataList[2 * tempSize :])

    dataListDiff = np.diff(dataList)
    meanDiff = np.nanmean(dataListDiff)
    stdDiff = np.nanstd(dataListDiff)
    minDiff = np.nanmin(dataListDiff)
    maxDiff = np.nanmax(dataListDiff)
    rangeDiff = maxDiff - minDiff
    skewnessDiff = skew(dataListDiff, nan_policy="omit")
    kurtosisDiff = kurtosis(dataListDiff, nan_policy="omit")

    tempSize = int(len(dataListDiff) / 4)
    firstFourthSlopeDiff = np.mean(dataListDiff[0:tempSize])
    secondFourthSlopeDiff = np.mean(dataListDiff[tempSize : 2 * tempSize])
    thirdFourthSlopeDiff = np.mean(dataListDiff[2 * tempSize : 3 * tempSize])
    forthFourthSlopeDiff = np.mean(dataListDiff[3 * tempSize :])
    firstHalfSlopeDiff = np.mean(dataList[0 : 2 * tempSize])
    secondHalfSlopeDiff = np.mean(dataList[2 * tempSize :])

    result.extend([rangeVal, meanVal, stdVal, minVal, maxVal, skewnessVal, kurtosisVal, firstFourthSlopeVal, secondFourthSlopeVal, thirdFourthSlopeVal, forthFourthSlopeVal, secondHalfSlopeVal - firstHalfSlopeVal])
    result.extend([rangeDiff, meanDiff, stdDiff, minDiff, maxDiff, skewnessDiff, kurtosisDiff, firstFourthSlopeDiff, secondFourthSlopeDiff, thirdFourthSlopeDiff, forthFourthSlopeDiff, secondHalfSlopeDiff - firstHalfSlopeDiff])
    return result


def E4StatFeatures(df, sensor):
    nanList = []
    for counter in range(14 * 1):
        nanList.extend([np.nan])

    dfTemp = df[df["Field"] == sensor]
    tempVal = dfTemp["Data1"].to_list()

    if len(tempVal) < 10:
        return nanList
    else:
        tempVal = np.asarray(tempVal).astype(float)
        tempVal = tempVal[~np.isnan(tempVal)]

        meanVal = np.nanmean(tempVal)
        stdVal = np.nanstd(tempVal)
        minVal = np.nanmin(tempVal)
        maxVal = np.nanmax(tempVal)
        rangeVal = maxVal - minVal
        skewnessVal = skew(tempVal, nan_policy="omit")
        kurtosisVal = kurtosis(tempVal, nan_policy="omit")

        dataTempDiff = np.diff(tempVal)
        meanDiff = np.nanmean(dataTempDiff)
        stdDiff = np.nanstd(dataTempDiff)
        minDiff = np.nanmin(dataTempDiff)
        maxDiff = np.nanmax(dataTempDiff)
        rangeDiff = maxDiff - minDiff
        skewnessDiff = skew(dataTempDiff, nan_policy="omit")
        kurtosisDiff = kurtosis(dataTempDiff, nan_policy="omit")
        return [rangeVal, meanVal, stdVal, minVal, maxVal, skewnessVal, kurtosisVal, rangeDiff, meanDiff, stdDiff, minDiff, maxDiff, skewnessDiff, kurtosisDiff]


def parallelCall(windowData):
    tempList = []
    outterWindowStart = windowData[0]
    outterWindowEnd = windowData[1]
    mealFlag = windowData[2]
    participant = windowData[3]
    mealStartList = windowData[4]
    mealEndList = windowData[5]

    nanList = []
    for counter in range(40 * 1):
        nanList.extend([np.nan])

    dfTempCM = dfParticipantCM[(dfParticipantCM["Time"] >= outterWindowStart) & (dfParticipantCM["Time"] < outterWindowEnd)]
    dfTempE4 = dfParticipantE4[(dfParticipantE4["Time"] >= outterWindowStart) & (dfParticipantE4["Time"] < outterWindowEnd)]
    dfTempCGM = dfParticipantCGM[
        (dfParticipantCGM["Time"] >= outterWindowStart) & (dfParticipantCGM["Time"] < outterWindowEnd + OUTTER_WINDOW_LENGTH)
    ]  # We are looking forward. For example if the meal is started at 12.10 we are interested to see the data from 12.10 to 12.40 (assuming that the outer window length is 30 min)
    if len(dfTempCM) >= MINIMUM_POINT * 10 * 0.5:  # Minimum point is the length of inner window per sec and this results in 50% of data
        listCM = motionCalculator(dfTempCM)
    else:
        listCM = nanList

    listEDA = E4StatFeatures(dfTempE4, "EDA")
    listHR = E4StatFeatures(dfTempE4, "HR")
    listTemperature = E4StatFeatures(dfTempE4, "Temperature")

    listCGM = CGMStatFeatures(dfTempCGM["Abbot"].to_list())

    tempList.append(listCM)  # 1
    tempList.append(listEDA)  # 1
    tempList.append(listHR)  # 1
    tempList.append(listTemperature)  # 1

    tempList.append(listCGM)  # 1

    tempList.append(outterWindowStart)  # 1
    tempList.append(outterWindowEnd)  # 1
    tempList.append(participant)  # 1

    tempList.append(mealFlag)  # mealFlag
    tempList.append(mealStartList)  # mealStarts
    tempList.append(mealEndList)  # mealEnds

    assert len(tempList) == 11
    return tempList


def outterWindowExtractorTotal(participant):
    participantDataList = []
    windowDatas = []
    experimentStart = dfParticipantCM["Time"].min()
    experimentEnd = dfParticipantCM["Time"].max()

    startQuerry = experimentStart
    endQuerry = startQuerry + OUTTER_WINDOW_STEP
    while endQuerry <= experimentEnd:
        dfTempMeal = dfParticipantMeal[(dfParticipantMeal["StartTime"] <= startQuerry) & (dfParticipantMeal["StartTime"] + timedelta(minutes=15) >= startQuerry)]
        # dfTempMeal = dfParticipantMeal[(dfParticipantMeal["StartTime"] >= startQuerry) & (dfParticipantMeal["StartTime"] + timedelta(minutes=15) < endQuerry)]
        mealFlag = min(len(dfTempMeal), 1)
        mealStartList = dfTempMeal["StartTime"].to_list()
        mealEndList = dfTempMeal["FinishTime"].to_list()

        windowDatas.append([startQuerry, endQuerry, mealFlag, participant, mealStartList, mealEndList])
        startQuerry += OUTTER_WINDOW_STEP
        endQuerry += OUTTER_WINDOW_STEP
    skipNumber = int(OUTTER_WINDOW_LENGTH.seconds / INNER_WINDOW_LENGTH.seconds)
    windowDatas = windowDatas[: len(windowDatas) - skipNumber]  # Skipping the last 30 windows (assuming outter of 30 and inner of 1 min) because the last window has not enough points for looking back and this causes issues in stat calculation
    # for counterOuter in tqdm(range(int(len(windowDatas)))):
    #     windowData = windowDatas[counterOuter]
    #     participantDataList.append(parallelCall(windowData))

    # participantDataList = Parallel(n_jobs=coreNumber,batch_size=80)(delayed(parallelCall)(windowData) for windowData in tqdm(windowDatas))

    pool = mp.Pool(coreNumber)
    participantDataList = pool.map(parallelCall, tqdm(windowDatas), chunksize=80)
    pool.close()
    return participantDataList


if os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl"))):
    os.remove(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl")))
if os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl"))):
    dfAllFeatures = pd.read_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl")))
else:
    dfAllFeatures = []
    participants = dfMeal["Participant"].to_list()
    participants = list(set(participants))
    participants.sort()

    columnHeaderList = ["CM", "EDA", "HR", "Temperature", "CGM", "StartTime", "FinishTime", "Participant", "MealLabel", "MealStartList", "MealEndList"]
    for participant in participants:
        print("Participant:", participant)
        if participant in exempts:
            continue
        dfParticipantMeal = dfMeal[dfMeal["Participant"] == participant]
        dfParticipantCM = dfCM[dfCM["Participant"] == participant]
        dfParticipantE4 = dfE4[dfE4["Participant"] == participant]
        dfParticipantCGM = dfCGM[dfCGM["Participant"] == participant]
        participantDataList = outterWindowExtractorTotal(participant)

        participantDataList = pd.DataFrame(participantDataList, columns=columnHeaderList)
        if len(dfAllFeatures) == 0:
            dfAllFeatures = participantDataList
        else:
            frames = [dfAllFeatures, participantDataList]
            dfAllFeatures = pd.concat(frames)
    dfAllFeatures.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
    dfAllFeatures.reset_index(drop=True, inplace=True)
    dfAllFeatures.to_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl")))


# %%
def xgClassifier(xTrain, xTest, yTrain, yTest, featturesName):
    thresholdBest = np.nan
    clf = xgb.XGBClassifier(scale_pos_weight=len(yTrain) / np.sum(yTrain), n_jobs=coreNumber, n_estimators=250, max_depth=3, objective="binary:logistic", eval_metric="error")
    clf.fit(xTrain, yTrain)
    # predVal = clf.predict(xVal)
    # f1Best = sklearn.metrics.f1_score(yVal, predVal)
    # thresholdBest = 0.5

    # for threshold in np.arange(0.1, 0.9, 0.01):
    #     predVal = clf.predict_proba(xVal)
    #     predVal = predVal[:, 1]
    #     predVal[predVal >= threshold] = 1
    #     predVal[predVal < threshold] = 0

    #     f1Val = sklearn.metrics.f1_score(yVal, predVal)
    #     if f1Val > 1.1 * f1Best:
    #         thresholdBest = threshold
    #         f1Best = f1Val

    predictionsTest = clf.predict_proba(xTest)
    predictionsTest = predictionsTest[:, 1]

    # itemIndex = np.where(yTest == 1)
    # itemIndex = itemIndex[0]
    # itemIndex = itemIndex[0]

    # predictionsTest = clf.predict_proba(xTest)
    # predictionsTest = predictionsTest[:, 1]
    # mamad=np.copy(predictionsTest)
    # for counter in range(len(mamad)-5):
    #     mamad[counter]=np.mean(predictionsTest[counter:counter+5])
    # plt.figure(figsize=(20, 10))
    # plt.plot(yTest[itemIndex: itemIndex + 2500], color="r")
    # plt.plot(predictionsTest[itemIndex: itemIndex + 2500], color="b")
    # plt.plot(mamad[itemIndex: itemIndex + 2500], color="g",linewidth=7)

    # plt.show()
    # raise

    featureImportance = clf.feature_importances_
    featureImportance = dict(zip(featturesName, featureImportance))

    return [thresholdBest, np.sum(yTest), len(yTest) - np.sum(yTest), featureImportance, yTest, predictionsTest]


def featureImportanceAverager(featureImportances):
    tempVals = []
    tempLabels = []
    for featureImportance in featureImportances:
        tempVals.append(list(featureImportance.values()))
        tempLabels = list(featureImportance.keys())
    featureImportances = np.asarray(tempVals)
    featureImportances = np.mean(featureImportances, axis=0)
    featureImportances = dict(zip(tempLabels, featureImportances))
    featureImportances = dict(sorted(featureImportances.items(), key=lambda item: item[1], reverse=True))
    featureImportances = [[featureImportances]]

    return featureImportances


def xDataGetter(dfParticipant, sensor):
    tempList = dfParticipant[sensor].to_list()
    tempList = np.asarray(tempList)
    return tempList


def testTrainSplit(dfParticipant, combinationList, NORMAL_FLAG, HOOVER_FLAG, featturesName):
    dfParticipant.insert(len(dfParticipant.columns), "MultiModalProb", np.nan)
    dfParticipant.insert(len(dfParticipant.columns), "Threshold", np.nan)
    dfParticipant.reset_index(drop=True, inplace=True)
    yData = dfParticipant["MealLabel"].to_list()
    yData = np.asarray(yData).astype(int)
    for combinationElement in combinationList:
        if combinationElement == "CM" and HOOVER_FLAG:
            continue  # Already handled by HooverModel
        xData = xDataGetter(dfParticipant, combinationElement)

        if NORMAL_FLAG:
            xData -= np.nanmean(xData, axis=0)
            xData /= np.nanstd(xData, axis=0)
        kf = KFold(n_splits=FOLD_NUMBER, shuffle=False)
        predictionTests = []
        for trainIndex, testIndex in kf.split(xData, yData):
            xTrain, xTest = xData[trainIndex, :], xData[testIndex, :]
            yTrain, yTest = yData[trainIndex], yData[testIndex]

            tempListReport = xgClassifier(xTrain, xTest, yTrain, yTest, featturesName)
            # tempListReport = lrClassifier(xTrain, xTest, yTrain, yTest,featturesName)
            # tempListReport = rfClassifier(xTrain, xTest, yTrain, yTest,featturesName)

            predictionTests.extend(tempListReport[-1])
        predictionTests = np.asarray(predictionTests)
        dfParticipant[combinationElement] = predictionTests
    blockAverager(dfParticipant, combinationList)
    multiModalModel(dfParticipant, combinationList)


def multiModalModel(df, combinationList):
    xData = []
    yData = df["MealLabel"].to_list()
    yData = np.asarray(yData)

    xData = df[combinationList].values
    xData = np.asarray(xData)

    kf = KFold(n_splits=FOLD_NUMBER, shuffle=False)
    predictionsTests = []
    thresholds = []

    for trainIndex, testIndex in kf.split(xData, yData):
        xTrain, xTest = xData[trainIndex, :], xData[testIndex, :]
        yTrain, yTest = yData[trainIndex], yData[testIndex]

        xVal = xTrain[int(0.8 * len(xTrain)) :, :]
        yVal = yTrain[int(0.8 * len(yTrain)) :]

        xTrain = xTrain[: int(0.8 * len(xTrain)), :]
        yTrain = yTrain[: int(0.8 * len(yTrain))]

        clf = LogisticRegression(class_weight="balanced", n_jobs=coreNumber)
        clf.fit(xTrain, yTrain)

        thresholdBest = -1
        f1Best = -1
        for threshold in np.arange(0, 1, 0.01):
            predVal = clf.predict_proba(xVal)
            predVal = predVal[:, 1]
            predVal[predVal >= threshold] = 1
            predVal[predVal < threshold] = 0
            f1Score = sklearn.metrics.f1_score(yVal, predVal)
            if f1Score >= f1Best:
                thresholdBest = threshold
                f1Best = f1Score

        predictionsTest = clf.predict_proba(xTest)
        predictionsTest = predictionsTest[:, 1]

        for counter in range(len(yTest)):
            thresholds.append(thresholdBest)

        predictionsTests.extend(predictionsTest)
    df["MultiModalProb"] = predictionsTests
    df["Threshold"] = thresholds


def featuresNameGetter(combinationString):
    CGMStat = ["-Range", "-Mean", "-STD", "-Min", "-Max", "-Skewness", "-Kurtosis", "-FirstFourthSlope", "-SecondFourthSlope", "-ThirdFourthSlope", "-FourthFourthSlope", "-HalvesSlope"]
    CGMStat.extend(["-RangeDiff", "-STDDiff", "-MinDiff", "-MaxDiff", "-SkewnessDiff", "-KurtosisDiff", "-FirstFourthSlopeDiff", "-SecondFourthSlopeDiff", "-ThirdFourthSlopeDiff", "-FourthFourthSlopeDiff", "-HalvesSlopeDiff"])
    E4Stat = ["-RangeVal", "-MeanVal", "-StdVal", "-MinVal", "-MaxVal", "-SkewnessVal", "-KurtosisVal", "-RangeDiff", "-MeanDiff", "-StdDiff", "-MinDiff", "-MaxDiff", "-SkewnessDiff", "-KurtosisDiff"]
    CMStat = ["-f1Mean", "-f1Std", "-f1Range", "-f2Mean", "-f2Std", "-f2Range"]

    statFeatures = []
    if "CGM" in combinationString:
        statFeatures.extend(("CGM" + element for element in CGMStat))

    if "CM" in combinationString:
        statFeatures.extend(("CM" + element for element in CMStat))

    if "EDA" in combinationString:
        statFeatures.extend(("EDA" + element for element in E4Stat))

    if "HR" in combinationString:
        statFeatures.extend(("HR" + element for element in E4Stat))

    if "Temperature" in combinationString:
        statFeatures.extend(("Temperature" + element for element in E4Stat))

    return statFeatures


def hooverPredictor(dfAllFeatures, NORMAL_FLAG):
    xData = dfAllFeatures["CM"].to_list()
    xData = np.asarray(xData)
    if NORMAL_FLAG:
        xData -= np.nanmean(xData, axis=0)
        xData /= np.nanstd(xData, axis=0)

    hooverModelAdd = "/home/grads/s/sorush.omidvar/CGMDataset/Hoover/HooverModel-0.5.sav"
    hooverModel = pickle.load(open(hooverModelAdd, "rb"))
    hooverModel.n_jobs = coreNumber

    hooverPredictions = hooverModel.predict_proba(xData)
    hooverPredictions = hooverPredictions[:, 1]

    hooverPredictions = np.asarray(hooverPredictions)
    dfAllFeatures["CM"] = hooverPredictions


def dfReader(combinationString):
    dfAllFeatures = pd.read_pickle(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-All-Features.pkl")))
    dfAllFeatures = dfAllFeatures.dropna()
    featturesName = featuresNameGetter(combinationString)
    return featturesName, dfAllFeatures


def blockAverager(df, combinationList):
    for counter in tqdm(range(len(df))):
        startTime = df["StartTime"].iloc[counter]
        dfTemp = df[(df["StartTime"] >= startTime) & (df["FinishTime"] <= startTime + AVERAGING_BLOCK)]

        mealLabel = dfTemp["MealLabel"].mean()
        if mealLabel >= 0.5:
            mealLabel = 1
        else:
            mealLabel = 0
        df["MealLabel"].iloc[counter] = mealLabel
        for combinationElement in combinationList:
            df[combinationElement].iloc[counter] = dfTemp[combinationElement].mean()


def metricCalc(df, combinationList):
    participants = df["Participant"].unique()
    combinationSummary = []
    for participant in participants:
        dfParticipant = df[df["Participant"] == participant]
        yTrues = dfParticipant["MealLabel"].to_list()
        yPreds = dfParticipant["MultiModalProb"].to_list()
        thresholds = dfParticipant["Threshold"].to_list()

        yTrues = np.asarray(yTrues).astype(int)
        yPreds = np.asarray(yPreds)
        thresholds = np.asarray(thresholds)

        rocAuc = roc_auc_score(yTrues, yPreds)
        precisionTemp, recallTemp, thresholdsTemp = precision_recall_curve(yTrues, yPreds)
        prAuc = auc(recallTemp, precisionTemp)

        for counter in range(len(yTrues)):
            if yPreds[counter] >= thresholds[counter]:
                yPreds[counter] = 1
            else:
                yPreds[counter] = 0

        accuracy = accuracy_score(yTrues, yPreds)
        recall = recall_score(yTrues, yPreds)
        precision = precision_score(yTrues, yPreds)
        f1Score = f1_score(yTrues, yPreds)

        combinationSummary.append([participant, "+".join(combinationList), rocAuc, prAuc, accuracy, recall, precision, f1Score])
    combinationSummary = pd.DataFrame(combinationSummary, columns=["Participant", "Combination", "ROC-AUC", "PR-AUC", "Accuracy", "Recall", "Precision", "F1"])
    return combinationSummary


def predictionMain(dfCombination, combinationList, NORMAL_FLAG, HOOVER_FLAG, featturesName):
    participants = dfCombination["Participant"].unique()
    participants.sort()
    classifierReports = []
    for participant in participants:
        # if participant != "p8":
        #     continue
        dfParticipant = dfCombination[dfCombination["Participant"] == participant]
        dfParticipant.sort_values(["StartTime"], ascending=(True), inplace=True)
        print("*************************", "Participant:", participant)
        testTrainSplit(dfParticipant, combinationList, NORMAL_FLAG, HOOVER_FLAG, featturesName)
        if len(classifierReports) == 0:
            classifierReports = dfParticipant
        else:
            frames = [classifierReports, dfParticipant]
            classifierReports = pd.concat(frames)
    return classifierReports


NORMAL_FLAG = True
HOOVER_FLAG = False
if os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-Final-Classifier.xlsx"))):
    os.remove(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-Final-Classifier.xlsx")))
if not os.path.exists(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-Final-Classifier.xlsx"))):
    dfClassifier = []
    # combinationLists = [["CGM", "CM", "EDA"], ["CGM", "CM"], ["CGM"], ["CM"]]
    combinationLists = [["CM"], ["CGM"], ["EDA"], ["CM", "CGM"], ["CM", "CGM", "EDA"], ["CM", "CGM", "EDA", "HR"], ["CM", "CGM", "EDA", "HR", "Temperature"]]
    combinationLists = [["CM"], ["CGM"], ["CM", "CGM"]]
    combinationSummary = []
    for combinationList in combinationLists:
        featturesName, dfAllFeatures = dfReader(combinationList)
        print("Total Number of Samples:", len(dfAllFeatures), "for combination:", combinationList)
        if "CM" in combinationList and HOOVER_FLAG:
            hooverPredictor(dfAllFeatures, NORMAL_FLAG)
        dfAllFeatures = predictionMain(dfAllFeatures, combinationList, NORMAL_FLAG, HOOVER_FLAG, featturesName)
        dfAllFeatures.sort_values(["Participant", "StartTime"], ascending=(True, True), inplace=True)
        dfAllFeatures.reset_index(drop=True, inplace=True)
        combinationSummaryTemp = metricCalc(dfAllFeatures, combinationList)
        if len(combinationSummary) == 0:
            combinationSummary = combinationSummaryTemp
        else:
            frames = [combinationSummary, combinationSummaryTemp]
            combinationSummary = pd.concat(frames)

    combinationSummary.to_excel(os.path.join(addResults, (str(OUTTER_WINDOW_LENGTH) + "-" + str(INNER_WINDOW_LENGTH) + "-Final-Classifier.xlsx")), index=False)


# %%
# fig = plt.figure(figsize=(30, 10))
#     gs = GridSpec(2, 5)
#     colors = plt.cm.get_cmap("tab10")

#     ax_calibration_curve = fig.add_subplot(gs[:2, :2])
#     calibration_displays = {}


#     display = CalibrationDisplay.from_estimator(
#         clf,
#         xVal,
#         yVal,
#         n_bins=10,
#         name='testmodel',
#         ax=ax_calibration_curve,
#         color=colors(0),
#         strategy='quantile'
#     )
#     calibration_displays['testmodel'] = display

#     ax_calibration_curve.grid()
#     ax_calibration_curve.set_title("Calibration plots")

#     # Add histogram
#     grid_positions = [(0, 2)]
#     # for i, m in enumerate(models):
#     name = 'CLF'
#     row, col = grid_positions[0]
#     ax = fig.add_subplot(gs[row, col])

#     ax.hist(
#         calibration_displays['testmodel'].y_prob,
#         range=(0, 1),
#         bins=10,
#         label=name,
#         color=colors(0)
#     )
#     ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

#     plt.tight_layout()
#     ax_calibration_curve.legend(loc='upper left')
#     plt.show()


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


def metricPlotter(metricName):
    participants = ["p1", "p3", "p5", "p6", "p7", "p8"]
    subplotCounter = 1
    fig = plt.figure(figsize=(10, 15))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for participant in participants:
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
    fig.savefig(os.path.join(addResults, "Eating-ROC-AUC Summary-" + metricName + "-" + str(INNER_WINDOW_LENGTH) + ".jpg"), dpi=600)
    plt.show()


metricPlotter(metricName="Recall")
metricPlotter(metricName="Precision")


# %%
# noNormal=[{'-MaxDiff': 0.1078463, '-SecondFourthSlopeDiff': 0.103150845, '-FourthFourthSlopeDiff': 0.07462696, '-ThirdFourthSlope': 0.06913031, '-Skewness': 0.056625057, '-SkewnessDiff': 0.050760787, '-STDDiff': 0.048788596, '-FirstFourthSlopeDiff': 0.046131063, '-RangeDiff': 0.04021073, '-HalvesSlopeDiff': 0.04018015, '-MinDiff': 0.033765636, '-HalvesSlope': 0.03357451, '-SecondFourthSlope': 0.033038698, '-FourthFourthSlope': 0.029650774, '-FirstFourthSlope': 0.02836506, '-Min': 0.025608739, '-Max': 0.024943182, '-STD': 0.024530888, '-Mean': 0.02407294, '-Range': 0.023151893, '-MeanDiff': 0.022112701, '-KurtosisDiff': 0.02190013, '-ThirdFourthSlopeDiff': 0.02160139, '-Kurtosis': 0.016232677},
# {'-MinDiff': 0.05836166, '-Max': 0.05825951, '-Mean': 0.056890287, '-FirstFourthSlopeDiff': 0.054751, '-SecondFourthSlope': 0.05199795, '-FirstFourthSlope': 0.051083237, '-RangeDiff': 0.047590412, '-FourthFourthSlopeDiff': 0.046937533, '-ThirdFourthSlope': 0.04655046, '-HalvesSlope': 0.044784807, '-Min': 0.043269206, '-SkewnessDiff': 0.042040505, '-MaxDiff': 0.041865278, '-HalvesSlopeDiff': 0.041717757, '-Skewness': 0.037697215, '-FourthFourthSlope': 0.034438096, '-MeanDiff': 0.03392487, '-STDDiff': 0.033244412, '-Range': 0.032868195, '-SecondFourthSlopeDiff': 0.031723518, '-KurtosisDiff': 0.029031616, '-Kurtosis': 0.028224358, '-ThirdFourthSlopeDiff': 0.027494926, '-STD': 0.025253225},
# {'-Range': 0.07485981, '-MaxDiff': 0.07235001, '-FourthFourthSlope': 0.061106034, '-Min': 0.06102726, '-ThirdFourthSlope': 0.05558133, '-SkewnessDiff': 0.054457445, '-HalvesSlope': 0.051805902, '-RangeDiff': 0.050594293, '-FirstFourthSlope': 0.04973889, '-FourthFourthSlopeDiff': 0.049186327, '-Mean': 0.046296857, '-Kurtosis': 0.04171118, '-SecondFourthSlopeDiff': 0.03683889, '-MeanDiff': 0.034761127, '-Max': 0.03353867, '-MinDiff': 0.03286376, '-SecondFourthSlope': 0.03176475, '-STDDiff': 0.028456414, '-KurtosisDiff': 0.028452437, '-ThirdFourthSlopeDiff': 0.024667135, '-FirstFourthSlopeDiff': 0.022733245, '-Skewness': 0.021999788, '-STD': 0.020657314, '-HalvesSlopeDiff': 0.014551135},
# {'-ThirdFourthSlope': 0.116321504, '-FourthFourthSlope': 0.06946923, '-MaxDiff': 0.06276192, '-Max': 0.059627842, '-Mean': 0.052711405, '-Range': 0.050306994, '-MinDiff': 0.049013417, '-RangeDiff': 0.048471592, '-Skewness': 0.047412317, '-FourthFourthSlopeDiff': 0.046264015, '-STDDiff': 0.040881716, '-SkewnessDiff': 0.038825456, '-SecondFourthSlope': 0.037260063, '-Min': 0.033757806, '-MeanDiff': 0.033217307, '-HalvesSlope': 0.031427175, '-KurtosisDiff': 0.03128091, '-FirstFourthSlope': 0.028787797, '-STD': 0.027287915, '-FirstFourthSlopeDiff': 0.024499362, '-Kurtosis': 0.021792239, '-SecondFourthSlopeDiff': 0.01951169, '-ThirdFourthSlopeDiff': 0.01877995, '-HalvesSlopeDiff': 0.010330347},
# {'-MaxDiff': 0.06764946, '-STD': 0.056324393, '-FourthFourthSlopeDiff': 0.055581857, '-Mean': 0.051104046, '-Range': 0.0502673, '-SkewnessDiff': 0.049853936, '-RangeDiff': 0.044252936, '-Min': 0.043754313, '-Max': 0.04371197, '-MinDiff': 0.042753506, '-KurtosisDiff': 0.039921276, '-HalvesSlopeDiff': 0.03888581, '-FirstFourthSlope': 0.038707566, '-FourthFourthSlope': 0.03850367, '-STDDiff': 0.038403533, '-Skewness': 0.038231872, '-SecondFourthSlope': 0.03812664, '-SecondFourthSlopeDiff': 0.037757747, '-Kurtosis': 0.037050135, '-ThirdFourthSlopeDiff': 0.03525641, '-ThirdFourthSlope': 0.032925814, '-MeanDiff': 0.031699583, '-HalvesSlope': 0.028091868, '-FirstFourthSlopeDiff': 0.021184346},
# {'-FourthFourthSlopeDiff': 0.07831435, '-SkewnessDiff': 0.06162001, '-Min': 0.050464742, '-SecondFourthSlopeDiff': 0.049488824, '-RangeDiff': 0.049330417, '-Range': 0.04861606, '-MaxDiff': 0.04708212, '-FourthFourthSlope': 0.046744823, '-KurtosisDiff': 0.046689652, '-STDDiff': 0.04431532, '-FirstFourthSlopeDiff': 0.042319566, '-MeanDiff': 0.041090157, '-MinDiff': 0.039877523, '-Skewness': 0.039017998, '-SecondFourthSlope': 0.038478997, '-Max': 0.03784959, '-Mean': 0.03724398, '-FirstFourthSlope': 0.037229583, '-ThirdFourthSlopeDiff': 0.033134893, '-Kurtosis': 0.031625576, '-HalvesSlope': 0.030476, '-ThirdFourthSlope': 0.026068594, '-STD': 0.023853524, '-HalvesSlopeDiff': 0.01906771}]

# for counter in range(len(noNormal)):
#     temp=noNormal[counter]
#     noNormal[counter]=dict(sorted(temp.items(), key=lambda item: item[0],reverse=True))
# y=[]
# for counter in range(len(noNormal)):
#     temp=noNormal[counter]
#     x=temp.keys()
#     y.append(list(temp.values()))
# y=np.asarray(y)
# x=list(x)

# y=y.mean(axis=0)
# indexSorted=np.argsort(y)
# x=np.asarray(x)
# x=x[indexSorted.astype(int)]
# y=np.sort(y)
# print("********")

# plt.figure(figsize=(15,15))
# plt.bar(x,y)
# plt.xticks(x, rotation='vertical')




