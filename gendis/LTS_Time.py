# -*- coding: UTF-8 -*-
import numpy as np
from genetic import GeneticExtractor
from sklearn.linear_model import LogisticRegression
np.random.seed(1337)  # Random seed for reproducibility
import sys
import os
sys.path.append(os.getcwd()[:-5])
import numpy as np
import random
from sklearn import preprocessing
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.stats import f_oneway
from sklearn.metrics import accuracy_score
import time

def TSC_data_loader(dataset_name):
    Train_dataset = np.loadtxt(
        os.getcwd()[:] + '/datasets/UCRArchive_2018/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        os.getcwd()[:] + '/datasets/UCRArchive_2018/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float64)
    Test_dataset = Test_dataset.astype(np.float64)

    # 划分数据集，第一列标签为y，后续数据为x
    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]

    # 将标签改为0和1
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))  # 删除维度，方便后续画图
    y_test = le.transform(np.squeeze(y_test, axis=1))

    return X_train, y_train, X_test, y_test
#
# data_name = [
#     "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
#     "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
#     "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate",
#     "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
#     "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf",
#     "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices",
#     "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
#     "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
#     "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga", "ACSF1", "AllGestureWiimoteX", "AllGestureWiimoteY",
#     "AllGestureWiimoteZ", "BME", "Chinatown", "Crop", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel",
#     "FreezerRegularTrain", "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPointAgeSpan",
#     "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "HouseTwenty", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "MelbournePedestrian", "MixedShapesRegularTrain",
#     "MixedShapesSmallTrain", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "PowerCons", "Rock", "SemgHandGenderCh2", "SemgHandMovementCh2",
#     "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "SmoothSubspace", "UMD" ]
data_name = [
    "Adiac", "CinCECGTorso", "Computers", "CricketX", "CricketY", "CricketZ", "ElectricDevices", "FordA", "FordB", "Ham",
    "Haptics", "InsectWingbeatSound", "LargeKitchenAppliances", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "RefrigerationDevices", "ScreenType", "ShapesAll", "SmallKitchenAppliances",
    "StarLightCurves", "Strawberry", "SwedishLeaf", "SyntheticControl", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]



# 用于保存结果的文件，以追加模式打开，你可以根据实际需求调整保存路径和文件名等
result_file = open('experiment_results1.txt', 'a')

for dataset_name in data_name:
    try:
        # 获取当前数据集的数据
        X_train, y_train, X_test, y_test = TSC_data_loader(dataset_name)
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)

        random.seed(2024)
        start_time = time.time()
        genetic_extractor = GeneticExtractor(verbose=True, population_size=16, iterations=100, plot=None)
        genetic_extractor.fit(X_train, y_train)
        distances_train = genetic_extractor.transform(X_train)
        distances_test = genetic_extractor.transform(X_test)

        window_size = 6

        fft_result_train = []
        significant_freqs_train = []
        for j in range(X_train.shape[0]):
            row = X_train[j, :]
            max_f_statistic = 0
            most_significant_spectrum = None
            significant_freq = None
            for i in range(0, len(row) - window_size + 1):
                window = row[i:i + window_size]
                spectrum = np.fft.fft(window)
                _, p_value = f_oneway(spectrum, np.zeros_like(spectrum))

                if p_value < 0.05 and p_value > max_f_statistic:
                    max_f_statistic = p_value
                    most_significant_spectrum = spectrum
                    significant_freq = most_significant_spectrum
            significant_freqs_train.append(significant_freq)
        fft_result_train = np.array(significant_freqs_train)
        review_train = np.fft.ifft(fft_result_train).real

        fft_result_test = []
        significant_freqs_test = []
        for j in range(X_test.shape[0]):
            row = X_test[j, :]
            max_f_statistic = 0
            most_significant_spectrum = None
            significant_freq = None
            for i in range(0, len(row) - window_size + 1):
                window = row[i:i + window_size]
                spectrum = np.fft.fft(window)
                _, p_value = f_oneway(spectrum, np.zeros_like(spectrum))

                if p_value < 0.05 and p_value > max_f_statistic:
                    max_f_statistic = p_value
                    most_significant_spectrum = spectrum
                    significant_freq = most_significant_spectrum
            significant_freqs_test.append(significant_freq)
        fft_result_test = np.array(significant_freqs_test)
        none_indices = [index for index, arr in enumerate(fft_result_test) if arr is None]

        review_test = np.fft.ifft(fft_result_test).real

        last_train = np.concatenate((review_train, distances_train), axis=1)
        last_test = np.concatenate((review_test, distances_test), axis=1)

        lr1 = LogisticRegression(random_state=2024)
        lr1.fit(last_train, y_train)
        end_time = time.time()
        transform_time = end_time - start_time

        accuracy = accuracy_score(y_test, lr1.predict(last_test))

        # 将当前数据集的结果写入文件，格式可以根据需求调整，这里简单示例为：数据集名,准确率,运行时间
        result_file.write(f"{dataset_name},{accuracy:.4f},{transform_time:.2f}\n")
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {e}")
        continue

result_file.close()