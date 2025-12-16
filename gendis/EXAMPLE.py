# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from genetic import GeneticExtractor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(1337)  # Random seed for reproducibility
import sys
import os
sys.path.append(os.getcwd()[:-5])
import numpy as np
import tensorflow as tf
import random
import sklearn
from sklearn import preprocessing
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.stats import f_oneway
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

# data_name = ['ArrowHead', 'Adiac', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration', 'Coffee', 'Computers',
#              'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes',
#              'ECG200', 'ECG5000', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', 'Fish', 'GunPoint'
#              , 'GunPointOldVersusYoung', 'Ham', 'Herring', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages',
#              'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Plane',
#              'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'TwoLeadECG', 'UMD', 'Wine', 'WordSynonyms']

data_name = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
    "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
    "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate",
    "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf",
    "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices",
    "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
    "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga", "ACSF1", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "BME", "Chinatown", "Crop", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel",
    "FreezerRegularTrain", "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "HouseTwenty", "InsectEPGRegularTrain", "InsectEPGSmallTrain", "MelbournePedestrian", "MixedShapesRegularTrain",
    "MixedShapesSmallTrain", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID", "PowerCons", "Rock", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "SmoothSubspace", "UMD" ]
#
# # 类型为numpy.ndarray，形状为(100, 96)
# #
# # GENDIS 和 E-STAR 的准确率
# gendis_accuracy = [82.0, 58.8, 87.5, 90.5, 83.0, 97.6, 57.5, 98.6, 96.5, 83.2, 81.5, 76.0, 73.7, 93.8, 100.0, 92.6, 94.1,
#                    89.0, 95.7, 61.8, 96.0, 79.1, 76.3, 63.1, 86.6, 78.7, 99.3, 95.6, 87.0, 90.7, 90.6, 86.4]
# estar_accuracy = [77.7, 60.0, 90.0, 90.0, 83.3, 99.1, 56.7, 100.0, 94.4, 77.0, 76.1, 68.3, 74.1, 93.5, 100.0, 97.4, 96.6,
#                   88.3, 98.7, 67.2, 95.8, 72.1, 79.5, 75.9, 91.5, 80.8, 100.0, 96.3, 90.8, 92.6, 91.6, 87.1]
#
# differences = np.array(estar_accuracy) - np.array(gendis_accuracy)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# bar_colors = ["green" if diff > 0 else "red" for diff in differences]
# ax.bar(data_name, differences, color=bar_colors)
# # Add zero line to the plot
# plt.axhline(0, color='black', linewidth=1)
#
# # Set labels and title
# plt.xlabel('Datasets')
# plt.ylabel('Difference in Accuracy (E-STAR - GENDIS)')
# plt.title('Difference in Accuracy between E-STAR and GENDIS')
# plt.xticks(rotation=90)
#
# # Display the plot
# plt.tight_layout()
# path = os.getcwd()[:] + 'Difference Bar.png'
# plt.savefig(path)
# plt.show()



X_train, y_train, X_test, y_test = TSC_data_loader('Computers')
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
X_train = np.squeeze(X_train)
X_test = np.squeeze(X_test)

# # Visualize the timeseries in the train and test set
# colors = ['r', 'b', 'g', 'y', 'c']
#
# plt.figure(figsize=(10, 5))
# for ts, label in zip(X_train, y_train):
#     plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
# plt.title('The timeseries in the train set')
# plt.show()
#
# plt.figure(figsize=(10, 5))
# for ts, label in zip(X_test, y_test):
#     plt.plot(range(len(ts)), ts, c=colors[label%len(colors)])
# plt.title('The timeseries in the test set')
# plt.show()
#
random.seed(2024)
start_time = time.time()
genetic_extractor = GeneticExtractor(verbose=True, population_size=24, iterations=100, plot=None)
# genetic_extractor = GeneticExtractor(verbose=True, population_size=25, iterations=200, plot=None)
genetic_extractor.fit(X_train, y_train)
distances_train = genetic_extractor.transform(X_train)
distances_test = genetic_extractor.transform(X_test)
# history3 = np.array([d[1]['avg'] for d in genetic_extractor.history])
# np.save('history1_crossover1+2+3_Herring.npy', history3)
# history1 = np.load('history1_crossover1_Herring.npy')
# history2 = np.load('history1_crossover1+2_Herring.npy')
# # # history3 = np.load('history3_max_ItalyPowerDemand.npy')
# # # history4 = np.load('history4_max_ItalyPowerDemand.npy')
# # history5 = np.load('history5_max_Herring.npy')
# plt.figure(figsize=(10, 6))
# plt.plot(X_train[2], label='ECG5000')
# plt.plot([], [], color='r', label='Shapelet')
#
# # plt.plot(list(range(len(history1))), history1, color='b', label='Crossover operator 1')
# # plt.plot(list(range(len(history2))), history2, color='r', label='Crossover operator 1+2')
# # # plt.plot(list(range(len(history3))), history3, color='y', label='Crossover operator 1+2+3')
# # plt.plot(list(range(len(history3))), history3, color='g', label='Crossover operator 1+2+3')
# # # plt.plot(list(range(len(history5))), history5, color='r', label='E-STAR Crossover operator')
# #
# # # 添加标题和标签
# plt.title('Explainable Shapelet features', fontsize=17)
#
# plt.xlabel('Time', fontsize=17)
# plt.ylabel('Value', fontsize=17)
# # 显示图例
# plt.legend(fontsize=17)
# # 显示图表
# plt.xticks(rotation=45)  # 旋转 x 轴标签，使其更易读
# plt.tight_layout()  # 调整布局，防止标签重叠
# path = os.getcwd()[:] + 'ECG5000_shapelet.png'
# plt.savefig(path)
# plt.show()
# plt.figure(figsize=(5, 3))
# plt.plot(distances_train[2], color='r')
# path = os.getcwd()[:] + 'shapelet.png'
# plt.savefig(path)
# plt.show()
# batch_size = 12
window_size = 16

# Finding the number of windows
# num_windows = int((X_train.shape[1] - (batch_size * window_size)) / batch_size)
#
# # Dynamic adjustment of the window size
# while num_windows < 0:
#     window_size -= 1
#     if window_size <= 0:
#         window_size = 1
#     num_windows = int((X_train.shape[1] - (batch_size * window_size)) / batch_size)
#     if window_size == 1 and num_windows < 0:
#         raise ValueError("Batch_size (%s) larger than sequence length (len=%s). Adjust it." % (
#             batch_size, X_train.shape[1]))



# def calculate_dtw(seq1, seq2):
#     distance, _ = fastdtw(seq1, seq2, dist=euclidean)
#     return distance
#
# def calculate_fft(window):
#     return np.fft.fft(window)
    # return np.abs(fft(window))


# def sliding_dtw_for_row(row, window_size):
#     num_cols = len(row)
#     max_dtw = float('inf')
#     max_dtw_window = None
#     max_dtw_start = None  # To store the starting index of max DTW window
#
#     for i in range(num_cols - window_size + 1):
#         window = row[i:i + window_size]
#         dtw = calculate_dtw(window, row)
#
#         if dtw < max_dtw:
#             max_dtw = dtw
#             max_dtw_window = window
#             max_dtw_start = i
#
#     return max_dtw_window, max_dtw_start
# plt.plot(X_train[1])
# plt.legend()
# plt.show()

# def hampel_filter(data, window_size=3, n_sigma=3):
#     """
#     Apply Hampel filter to the input data.
#
#     Parameters:
#         - data: Input time series data (1D array).
#         - window_size: Size of the window for the filter.
#         - n_sigma: Number of standard deviations for outlier detection.
#
#     Returns:
#         - filtered_data: Data after Hampel filtering.
#         - outliers: Boolean array indicating the position of outliers.
#     """
#     # Apply median filter to the data
#     median_filtered = medfilt(data, kernel_size=window_size)
#
#     # Calculate the median absolute deviation (MAD)
#     mad = np.median(np.abs(data - median_filtered))
#
#     # Detect outliers
#     outliers = np.abs(data - median_filtered) > n_sigma * mad
#
#     # Replace outliers with the median value
#     filtered_data = np.where(outliers, median_filtered, data)
#
#     return filtered_data, outliers
# filtered_ecg_data_train = np.zeros_like(X_train)
# outliers_indices_train = []
# for i in range(X_train.shape[0]):
#     filtered_row, outliers = hampel_filter(X_train[i, :])
#     filtered_ecg_data_train[i, :] = filtered_row
#     outliers_indices_train.append(np.where(outliers)[0])
#
# filtered_ecg_data_test = np.zeros_like(X_test)
# outliers_indices_test = []
# for i in range(X_test.shape[0]):
#     filtered_row, outliers = hampel_filter(X_test[i, :])
#     filtered_ecg_data_test[i, :] = filtered_row
#     outliers_indices_test.append(np.where(outliers)[0])
# plt.plot(filtered_ecg_data[1])
# plt.plot(X_train[1])
# plt.legend()
# plt.show()

fft_result_train = []
# dtw_result_train = []
significant_freqs_train = []
for j in range(X_train.shape[0]):
    row = X_train[j, :]
    max_f_statistic = 0
    # max_dtw_window, max_dtw_start = sliding_dtw_for_row(row, window_size)
    # print(f"For row {i + 1}, max DTW window:", max_dtw_window)
    # plt.plot(row, label='Original Sequence')
    # max_dtw_end = max_dtw_start + window_size
    # plt.plot(range(max_dtw_start, max_dtw_end), max_dtw_window, label='Max DTW Window Sequence', linestyle='--')
    #
    # # Mark the start and end positions of the max DTW window
    # plt.axvline(x=max_dtw_start, color='r', linestyle='--', label='Max DTW Window Start')
    # plt.axvline(x=max_dtw_end, color='g', linestyle='--', label='Max DTW Window End')
    #
    # plt.title(f'Row {i + 1} - Original vs Max DTW Window')
    # plt.legend()
    # plt.show()
    # dtw_result_train.append(max_dtw_window)
    most_significant_spectrum = None
    significant_freq = None
    for i in range(0, len(row) - window_size + 1):
        window = row[i:i+window_size]
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
# dtw_result_test = []
significant_freqs_test = []
for j in range(X_test.shape[0]):
    row = X_test[j, :]
    max_f_statistic = 0
    # max_dtw_window, max_dtw_start = sliding_dtw_for_row(row, window_size)
    # dtw_result_test.append(max_dtw_window)
    most_significant_spectrum = None
    significant_freq = None
    for i in range(0, len(row) - window_size + 1):
        window = row[i:i+window_size]
        spectrum = np.fft.fft(window)
        _, p_value = f_oneway(spectrum, np.zeros_like(spectrum))

        if p_value < 0.05 and p_value > max_f_statistic:
            max_f_statistic = p_value
            most_significant_spectrum = spectrum
            significant_freq = most_significant_spectrum
    significant_freqs_test.append(significant_freq)
fft_result_test = np.array(significant_freqs_test)
none_indices = [index for index, arr in enumerate(fft_result_test) if arr is None]

# 输出 None 的位置
print("None 的位置:", none_indices)

review_test = np.fft.ifft(fft_result_test).real

# plt.figure(figsize=(10, 6))
# plt.plot(review_train[2], color='r')
# plt.show()


last_train = np.concatenate((review_train, distances_train),axis=1)
last_test = np.concatenate((review_test, distances_test), axis=1)
# clf1 = svm.SVC(C=10, kernel='rbf', gamma='auto', decision_function_shape='ovo')
# clf1.fit(last_train, y_train)
# predictions1 = clf1.predict(last_test)
# scores1 = accuracy_score(predictions1, y_test)
# print('Shapelets特征分类正确率：', scores1)


# lr = LogisticRegression(random_state=2020)
lr1 = LogisticRegression(random_state=2024)
# lr.fit(distances_train, y_train)
lr1.fit(last_train, y_train)
end_time = time.time()
transform_time = end_time - start_time
print(f"Shapelet Transform training time: {transform_time:.2f} seconds")
# Print the accuracy score on the test set
# print('Accuracy LR = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))
print('Accuracy LR1 = {}'.format(accuracy_score(y_test, lr1.predict(last_test))))