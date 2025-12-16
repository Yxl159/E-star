# -*- coding: UTF-8 -*-
import time
import numpy as np
import sys
import os
from pyts.transformation import ShapeletTransform
from sklearn.pipeline import make_pipeline
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.svm import LinearSVC
from sklearn import preprocessing
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
data_name = ['ArrowHead', 'Adiac', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration', 'Coffee', 'Computers',
             'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes',
             'ECG200', 'ECG5000', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', 'Fish', 'GunPoint'
             , 'GunPointOldVersusYoung', 'Ham', 'Herring', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages',
             'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Plane',
             'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'TwoLeadECG', 'UMD', 'Wine', 'WordSynonyms']

X_train, y_train, X_test, y_test = TSC_data_loader('ChlorineConcentration')
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
X_train = np.squeeze(X_train)
X_test = np.squeeze(X_test)

start_time = time.time()
shapelet = ShapeletTransform(window_sizes=np.arange(10, 30, 1), random_state=42)
svc = LinearSVC()
clf = make_pipeline(shapelet, svc)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
end_time = time.time()
transform_time = end_time - start_time
print(f"Shapelet Transform training time: {transform_time:.2f} seconds")
print(f"Test accuracy: {accuracy:.2f}")