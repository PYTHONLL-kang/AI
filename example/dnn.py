#Author: Gentry Atkinson
#Organization: Texas University
#Data: 31 October, 2020
#Train and return a deep neural network

import numpy as np #데이터를 정리하는 라이브러리
import random as rand #무작위로 값을 정하게 해주는 라이브러리
import tensorflow as tf #딥러닝 프로그램 구현을 도와주는 라이브러리
from tensorflow.keras import Sequential #텐서플로우 위에서 동작하는 사용자 친화적인 케라스 라이브러리에서 하위 라이브러리 불러오기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.layers import Input, Conv1DTranspose, Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt #데이터 시각화 라이브러리
from sklearn.metrics import confusion_matrix #파이썬 기반 머신러닝 라이브러리 sklearn에서 혼동행렬 불러오기
from sklearn.model_selection import train_test_split #과적합을 예방하기 위해 train과 test를 분리하는 것
from tensorflow.keras.utils import to_categorical #원-핫 인코딩을 하는 라이브러리


def build_dnn(train_example, num_labels=2): #dnn 구조 만드는 함수
    print("Input shape: ", train_example.shape) #입력 모양으로 train 모양 출력.
    model = Sequential([ #Sequential을 사용
        Input(shape=train_example.shape),
        BatchNormalization(scale=False),
        Dense(128, activation='relu'), #클래스 수 128개, 활성화 함수 relu 사용. 렐루 함수는 선형적임. 0이다가 특정 시점에 1 됨.
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'), #클래스 수 64개, 활성화 함수 relu 사용
        Dense(num_labels, activation='softmax') #클래스 수 num_labels 변수만큼, 활성화 함수로 소프트맥스 사용. 소프트맥수 함수는 로지스틱 회귀에서 사용, 출력 총합 1
                                                #다변수에서 사용
    ])
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc']) #RMSprop 최신 기울기들이 더 많이 반영되도록 기울기를 조정.
    model.summary()#모델 구조 요약해 출력                                                 #categorical_crossentropy 클래스가 여러개인 다중분류 문제 해결 시 사용.
    print("Output shape: ", model.output.shape) #출력 형태 출력
    return model #모델값 반환

def train_dnn(dnn, X, y, withEvaluation=False): #훈련 모델
    if withEvaluation: #초기값이 False인 변수가 True가 되었을 때 동작. 원할 때만 동작시키기 위한 코드인 듯
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=23) #실제 값과 train 데이터를 랜덤하게 변수에 저장

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4) #과적합 방지를 위한 조기 종료 정의. 4번 이상 성능 개선이 안 되면 멈추기
    dnn.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10) #모델 훈련. 반복 100번, 배치 사이즈 10

    if withEvaluation:
        y_pred = np.argmax(dnn.predict(X_test), axis=-1) #예측된 Y값
        y_test = np.argmax(y_test, axis=-1) #실제 Y값
        mat = confusion_matrix(y_test, y_pred) #혼동행렬
        print("predicted labels ", y_pred) #예측한 Y값 출력
        print("true labels ", y_test) #실제 Y값 출력
        print("Confuxion matrix: ") #혼동행렬 출력
        print(mat)

    return dnn

def get_trained_dnn(X, y):
    numLabels = y.shape[1] #lables는 y.shape[1]
    dnn = build_dnn(X[0], num_labels=numLabels) #위에 있던 dnn 구조 만드는 함수로 train set으로 X에 0번째 인수 사용
    dnn = train_dnn(dnn, X, y, withEvaluation=True) #만들어진 dnn 구조로 훈련 시킴.
    return dnn

if __name__ == "__main__": #모듈로 불러오기 된 것이 아닐 때 동작
    print("Test model building")
    X = np.genfromtxt('data/synthetic_test_data.csv', delimiter=',') #데이터 읽어 오기
    y = np.genfromtxt('data/synthetic_test_labels.csv', delimiter=',')
    print("load data")
    y = to_categorical(y) #손실함수인 Categorical cross entropy 적용을 위해 one-hot encoding 필요. 읽어온 Y값 one-hot encoding.
    print("data shape: ", X.shape)
    print("label shape: ", y.shape)
    norm = np.max(X) #X 데이터의 최대값 구함
    X = X/norm
    print("labels", y)

    dnn = build_dnn(X[0], num_labels=3) #만든 dnn 구조 만드는 함수 사용
    dnn = train_dnn(dnn, X, y, withEvaluation=True) #뉴럴 넷 훈련시키는 함수 사용
    predict = dnn.predict(X)
    print("Output feature set: ", predict.shape)
