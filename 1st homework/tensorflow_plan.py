import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
#excel로 파일 리딩을 해서 7~8개의 변수를 선택해 값을 매개변수로 두고, 종속변수는 과밀화 변화율, 
# 또는 log(과밀화 변화율) 또는 0<x<1형태로 존재하게 만들 예정이다.
#총 280개 정도의 row가 나오는데 data가 조금 부족하지 않을까 하는 생각으로 고민이 조금 있는 실정이다.
#자료가 많지 않고 통계가 많지 않아 이러한 식으로 원인을 구할 수 있는 모든 통계로 잡고 원인을 찾는 과정의 논문이 거의 찾아보기 힘들다.
print(x)
print('-------x reshape-----------')
x = x.reshape((x.shape[0], x.shape[1],1))
#x reshape 과정 : 이거 왜 하는 지 모르겠다.(물론 excel로 reading 할 경우 필요할 것 같다.)
print('x shape : ',x.shape)
print(x)
# 2. 모델 구성
model = Sequential()
model.add(keras.layers.LSTM(10, activation = 'relu', input_shape=(3,1)))
# 단변량 시계열 분석에는 lstm이 가장 효과가 좋았고, dnn도 비슷한 성능을 보였다. lstm의 특성상 자료가 많을 수록 
# 정확도가 높아지는데(상대적으로 다른 학습방법보다 더) 그래서 dnn을 써야 할지 아니면 그냥 머신러닝처럼 노드를 한개로 구성할 지 고민이다.
model.add(keras.layers.Dense(5))
model.add(keras.layers.Dense(1))
# 3. 실행
# optimizer의 경우 node한개로 test삼아 돌려보던중, sgd가 발산하는 형태가 자꾸 나오게 되어 조정을 해보다 adam으로 선택했다. 
# 매개변수의 10의 자리 단위가 들쯕날쭉 한 만큼 조금의 수정과정은 필요 할 것 같다.
# loss의 경우 mae와 mse가 가장 효과가 있는 편이였다.
model.compile(optimizer='adam', loss='mse')
# 예제이기 떄문에 x의 개수와 y의 개수, epochs가 적은 형태이나, 형태가 발전되면 조금 더 큰 epochs가 들어 와야 할 것이다.
model.fit(x, y, epochs=10000, batch_size=1)
x_input = np.array([[6,7,8]])
x_input = x_input.reshape((1,3,1))
yhat = model.predict(x_input)
#predict의 경우 lstm으로 사용 할 경우 같은 데이터를 넣었을 때와 많은 차이를 보였다.
# 그리고 lstm은 선형적인 모습이 보이기 때문에 lstm이 조금 activation linear의 dnn과 비슷한 성능을 보였다.
print(yhat)


