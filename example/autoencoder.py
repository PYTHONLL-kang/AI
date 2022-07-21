from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

#tensorflow에서 라이브러리 불러오기

mnist = tf.keras.datasets.mnist #keras에 mnist 데이터 불러오기
(x_train, y_train), (x_valid, y_valid) = mnist.load_data() #x, y 훈련값, 성능 평가 위한 x, y 값 나눠 저장
x_train.shape, y_train.shape

x_train = x_train.reshape(-1, 28, 28, 1) #훈련할 수 있는 데이터 모형으로 변환. 
x_train = x_train / 127.5 - 1
x_train.min(), x_train.max() #최대값과 최소값 구함

encoder_input = Input(shape=(28, 28, 1)) #훈련시킬 모양을 encoder에 넣어 둠

# 28 X 28
x = Conv2D(32, 3, padding='same')(encoder_input) 
x = BatchNormalization()(x) #배치 정규화
x = LeakyReLU()(x)  #LeakyReLU 활성화 함수 사용. ReLU에 단점을 보완한 함수라 함.

# 28 X 28 -> 14 X 14
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x) 
x = LeakyReLU()(x)

# 14 X 14 -> 7 X 7
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 17 X 7
x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

#차원 축소. 28*28에서 17*7로

x = Flatten()(x) # 배치 사이즈에 영향 없이 2dimension 대신 single dimension으로 입력값을 바꾸는 함수라 함.

# 2D 좌표로 표기하기 위하여 2를 출력값으로 지정.
encoder_output = Dense(2)(x)
encoder = Model(encoder_input, encoder_output)
encoder.summary()

# Input으로는 2D 좌표가 들어감.
decoder_input = Input(shape=(2, ))

# 2D 좌표를 7*7*64 개의 neuron 출력 값을 가지도록 변경.
x = Dense(7*7*64)(decoder_input)
x = Reshape( (7, 7, 64))(x)

# 7 X 7 -> 7 X 7
x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 X 7 -> 14 X 14
x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 14 X 14 -> 28 X 28
x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 28 X 28 -> 28 X 28
x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 최종 output
decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)

decoder = Model(decoder_input, decoder_output)
decoder.summary()

LEARNING_RATE = 0.0005 #학습률 0.0005
BATCH_SIZE = 32 #배치 사이즈 32

encoder_in = Input(shape=(28, 28, 1)) #인코더에 값 넣기
x = encoder(encoder_in)
decoder_out = decoder(x) #디코더에 인코딩한 값 넣기

auto_encoder = Model(encoder_in, decoder_out) #인코딩에 넣은 값과 디코더에서 출력한 값으로 모델 만들기
auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.MeanSquaredError())
#optimizer는 Adam으로, 손실함수는 평균제곱오차

checkpoint_path = 'tmp/01-basic-auto-encoder-MNIST.ckpt' #중간 저장할 위치
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='loss', 
                             verbose=1)

auto_encoder.fit(x_train, x_train, #오토인코더 훈련
                 batch_size=BATCH_SIZE, 
                 epochs=100, 
                 callbacks=[checkpoint], 
                )

auto_encoder.load_weights(checkpoint_path) #저장한 곳에서 가중치 불러오기

import matplotlib.pyplot as plt

%matplotlib inline
# MNIST 이미지에 대하여 x, y 좌표로 뽑음.
xy = encoder.predict(x_train) #인코더 예측값
xy.shape, y_train.shape

plt.figure(figsize=(15, 12)) #plt 사용해서 그림 그리기.
plt.scatter(x=xy[:, 0], y=xy[:, 1], c=y_train, cmap=plt.get_cmap('Paired'), s=3)
plt.colorbar()
plt.show()

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(x_train[i].reshape(28, 28), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Original Images')
plt.show()

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    axes[i//5, i%5].imshow(decoded_images[i].reshape(28, 28), cmap='gray')
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Auto Encoder Images')
plt.show()
