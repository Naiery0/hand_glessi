import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 추가

# 내가 만든 데이터 로드
x_custom = []
y_custom = []

for i in range(20):
    img = Image.open('hand_glessi/upload/image/' + str(i) + '.png').convert('L')
    img = img.resize((28,28))
    x = np.asarray(img)
    x = x / 255.0
    x_custom.append(x)
    y_custom.append(i % 10)  # 이미지가 10개씩 2개씩 묶여있으므로 10으로 나눈 나머지가 라벨 값이 됩니다.

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 내가 만든 데이터셋 추가
x_train = np.concatenate((x_train, x_custom))
y_train = np.concatenate((y_train, y_custom))

# 데이터셋 shuffle
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]

# 모델 정의
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 학습 진행
hist = model.fit(x_train, y_train, epochs=20, batch_size=1000, validation_data=(x_test, y_test))

# 내 손글씨 로드 
# pip install opencv-python
import glob
from PIL import Image
import numpy as np 
import cv2
sample = [] 

groups_folder_path = 'hand_glessi/upload/sample' 
    
image_dir = groups_folder_path + '/'
files = glob.glob(image_dir + '/*.png')
    

for i in files:
   
    img = Image.open(i).convert('L')
    img = img.resize((28,28))
    train = np.asarray(img)
    a, train = cv2.threshold(train,100,255,cv2.THRESH_BINARY) 
    sample.append(train)    
plt.imshow(sample[9])
plt.imshow(x_train[5])
sample = np.array(sample)
sample.reshape(10,28,28,1)
import cv2
sample = cv2.bitwise_not(sample)
sample
predictions = model.predict(sample)
np.argmax(predictions[8])
def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{},({})".format(predicted_label,100*np.max(predictions_array))) 
                                    
#test셋과 예측값 / 원래 라벨 비교 ( i에 보고싶은 test셋의 인덱스를 적으세요)


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, sample)
#plt.subplot(1,2,2)    
#plot_value_array(i, predictions,  Y_test)
plt.show()
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(predictions[i]))
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
import matplotlib.pyplot as plt
from matplotlib.image import imread
n = 4
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

print('내가 본 글자는 ', model.predict(x_test[n].reshape((1, 28, 28))))
import random

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=0)

wrong_result = []


#test_labels
predicted_labels[0]
model.save('myhand_CNN_model.h5')