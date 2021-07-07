from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

iris = load_iris()  # sample data load

data_X = iris.data
data_y = iris.target
(X_train, X_test, y_train, y_test) = train_test_split(
    data_X, data_y, train_size=0.8, random_state=1)

y_train = to_categorical(y_train)  # 0 1 2 를 원-핫 백터로
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))  # 3은 출력값이 3
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=1,
                    epochs=200, validation_data=(X_test, y_test))
