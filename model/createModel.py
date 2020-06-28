from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_model(n_rows, n_cols, uniques):
    m = Sequential()
    m.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(n_rows, n_cols, 1)))

    m.add(Conv2D(64, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(uniques, activation='softmax'))

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m
