import tkinter as tk
import cv2
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
model = keras.Sequential()

model.add(Flatten(input_shape=(28, 28, 1)))

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2)


def event_function(event):
    x = event.x
    y = event.y

    x1 = x - 10
    y1 = y - 10

    x2 = x + 10
    y2 = y + 10

    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='white')


def save():
    global count

    img_array = np.array(img)
    img_array = cv2.resize(img_array, (28, 28))

    cv2.imwrite(str(count) + '.jpg', img_array)
    count = count + 1


def clear():
    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (500, 500), (0, 0, 0))
    img_draw = ImageDraw.Draw(img)

    label_status.config(text='PREDICTED DIGIT: NONE')


def predict():
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28))

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    result = model.predict(img_array)
    label = np.argmax(result, axis=1)

    label_status.config(text='PREDICTED DIGIT:' + str(label))


count = 0

win = tk.Tk()

canvas = tk.Canvas(win, width=500, height=500, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='white', fg='black', font='Times 20 bold', command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='white', fg='black', font='Times 20 bold', command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='white', fg='black', font='Times 20 bold', command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='white', fg='black', font='Times 20 bold', command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='PREDICTED DIGIT: NONE', bg='white', font='Times 20 bold')
label_status.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', event_function)
img = Image.new('RGB', (500, 500), (0, 0, 0))
img_draw = ImageDraw.Draw(img)

win.mainloop()
