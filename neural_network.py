import random

import tensorflow as tf
import numpy as np


pattern_in_list_X = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
pattern_in_list_0 = [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]
pattern_to_matrix_X = np.array(pattern_in_list_X)
pattern_to_matrix_0 = np.array(pattern_in_list_0)
counter = 0

x_data = []
x_label = []

y_data = []
y_label = []

for i in range(100000):
    x_data.append(np.random.choice([0, 1], size=(16,)))

for i in range(5000):
    x_data.append(pattern_to_matrix_X)
    x_data.append(pattern_to_matrix_0)

random.shuffle(x_data)

for x in x_data:
    for v, p in zip(x, pattern_to_matrix_X):
        if v == p:
            counter += 1

    if counter >= 14:
        x_label.append(1)
    elif counter <= 2:
        x_label.append(0)
    else:
        x_label.append(2)
    counter = 0


for i in range(50000):
    y_data.append(np.random.choice([0, 1], size=(16,)))


for i in range(2500):
    y_data.append(pattern_to_matrix_X)
    y_data.append(pattern_to_matrix_0)


random.shuffle(y_data)

for y in y_data:
    for v, p in zip(y, pattern_to_matrix_X):
        if v == p:
            counter += 1

    if counter >= 14:
        y_label.append(1)
    elif counter <= 2:
        y_label.append(0)
    else:
        y_label.append(2)
    counter = 0


x_data = np.array(x_data, dtype=np.int8)
x_label = np.array(x_label, dtype=np.int8)

y_data = np.array(y_data, dtype=np.int8)
y_label = np.array(y_label, dtype=np.int8)

x_label = tf.keras.utils.to_categorical(x_label)  # Converts a class vector (integers) to binary class matrix.
y_label = tf.keras.utils.to_categorical(y_label)

# 2. inicjalizacja modelu
model = tf.keras.models.Sequential()
# 3. dodanie warstwy wejściowej
model.add(tf.keras.layers.Flatten())
# 4. dodanie warstwy ukrytej z trzema neuronami, funkcja aktywacji relu
model.add(tf.keras.layers.Dense(3, activation=tf.nn.relu))
# 5. dodanie warstwy wyjściowej z trzema neuronami, funkcja aktywacji softmax
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

# 6. kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. trening modelu
model.fit(x_data, x_label, epochs=20, batch_size=256)

# 8. wygenerowanie i zapisanie wag do plików tekstowych
weights = model.get_weights()
#np.savetxt("weights1.txt", weights[0], fmt='%.2f')
#np.savetxt("biases1.txt", weights[1], fmt='%.2f')
#np.savetxt("weights2.txt", weights[2], fmt='%.2f')
#np.savetxt("biases2.txt", weights[3], fmt='%.2f')


# Example predictions
testdata1 = np.array([[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]])
# X
testdata2 = np.array([[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]])
# O
testdata3 = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]])
# incorrect
testdata4 = np.array([[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
prediction = model.predict(testdata2)

predicted_class = np . argmax ( prediction )
class_probabilities = prediction [0]

# Mapowanie etykiet na nazwy klas
labels = ['0', 'X', 'ignor']
predicted_label = labels[ predicted_class ]
print ( f" Obraz przedstawia wzorzec : { predicted_label }")
print (" Prawdopodobienstwa wystapienia :")
for label , probability in zip ( labels , class_probabilities ) :
    print ( f"{ label }: { probability :.5f}")







