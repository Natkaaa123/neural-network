
import numpy as np



def recognize(v):
    pattern_in_list_X = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    pattern_to_matrix_X = np.array(pattern_in_list_X)
    counter = 0
    for x, p in zip(v, pattern_to_matrix_X):
        if x == p:
            counter += 1

    if counter >= 14:
        return "X"
    elif counter <= 2:
        return "0"
    else:
        return "Nie rozpoznano"

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_values = np.exp(x - np.max(x))  # subtracting the maximum value for numerical stability
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

if __name__ == "__main__":
    weights1 = np.loadtxt("weights1.txt")
    weights2 = np.loadtxt("weights2.txt")
    biases1 = np.loadtxt("biases1.txt")
    biases2 = np.loadtxt("biases2.txt")


    vector0 = [0,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0]
    vectorX = [1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,1]

    test_X = np.array([[int(x) for x in vectorX]])
    hidden1_output = relu(np.dot(test_X, weights1) + biases1)

    print("res: ", hidden1_output)
    output = softmax(np.dot(hidden1_output, weights2) + biases2)
    check_out = np.dot(hidden1_output, weights2) + biases2
    print("res_soft = ", check_out)
    print("res_soft z funkcją softmax: ", output)
    result = recognize(vectorX)
    if max(output[0][0], output[0][2])<output[0][1]:
        nn_result = 'X'
    elif output[0][0]>output[0][2]:
        nn_result = '0'
    else:
        nn_result = 'Nie rozpoznano'


    print(f"Wprowadzony wektor w postaci ramki 4x4: " + "\n"
        f"{test_X.reshape(4, 4)}")

    print("Wynik funkcji recognize(): ", result)
    print("Wynik sieci neuronowej: ", nn_result)

