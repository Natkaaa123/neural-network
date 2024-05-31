import random
import numpy as np


def recognize_pattern_O(array):
    poprawny_O = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

    blad0 = False
    blad1 = False
    for i in range(16):
        if poprawny_O[i] != array[i]:
            if array[i] == 1:
                if blad1 == False:
                    blad1 = True
                else:
                    return False
            elif array[i] == 0:
                if blad0 == False:
                    blad0 = True
                else:
                    return False
    return True


def recognize_pattern_X(array):
    poprawny_X = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]

    blad0 = False
    blad1 = False
    for i in range(16):
        if poprawny_X[i] != array[i]:
            if array[i] == 1:
                if blad1 == False:
                    blad0 = True
                else:
                    return False
            elif array[i] == 0:
                if blad0 == False:
                    blad0 = True
                else:
                    return False
    return True


if __name__ == "__main__":

    oj = []

    for i in range(5000):

        wejscie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        while (wejscie.count(1) < random.randint(7, 9)):
            wejscie[random.randint(0, 15)] = 1

        poprawny_O = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

        poprawny_X = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]

        if recognize_pattern_O(wejscie):
            wejscie.append(0)
            if wejscie not in oj:
                oj.append(wejscie)
            ##continue

        elif recognize_pattern_X(wejscie):
            wejscie.append(1)
            if wejscie not in oj:
                oj.append(wejscie)
            ##continue

        else:
            wejscie.append(2)
            if wejscie not in oj:
                oj.append(wejscie)

    print("koniec")

    with open('data.txt', 'w') as file:
        for item in oj:
            for el in item:
                file.write(str(el) + " ")
            file.write("\n")




    ##print(oj)