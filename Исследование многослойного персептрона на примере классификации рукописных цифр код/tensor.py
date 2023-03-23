import random
import math

def scal_mul(a, b, c=1):
    if c == 1:  # Скалярное произведение матрицы на миссив
        output = []
        out1 = 0
        for i in range(len(b[0])):
            for j in range(len(a)):
                out1 += b[j][i] * a[j]
            output.append(out1)
            out1 = 0
        return output

    elif c == 2:
        output = 0  # Скалярное произведение двух массивов
        for i in range(len(a)):
            output += a[i] * b[i]
        return output

    elif c == 3:  # Прроизведение двух массивов
        new_mass = []
        for i in range(len(a)):
            new_mass.append(a[i] * b[i])
        return new_mass

    elif c == 4:  # Произведение matrix на число
        new_matrix = a.copy()
        for i in range(len(a)):
            for j in range(len(a[i])):
                new_matrix[i][j] = a[i][j] * b

        return new_matrix

    elif c == 5: # Произведение двух массивов
        new_matrix = [([0] * len(b)).copy() for i in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b)):
                new_matrix[i][j] = a[i] * b[j]
        return new_matrix

    elif c == 6: # Произведение массива на число
      for i in range(len(a)):
        a[i] *= b
      return a


def matrix_sum(a, b):  # Сумма двух массивов
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j] += b[i][j]

def relu(x): # Функция активации релу
    new_mass = []
    for i in range(len(x)):
        if x[i] < 0:
            new_mass.append(0)
        else:
            new_mass.append(x[i])
    return new_mass


def mass_max(x):  # Нахождение максимального элемента массива
    max = -1000
    for i in range(len(x)):
        if x[i] > max:
            max = x[i]
    return max


def mass_argmax(x): # Нахождение индекса максимального элемента
    max = -1000
    argmax = 0
    for i in range(len(x)):
        if x[i] > max:
            max = x[i]
            argmax = i
    return argmax



def matrix3_to_2(x):  # Превращение трехмерного массива в двумерный
    new_matrix = []
    new_mass = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                new_mass.append(x[i][j][k] / 255)
        new_matrix.append(new_mass)
        new_mass = []
    return new_matrix

def matrix2_to_1(x): # Превращение двумерного массива в список
    new_matrix = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            new_matrix.append(x[i][j])
    return new_matrix


def error(x, y): #Разность массивов
    new_mass = []
    for i in range(len(x)):
        new_mass.append(x[i] - y[i])
    return new_mass


def reverse(x): # Разворот матрицы
    new_matrix = []
    new_mass = []
    for i in range(len(x[0])):
        for j in range(len(x)):
            new_mass.append(x[j][i])
        new_matrix.append(new_mass)
        new_mass = []
    return new_matrix


def relu2(x): # Функция обратная релу
    new_mass = []
    for i in range(len(x)):
        if x[i] < 0:
            new_mass.append(0)
        elif x[i] >= 0:
            new_mass.append(1)
    return new_mass


def spawn_weights(x, y, bias = False, z = 0.2):  # Генерация весов
    if not bias:
        new_matrics = [([0] * y).copy() for i in range(x)]
        for i in range(x):
            for j in range(y):
                new_matrics[i][j] = random.random() * z - 0.1
        return new_matrics
    if bias:
        new_matrics = [([0] * y).copy() for i in range(x + 1)]
        for i in range(x):
            for j in range(y):
                new_matrics[i][j] = random.random() * z - 0.1
        return new_matrics

def add_bias_to_layer(x):
    new_mass = x.copy()
    new_mass.append(int(1))
    return new_mass

def answser_to_mass(x):
    new_matrix = [([0] * 10).copy() for i in range(len(x))]
    for i in range(len(x)):
        new_matrix[i][x[i]] = 1
    return new_matrix

def create_zero_matrix(x,y):
    new_matrics = [([0] * y).copy() for i in range(x)]
    return new_matrics

def mass_sum(x):
  suma = 0
  for i in range(len(x)):
    suma += x[i]
  return suma

def dropout(a, b = 2, bias = False):
    if not bias:
        new_mass = [1] * len(a)
        for i in range(len(a)):
            if random.randint(0, b - 1) == 0:
                new_mass[i] = 0
        return new_mass
    if bias:
        new_mass = [1] * len(a)
        for i in range(len(a)):
            if random.randint(0, b - 1) == 0:
                new_mass[i] = 0
        new_mass[len(a) - 1] = 1
        return new_mass

def change_matrix(x):
    for i in range(len(x)):
        mark = False
        mark_2 = True
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                if x[i][j][k] != 0:
                    mark_2 = False
            if not mark_2:
                break
            else:
                k_end = j
                mark = True
        if mark:
            b = shiiiii(x[i][k_end + 1:].copy(), x[i][0: k_end + 1].copy())
            x[i] = b.copy()

    return x

def shiiiii(x,y):
    new_matrix = []
    for i in range(len(x)):
        new_matrix.append(x[i])
    for i in range(len(y)):
        new_matrix.append(y[i])
    return new_matrix

def duble_reverse(x):
    new_matrix = [([0] * len(x[0])).copy() for i in range(len(x))]
    for i in range(len(x[0])):
        for j in range(len(x)):
            new_matrix[i][j] = x[len(x) - i - 1][len(x[0]) - j - 1]
    return new_matrix


def duble_change_matrix(x):
    for i in range(len(x)):
        x[i] = duble_reverse(x[i])
    x = change_matrix(x)
    for i in range(len(x)):
        x[i] = duble_reverse(x[i])

    return x

def softmax(x):
    new_mass = []
    suma = 0
    for i in range(len(x)):
        suma += math.exp(x[i])
    for i in range(len(x)):
        new_mass.append(math.exp(x[i]) / suma)
    return new_mass

def div(x, y):
    new_mass = []
    for i in range(len(x)):
        new_mass.append(x[i] / y)
    return new_mass