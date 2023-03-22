import math

import numpy as np
from PIL import Image, ImageOps
from random import random, randrange, randint


def objParse_v(filename):
    file = open(filename, 'r')
    v_arr = []
    for line in file:

        line = line.split()

        try:
            if line[0] == 'v':
                tmp = [float(line[1]), float(line[2]), float(line[3])]
                v_arr.append(tmp)

        except:
            None

    return v_arr


def objParse_f(filename):
    file = open(filename, 'r')
    f_arr = []
    for line in file:

        line = line.split()

        try:
            if line[0] == 'f':
                tmp = [float(line[1].split("/")[0]), float(line[2].split("/")[0]), float(line[3].split("/")[0])]
                f_arr.append(tmp)

        except:
            None
    return f_arr


def line1(x0, y0, x1, y1, img):
    t = 0.0
    while t < 1.0:
        x = x0 * (1. - t) + x1 * t
        y = y0 * (1. - t) + y1 * t
        img.putpixel((int(x), int(y)), 255)
        t += 0.01


def line2(x0, y0, x1, y1, img):
    for i in range(int(x0), int(x1)):
        t = (i - x0) / (float)(x1 - x0)
        y = y0 * (1. - t) + y1 * t
        img.putpixel((round(i), round(y)), 255)


def line3(x0, y0, x1, y1, img):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for i in range(int(x0), int(x1)):
        t = (i - x0) / (float)(x1 - x0)
        y = y0 * (1. - t) + y1 * t
        if (steep):
            img.putpixel((int(y), int(i)), 255)
        else:
            img.putpixel((int(i), int(y)), 255)


def line4(x0, y0, x1, y1, img):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    derror = abs(dy / dx)

    error = 0
    y = y0

    for i in range(int(x0), int(x1)):
        t = (i - x0) / (float)(x1 - x0)
        y = y0 * (1. - t) + y1 * t
        if (steep):
            img.putpixel((int(y), int(i)), 255)
        else:
            img.putpixel((int(i), int(y)), 255)

        error += derror
        if error > 0.5:
            if y1 > y0:
                y += 1
            else:
                y -= 1

            error -= 1


def baicenterCords(x0, x1, x2, y0, y1, y2, x, y):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    if np.isclose(lambda0 + lambda1 + lambda2, 1.0):
        t = [lambda0, lambda1, lambda2]
        return t
    else:
        return 0


def bariSolve(v, f):
    bariCords = []
    for i in range(len(f)):
        x0 = v[int(f[i][0]) - 1][0]
        x1 = v[int(f[i][1]) - 1][0]
        x2 = v[int(f[i][2]) - 1][0]
        y0 = v[int(f[i][0]) - 1][1]
        y1 = v[int(f[i][1]) - 1][1]
        y2 = v[int(f[i][2]) - 1][1]
        x = 1
        y = 1
        bariCords.append(baicenterCords(x0, x1, x2, y0, y1, y2, x, y))

    return bariCords


def bariTrangles(v, f):
    trangles = []
    for i in range(len(f)):
        pixels = []
        x0 = v[int(f[i][0]) - 1][0] * 10000 + 500
        x1 = v[int(f[i][1]) - 1][0] * 10000 + 500
        x2 = v[int(f[i][2]) - 1][0] * 10000 + 500
        y0 = v[int(f[i][0]) - 1][1] * -10000
        y1 = v[int(f[i][1]) - 1][1] * -10000
        y2 = v[int(f[i][2]) - 1][1] * -10000
        xmin = min(x0, x1, x2)
        ymin = min(y0, y1, y2)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)
        if xmin < 0: xmin = 0
        if ymin > 0: ymin = 0
        for x in range(int(xmin), int(xmax)+1):
            for y in range(int(ymin), int(ymax)+1):
                flag = True
                for g in baicenterCords(x0, x1, x2, y0, y1, y2, x, y):
                    if g < 0:
                        flag = False
                if(flag):
                    pixels.append([x, y])

        trangles.append(pixels)
    return trangles


image_matrix_white = np.full((200, 200), 255, dtype=np.uint8)
image_matrix_black = np.full((200, 200), 0, dtype=np.uint8)
image_matrix_red = np.full((200, 200, 3), (255, 0, 0), dtype=np.uint8)

matrix = np.matrix((200, 200))

x0, y0 = 10, 170
x1, y1 = 80, 20

point_count = 100

image_matrix = np.full((200, 200, 3), (0, 0, 0), dtype=np.uint8)

img = Image.new(mode='L', size=[200, 200])
for i in range(len(image_matrix)):
    for j in range(len(image_matrix[i])):
        img.putpixel((i, j), ((i + j) % 256))

image1 = Image.fromarray(image_matrix_white, mode='L')
image2 = Image.fromarray(image_matrix_black, mode='L')
image3 = Image.fromarray(image_matrix_red, mode='RGB')
img.save('img.png')
image1.save('image1.png')
image2.save('image2.png')
image3.save('image3.png')

line_img1 = Image.new(mode='L', size=[200, 200])
line_img2 = Image.new(mode='L', size=[200, 200])
line_img3 = Image.new(mode='L', size=[200, 200])
line_img4 = Image.new(mode='L', size=[200, 200])

for a in range(12 + 1):
    line1(100, 100, 100 + 95 * math.cos((2 * math.pi * a) / 13), 100 + 95 * math.sin((2 * math.pi * a) / 13), line_img1)
    line2(100, 100, 100 + 95 * math.cos((2 * math.pi * a) / 13), 100 + 95 * math.sin((2 * math.pi * a) / 13), line_img2)
    line3(100, 100, 100 + 95 * math.cos((2 * math.pi * a) / 13), 100 + 95 * math.sin((2 * math.pi * a) / 13), line_img3)
    line4(100, 100, 100 + 95 * math.cos((2 * math.pi * a) / 13), 100 + 95 * math.sin((2 * math.pi * a) / 13), line_img4)

line_img1.save("lines1.png")
line_img2.save("lines2.png")
line_img3.save("lines3.png")
line_img4.save("lines4.png")

model = Image.new(mode='L', size=[1000, 1000])
v = objParse_v('model_1.obj')
f = objParse_f('model_1.obj')
for i in v:
    model.putpixel((int(i[0] * 10000) - 500, int(i[1] * -10000)), 255)

model.save("model.png")

model_v = Image.new(mode='L', size=[1000, 1000])


def draw(v, f, img):
    for i in f:
        line4(v[int(i[0] - 1)][0] * 10000 - 500, v[int(i[0] - 1)][1] * -10000, v[int(i[1] - 1)][0] * 10000 - 500,
              v[int(i[1] - 1)][1] * -10000, img)
        line4(v[int(i[1] - 1)][0] * 10000 - 500, v[int(i[1] - 1)][1] * -10000, v[int(i[2] - 1)][0] * 10000 - 500,
              v[int(i[2] - 1)][1] * -10000, img)
        line4(v[int(i[2] - 1)][0] * 10000 - 500, v[int(i[2] - 1)][1] * -10000, v[int(i[0] - 1)][0] * 10000 - 500,
              v[int(i[0] - 1)][1] * -10000, img)

def normal(v, f):
    a = []
    for i in range(len(f)):
        x0 = v[int(f[i][0]) - 1][0]
        x1 = v[int(f[i][1]) - 1][0]
        x2 = v[int(f[i][2]) - 1][0]
        y0 = v[int(f[i][0]) - 1][1]
        y1 = v[int(f[i][1]) - 1][1]
        y2 = v[int(f[i][2]) - 1][1]
        z0 = v[int(f[i][0]) - 1][2]
        z1 = v[int(f[i][1]) - 1][2]
        z2 = v[int(f[i][2]) - 1][2]
        n = np.cross([x1 - x0, y1 - y0, z1 - z0], [x1 - x2, y1 - y2, z1 - z2])
        array = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[x1 - x0, 0, 0], [0, y1 - y0, 0], [0, 0, z1 - z0]],
                 [[x1 - x2, 0, 0], [0, y1 - y2, 0], [0, 0, z1 - z2]]]
        detn = np.linalg.det(array)
        a.append([list(n), list(detn)])

    return a


draw(v, f, model_v)

model_v.save("modelv.png")


#podpivas = Image.new(mode='RGB', size=[1000, 1000])
#var = bariTrangles(v, f)
#for i in range(len(var)):
#    colour0 = randint(0, 255)
#    colour1 = randint(0, 255)
#    colour2 = randint(0, 255)
#    for j in var[i]:
#        podpivas.putpixel((int(j[0]), int(j[1])), (colour0, colour1, colour2))


#podpivas.save("podpivas.png")

print(normal(v, f))