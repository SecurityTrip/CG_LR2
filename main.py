import numpy as np
from dataclasses import dataclass
from PIL import Image


def background(width, height):
    return Image.new('RGB', (width, height), (0, 0, 0))


@dataclass
class Vertex:
    x: float
    y: float
    z: float

    def __init__(self, x, y, z):
        self.xS, self.yS, self.zS = x, y, z
        self.x, self.y, self.z = self.modifyProjection(x, y, z)

        # self.x = self.modify(x, offset=500)
        # self.y = self.modify(y, mirror=True, offset=-300)
        # self.z = self.modify(z)

    def __str__(self):
        return f'Vertex: {self.x} {self.y} {self.z}'

    def toList(self):
        return [self.xS, self.yS, self.zS]

    @staticmethod
    def modify(x, mirror=False, offset=0):
        return x * 5000.0 * (-1 if mirror else 1) + offset

    @staticmethod
    def modifyProjection(a, b, c):
        alpha, beta, gamma = 0, 0, 0

        rotate_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), np.sin(alpha)],
            [0, -np.sin(alpha), np.cos(alpha)],
        ])

        rotate_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        rotate_z = np.array([
            [np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        r = np.dot(rotate_x, rotate_y)
        r = np.dot(r, rotate_z)

        x, y, z = list(np.dot(r, np.array([a, b, c])))

        scale = [50000, -50000]
        center = [500, 600]
        offset = [0.005, -0.045, 15.0]

        offset[0] += x
        offset[1] += y
        offset[2] += z

        z = offset[2] + c
        x = scale[0] * (a + offset[0]) / z + center[0]
        y = scale[1] * (b + offset[1]) / z + center[1]

        return x, y, z


@dataclass
class Polygon:
    v1: Vertex
    v2: Vertex
    v3: Vertex

    vn1: Vertex
    vn2: Vertex
    vn3: Vertex


def baricenter(v1: Vertex, v2: Vertex, v3: Vertex, point: tuple):
    x0, y0 = v1.x, v1.y
    x1, y1 = v2.x, v2.y
    x2, y2 = v3.x, v3.y
    x, y = point

    l1 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    l2 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    l3 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return l1, l2, l3


def triangle(p: Polygon, size: list, image: Image.Image, z_buffer: list):
    v1, v2, v3 = p.v1, p.v2, p.v3
    width, height = size
    min_x, max_x = max(int(min(v1.x, v2.x, v3.x, width)), 0), min(int(max(v1.x, v2.x, v3.x, 0)), width)
    min_y, max_y = max(int(min(v1.y, v2.y, v3.y, height)), 0), min(int(max(v1.y, v2.y, v3.y, 0)), height)

    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):

            l0, l1, l2 = baricenter(v1, v2, v3, (i, j))
            if all(map(lambda x: x > 0, (l0, l1, l2))):
                z = l0 * v1.z + l1 * v2.z + l2 * v3.z
                if z < z_buffer[i][j]:
                    z_buffer[i][j] = z

                light = np.array([1, 1, 1])

                vn1 = np.array(p.vn1.toList())
                a0 = np.dot(vn1, light) / np.linalg.norm(vn1) / np.linalg.norm(light)

                vn2 = np.array(p.vn1.toList())
                a1 = np.dot(vn2, light) / np.linalg.norm(vn2) / np.linalg.norm(light)

                vn3 = np.array(p.vn1.toList())
                a2 = np.dot(vn3, light) / np.linalg.norm(vn3) / np.linalg.norm(light)

                k = int(255 * (l0 * a0 + l1 * a1 + l2 * a2))
                image.putpixel((i, j), (k, k, k))


def calc_cross(p1, p2, p3):
    return list(np.cross(
        np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]),
        np.array([p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]]),
    ))


def angle_cos(normal):
    x1, y1, z1 = normal
    return z1 / ((x1 ** 2 + y1 ** 2 + z1 ** 2) ** 0.5)


def parse(filename):
    s = []
    n = []
    f = []
    with open(filename, 'r') as file:
        for line in file:
            if line:
                if line[:2] == 'v ':
                    v, x, y, z = line.split()
                    s.append(Vertex(float(x), float(y), float(z)))
                elif line[:3] == 'vn ':
                    v, x, y, z = line.split()
                    n.append(Vertex(float(x), float(y), float(z)))
                elif line[:2] == 'f ':
                    v, v1, v2, v3 = line.split()
                    f.append(Polygon(
                        s[int(v1.split('/')[0]) - 1],
                        s[int(v2.split('/')[0]) - 1],
                        s[int(v3.split('/')[0]) - 1],

                        n[int(v1.split('/')[2]) - 1],
                        n[int(v2.split('/')[2]) - 1],
                        n[int(v3.split('/')[2]) - 1],
                    ))
    return f


def main():
    size = [1000, 1000]
    z_buffer = [[999999.0 for _ in range(size[0])] for _ in range(size[1])]
    image = background(*size)

    f = parse('model_1.obj')
    for i, p in enumerate(f):
        cos = angle_cos(calc_cross(p.v1.toList(), p.v2.toList(), p.v3.toList()))
        if cos >= 0.0:
            continue

        triangle(p, size, image, z_buffer)
    image.save('model_triangles.jpg')


if __name__ == '__main__':
    main()
