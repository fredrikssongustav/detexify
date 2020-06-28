from pandas import np


def project_coordinate(value, smaller_dim, larger_dim):
    if value > larger_dim - 1:
        return smaller_dim
    else:
        return int(value * smaller_dim / larger_dim)


class Image:
    def __init__(self, sample, image_dim, bitmap_dim):
        self.sample = sample
        self.image_dim = image_dim
        self.bitmap_dim = bitmap_dim
        self.BITMAP = self.create_BITMAP()

    def create_BITMAP(self):
        BITMAP = np.ones([self.bitmap_dim, self.bitmap_dim], dtype=int)

        for line in self.sample:
            for dot in line:
                x = project_coordinate(dot[1], self.bitmap_dim, self.image_dim)
                y = project_coordinate(dot[0], self.bitmap_dim, self.image_dim)

                BITMAP[x][y] = 0

        return BITMAP

    def return_as_np(self):
        return self.BITMAP.reshape(1, self.bitmap_dim, self.bitmap_dim, 1)
