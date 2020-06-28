import unittest

from pandas import np

from model.Image.Image import Image, project_coordinate


class TestClassProjection(unittest.TestCase):
    def test_sanity_check_sum_of_bitmap(self):
        image = Image([[[1, 1, 0], [2, 2, 0], [3, 3, 0]]], 32, 32)
        self.assertEqual(32 * 32 - 3, np.sum(image.return_as_np()))

    def test_that_we_project_larger_image_to_smaller(self):
        image = Image([[[4, 4, 0]]], 4, 2)
        print(np.asarray([[1, 1], [1, 0]]))
        self.assertSequenceEqual([[1, 1], [1, 0]], image.BITMAP.tolist())

    def test_that_outcoming_dimension_are_correct(self):
        desired_dimension = 4
        image = Image([[[4, 4, 0]]], 8, desired_dimension)
        self.assertEqual(desired_dimension, len(image.BITMAP.tolist()))


class TestProjectCoordinate(unittest.TestCase):
    # Use basic example with incoming image dimension of 4
    # projection (bitmap) dim of 2

    def test_return_max_as_larger(self):
        x_or_y = project_coordinate(5, 2, 4)

        self.assertEqual(2, x_or_y)

    def test_return_coordinate_as_interpolation(self):
        x_or_y = project_coordinate(60, 10, 100)

        self.assertEqual(6, x_or_y)


if __name__ == '__main__':
    unittest.main()
