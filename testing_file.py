"""
Image, Input =
[0, 0, 1, 0, 0]
[0, 1, 1, 1, 0]
[1, 1, 1, 1, 1]
[0, 1, 1, 1, 0]
[0, 0, 1, 0, 0]

Kernel, Filter =
[-1, -.5, 0]
[-.5, 0, .5]
[0, .5, 1]
random values but if not numbers go diagonally, horizontally, vertically, etc.
"""
import numpy as np

image = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]])

kernel = np.array([[-1, -.5, 0],
                   [-.5, 0, .5],
                   [0, .5, 1]])

# Get the shape of the matrices
rows_5x5, cols_5x5 = image.shape
rows_3x3, cols_3x3 = kernel.shape

# Initialize an empty result matrix
result = np.zeros((rows_5x5 - rows_3x3 + 1, cols_5x5 - cols_3x3 + 1))
# turn kernel by 180°
rot_kernel = np.rot90(np.rot90(np.array(
    kernel)))  # the kernel when applied to the image appears exactly the same just rotated by 180°, so we counteract that

# Perform the sliding window operation
for row in range(rows_5x5 - rows_3x3 + 1):
    for col in range(cols_5x5 - cols_3x3 + 1):
        window = image[row:row + rows_3x3, col:col + cols_3x3]
        result[row, col] = np.sum(
            window * rot_kernel)  # output_size = (image_size - kernel_size) + 1 || ((5, 5) - (3, 3) + 1 = (2, 2) + 1 = (3, 3))

print("Result Matrix:")
print(result)
print(result.shape)
