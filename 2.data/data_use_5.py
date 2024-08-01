import numpy as np

if __name__ == '__main__':
        arr_1 = np.arange(12).reshape(2, 3, 2)
        arr_2 = np.array([[[1, 2], [3, 4], [5, 6]],
                              [[7, 8], [9, 10], [11, 12]]],
                             dtype=np.float32)
        # arr = torch.cat((arr_1, arr_2), dim=0)
        # arr_1[0][0] = 100
        print(arr_1)
        print(arr_2)

