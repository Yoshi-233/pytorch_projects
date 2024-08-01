import torch

print(torch.__version__)

if __name__ == '__main__':
        if not torch.cuda.is_available():
                print("CUDA is not available")
                pass
        else:
                print("CUDA is available")
                arr_1 = torch.arange(12).reshape(2, 3, 2)
                arr_2 = torch.tensor([[[1, 2], [3, 4], [5, 6]],
                                      [[7, 8], [9, 10], [11, 12]]],
                                     dtype=torch.float32)
                # arr = torch.cat((arr_1, arr_2), dim=0)
                # arr_1[0][0] = 100
                print(arr_1)
                print(arr_2)

                arr = torch.cat((arr_1, arr_2), dim=1)
                print(arr)
