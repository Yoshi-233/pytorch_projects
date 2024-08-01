import torch

print(torch.__version__)

if __name__ == '__main__':
        if not torch.cuda.is_available():
                print("CUDA is not available")
                pass
        else:
                print("CUDA is available")
                arr = torch.arange(2 * 20).reshape((2, 5, 4))
                print(arr)

                print(arr.sum(axis=[0], keepdim=True))
                print(arr.numel())
                arr2 = torch.arange(20).reshape((5, 4))
                print(arr2.cumsum(axis=1)) # 计算某个轴的累加和