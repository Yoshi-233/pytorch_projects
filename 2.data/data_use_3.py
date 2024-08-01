import torch

print(torch.__version__)

if __name__ == '__main__':
        if not torch.cuda.is_available():
                print("CUDA is not available")
                pass
        else:
                print("CUDA is available")
                arr_org = torch.arange(12)
                arr_1 = arr_org.type(torch.float32)
                arr_2 = arr_org.to(torch.float32)
                arr_1[0] = 10
                arr_2[0] = 100
                print("data:")
                print(arr_org[0])
                print(arr_1[0])
                print(arr_2[0])
                print("data_ptr:")
                print(hex(id(arr_org.data_ptr())))
                print(hex(id(arr_1.data_ptr())))
                print(hex(id(arr_2.data_ptr())))
                print("arr[0]:")
                print(hex(id(arr_org[0])))
                print(hex(id(arr_1[0])))
                print(hex(id(arr_2[0])))
                print("arr:")
                print(hex(id(arr_org)))
                print(hex(id(arr_1)))
                print(hex(id(arr_2)))
