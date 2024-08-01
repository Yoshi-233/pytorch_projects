import torch

print(torch.__version__)

if __name__ == '__main__':
        if not torch.cuda.is_available():
                print("CUDA is not available")
                pass
        else:
                print("CUDA is available")
                arr_org = torch.arange(12).reshape(3, 4)
                arr_0 = arr_org.view(3, 4)
                arr_1 = arr_org.reshape(2, 6)
                # arr_1.resize_(4, 4)
                # arr_0[0][0] = 100
                # arr_1[3][0] = 200
                # print("data:")
                # print(arr_org)
                # print(arr_0)
                # print(arr_1)
                # print("data_ptr:")
                # print(hex(id(arr_org.data_ptr())))
                # print(hex(id(arr_0.data_ptr())))
                # print(hex(id(arr_1.data_ptr())))
                # print("arr_org is arr_1:", arr_org is arr_0)
                # print("arr[0]:")
                # print(hex(id(arr_org[0])))
                # print(hex(id(arr_0[0])))
                # print(hex(id(arr_1[0])))
                # print("arr:")
                # print(hex(id(arr_org)))
                # print(hex(id(arr_0)))
                # print(hex(id(arr_1)))

                print(type(arr_org.reshape(2, 6)))

