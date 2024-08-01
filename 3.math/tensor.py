import torch

def f(a: torch.Tensor):
        b = a * 2
        while b.norm() < 1000:
                b = b * 2
        if b.sum() > 0:
                c = b
        else:
                c = 100 * b
        return c


if __name__ == '__main__':
        print(torch.__version__)
        # create a tensor
        if not torch.cuda.is_available():
                pass
        else:
                x = torch.tensor([1.0, 2.0, 3.0, 4.0])
                x.requires_grad = True
                print(x.shape)
                y = torch.dot(x, x)
                print(y, y.shape)
                y.backward()
                print(x.grad)
                y = torch.sum(x)
                print(y, y.shape)
                x.grad.zero_()
                y.backward()
                print(x.grad)

                x.grad.zero_()
                y = x * x
                u = y.detach()
                z = u * x
                print(u)
                print(x.grad)

                a = torch.randn(size=(), requires_grad=True)
                d = f(a)
                d.backward()
                print(a, d)
                print(a.grad)
                print(d / a)
