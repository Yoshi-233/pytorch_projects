from IPython import display
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from tqdm import tqdm, trange
from collections import abc

def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition


def net(X):
        return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(pred, label):
        return -torch.log(pred[range(len(pred)), label]).requires_grad_(True)

def accuracy(pred, label):
        if len(pred) > 1 and pred.shape[1] > 1:
                pred = pred.argmax(axis=1)
        cmp = pred.type(label.dtype) == label
        return float(cmp.type(torch.float32).mean())

class Accumulator:
        def __init__(self, n):
                self.data = [0.0] * n

        def add(self, *args):
                self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
                self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
                return self.data[idx]


def evaluate_accuracy(data_iter, net):
        if isinstance(net, torch.nn.Module):
                net.eval()  # Set the model to evaluation mode
        metric = Accumulator(2)  # Accumulator for loss and accuracy
        for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())

        return metric[0] / metric[1]


def train_epoch(net, train_iter, lose, optimizer):
        if isinstance(net, torch.nn.Module):
                net.train()  # Set the model to training mode
        metric = Accumulator(3)  # Accumulator for loss, accuracy, and count
        for X, y in train_iter:
                y_hat = net(X)
                l = lose(y_hat, y)
                if isinstance(optimizer, torch.optim.Optimizer):
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
                        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
                else:
                        l.sum().backward()
                        optimizer(X.shape[0])
                        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                     ylim=None, xscale='linear', yscale='linear',
                     fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                     figsize=(3.5, 2.5)):
                """Incrementally plot multiple lines."""
                if legend is None:
                        legend = []
                d2l.use_svg_display()
                self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
                if nrows * ncols == 1:
                        self.axes = [self.axes, ]
                # Use a lambda function to capture arguments
                self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale,
                                                        legend)
                self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):
                """Add multiple data points into the figure."""
                if not hasattr(y, "__len__"):
                        y = [y]
                n = len(y)
                if not hasattr(x, "__len__"):
                        x = [x] * n
                if not self.X:
                        self.X = [[] for _ in range(n)]
                if not self.Y:
                        self.Y = [[] for _ in range(n)]
                for i, (a, b) in enumerate(zip(x, y)):
                        if a is not None and b is not None:
                                self.X[i].append(a)
                                self.Y[i].append(b)
                self.axes[0].cla()
                for x, y, fmt in zip(self.X, self.Y, self.fmts):
                        self.axes[0].plot(x, y, fmt)
                self.config_axes()
                display.display(self.fig)
                display.clear_output(wait=True)


def train(net, train_iter, test_iter, loss, num_epochs, optimizer):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])

        for epoch in trange(num_epochs):
                train_metrics = train_epoch(net, train_iter, loss, optimizer)
                test_acc = evaluate_accuracy(test_iter, net)
                animator.add(epoch + 1, train_metrics + (test_acc,))

        train_loss, train_acc = train_metrics


def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 1, (num_inputs, num_outputs)).requires_grad_(True)
b = torch.zeros(num_outputs).requires_grad_(True)

lr = 0.1
num_epochs = 10

if __name__ == '__main__':
        train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
