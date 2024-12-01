import os

import torch
import torch.nn as nn
import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DenseNet(nn.Module):
    def __init__(self, n_class, name):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 7, stride=2, padding=7),
            nn.MaxPool2d(6, 2),
        )

        cs = utils.get_config()["models"][name]

        self.layer_dense2 = DenseBlock(cs["c1"])
        self.transition2 = self.Transition_Layer(cs["c1"] * 5, cs["c2"])

        self.layer_dense3 = DenseBlock(cs["c2"])
        self.transition3 = self.Transition_Layer(cs["c2"] * 5, cs["c3"])

        self.layer_dense4 = DenseBlock(cs["c3"])
        self.transition4 = self.Transition_Layer(cs["c3"] * 5, cs["c4"])

        self.layer_dense5 = DenseBlock(cs["c4"])

        self.layer_pool5 = nn.AvgPool2d(7, 7)

        self.Linear = nn.Linear(240, n_class)

    def Transition_Layer(self, in_, out):
        transition = nn.Sequential(
            nn.BatchNorm2d(in_),
            nn.ReLU(),
            nn.Conv2d(in_, out, 1),
            nn.AvgPool2d(2, 2),
        )
        return transition

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_dense2(x)
        x = self.transition2(x)

        x = self.layer_dense3(x)
        x = self.transition3(x)

        x = self.layer_dense4(x)
        x = self.transition4(x)

        x = self.layer_dense5(x)

        x = self.layer_pool5(x)

        x = x.view(x.size(0), -1)

        x = self.Linear(x)
        return x

    @staticmethod
    def get_model(name, n_class):
        if name in ["dense-net-121", "dense-net-169", "dense-net-201", "dense-net-264"]:
            return DenseNet(n_class, name)
        else:
            raise ValueError("Model not found")


class DenseBlock(nn.Module):
    def __init__(self, in_channel):
        super(DenseBlock, self).__init__()
        self.d1 = self.Conv_Block(in_channel, in_channel)
        self.d2 = self.Conv_Block(2 * in_channel, in_channel)
        self.d3 = self.Conv_Block(4 * in_channel, in_channel)
        self.d4 = self.Conv_Block(8 * in_channel, in_channel)

    @staticmethod
    def Conv_Block(in_channel, out):
        Conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out, 1),
            nn.BatchNorm2d(out),
            nn.ReLU(),
            nn.Conv2d(out, out, 3, padding=1),
        )
        return Conv

    def forward(self, x):
        x1 = self.d1(x)
        x_cat1 = torch.cat((x, x1), dim=1)

        x2 = self.d2(x_cat1)
        x_cat2 = torch.cat((x2, x_cat1, x1), dim=1)

        x3 = self.d3(x_cat2)
        x_cat3 = torch.cat((x3, x_cat2, x_cat1, x1), dim=1)

        x4 = self.d4(x_cat3)

        x = torch.cat((x4, x3, x2, x1, x), dim=1)

        return x


if __name__ == "__main__":
    inputs = torch.randn(10, 3, 224, 224)
    model = DenseNet(n_class=6)
    outputs = model(inputs)
    print(model)
    print(outputs.shape)
