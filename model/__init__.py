from model.dense_net import DenseNet
from model.res_net import ResNet


def get_model(name, config, n_class):
    if name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return ResNet(config, n_class, name)
    elif name in ["densenet121", "densenet169", "densenet201", "densenet264"]:
        return DenseNet(config, n_class, name)
    else:
        raise ValueError("Model not supported")
