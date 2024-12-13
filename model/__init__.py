from model.dense_net import DenseNet
from model.res_net import ResNet
from model.vi_transformer import ViTForClassfication


def get_model(name, config, n_class):
    if name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        return ResNet(config, n_class, name)
    elif name in ["densenet121", "densenet169", "densenet201", "densenet264"]:
        return DenseNet(config, n_class, name)
    elif name in ["vit_224"]:
        return ViTForClassfication(config[name], n_class)
    else:
        raise ValueError("Model not supported")
