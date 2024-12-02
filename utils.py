import torch
import yaml

config = [None]


def load_config():
    with open("./cfg.yml", "r") as f:
        config[0] = yaml.safe_load(f)
    return config[0]


def get_config():
    if config[0] is None:
        config[0] = load_config()
    return config[0]


def get_criterion():
    criterion = None
    config = get_config()["train"]
    match config["loss"]:
        case "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        case "nll":
            criterion = torch.nn.NLLLoss()
        case "bce":
            criterion = torch.nn.BCELoss()
        case "bce_with_logits":
            criterion = torch.nn.BCEWithLogitsLoss()
        case "mse":
            criterion = torch.nn.MSELoss()
        case "l1":
            criterion = torch.nn.L1Loss()
        case "huber":
            criterion = torch.nn.SmoothL1Loss()
        case _:
            raise ValueError("Criterion not supported")
    return criterion


def get_optimizer(model):
    optimizer = None
    config = get_config()["train"]
    match config["optimizer"]:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        case "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=config["lr"], momentum=config["momentum"]
            )
        case _:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")
    return optimizer


def new_train_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=torch.utils.data.RandomSampler(dataset),
        pin_memory=True,
    )


def new_test_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        pin_memory=True,
        shuffle=False,
    )
