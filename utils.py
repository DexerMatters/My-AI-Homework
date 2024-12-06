import torch
import yaml
import dataset

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
                model.parameters(),
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["l2"],
            )
        case _:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")
    return optimizer


def new_dataloaders(train_set, val_set, test_set, batch_sizes):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_sizes[0], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_sizes[1])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_sizes[2])
    return train_loader, val_loader, test_loader


def split_dataset(ds):
    splits = get_config()["data"]["splits"]
    use_augment = get_config()["data"]["use_augment"] == 1
    dataset_size = len(ds)

    # Split the dataset
    train_size = int(splits[0] * dataset_size)
    test_size = int(splits[1] * dataset_size)

    # Randomly split the dataset into ImageDataset objects
    indices = torch.randperm(dataset_size).tolist()
    train_dataset = dataset.ImageSubset(ds, indices[:train_size], use_augment)
    test_dataset = dataset.ImageSubset(ds, indices[train_size : train_size + test_size])
    val_dataset = dataset.ImageSubset(ds, indices[train_size + test_size :])
    return train_dataset, test_dataset, val_dataset


def adjust_learning_rate(optimizer, epoch, lr_strategy, lr_decay_step):
    current_learning_rate = lr_strategy[epoch // lr_decay_step]
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_learning_rate
        print("Learning rate sets to {}.".format(param_group["lr"]))
