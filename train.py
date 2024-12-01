from sklearn.model_selection import KFold
import dataset as ds
import model.dense_net as mdl
import torch
import tqdm
import utils
import matplotlib.pyplot as plt

def main():
    
    config = utils.get_config()["train"]
    model = mdl.DenseNet.get_model("dense-net-201", 150)
    
    # Select the optimizer
    optimizer = None
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")
    
    # Select the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config["lr_decay_step"], 
        gamma=config["lr_decay_rate"])
    
    # Select the loss function
    criterion = None
    if config["loss"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif config["loss"] == "mse":
        criterion = torch.nn.MSELoss()
    elif config["loss"] == "nll":
        criterion = torch.nn.NLLLoss()
    else:
        raise ValueError(f"Loss {config['loss']} not supported")
    
    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = ds.ImageDataset("./data", augmented=False)
    batch_size = config["batch_size"]
    
    # K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        train_dataset = ds.ImageSubset(dataset, train_idx, augmented=True)
        test_dataset = ds.ImageSubset(dataset, test_idx, augmented=False)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size, 
            sampler=torch.utils.data.RandomSampler(train_dataset),
            pin_memory=True, 
            shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size, 
            sampler=torch.utils.data.RandomSampler(test_dataset),
            pin_memory=True, 
            shuffle=False)
        
        loss_history = []
        auc_history = []
        
        for epoch in range(config["epochs"]):
            model.train()
            for images, labels in tqdm.tqdm(train_dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                loss_history.append(loss.item())
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                for images, labels in test_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    auc_history.append(correct / total)
                print(f"Fold {fold}, Epoch {epoch}, Accuracy: {correct / total}")
        lr_scheduler.step()
    
    

if __name__ == "__main__":
    main()
    