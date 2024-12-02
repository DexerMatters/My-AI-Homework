from sklearn.model_selection import KFold
import dataset as ds
import model as mdl
import torch
import tqdm
import utils
import matplotlib.pyplot as plt


def main():

    config = utils.get_config()["train"]
    models = utils.get_config()["models"]
    model = mdl.get_model("densenet201", models, 150)

    # Select the loss function
    criterion = utils.get_criterion()

    # Select the optimizer
    optimizer = utils.get_optimizer(model)

    # Select the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_decay_step"], gamma=config["lr_decay_rate"]
    )

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ds.ImageDataset("./data")
    batch_size = config["batch_size"]

    loss_history = []
    auc_history = []

    # K-fold cross validation
    kf = KFold(n_splits=config["kfolds"], shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        # Create the train and test dataloaders
        train_dataset = ds.ImageSubset(dataset, train_idx, augmented=True)
        test_dataset = ds.ImageSubset(dataset, test_idx, augmented=False)

        # Train the model
        train_dataloader = utils.new_train_dataloader(train_dataset, batch_size)
        test_dataloader = utils.new_test_dataloader(test_dataset, batch_size)

        best_accuracy = 0.0
        for epoch in range(config["epochs"] // config["kfolds"]):

            # Train the model
            train(model, train_dataloader, criterion, optimizer, device, loss_history)

            # Test the model
            validate(
                model, test_dataloader, device, auc_history, best_accuracy, fold, epoch
            )

        # Update the learning rate
        lr_scheduler.step()

        # Plot the loss and accuracy
        plt.plot(loss_history)
        plt.plot(auc_history)
        plt.show()


def train(model, train_dataloader, criterion, optimizer, device, loss_history):
    model.train()
    losses = []
    for images, labels in tqdm.tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    loss_history.append(sum(losses) / len(losses))


def validate(model, test_dataloader, device, auc_history, best_accuracy, fold, epoch):
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
        accuracy = correct / total
        print(f"Fold {fold}, Epoch {epoch}, Accuracy: {accuracy}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"model_best_fold_{fold}.pth")


if __name__ == "__main__":
    main()
