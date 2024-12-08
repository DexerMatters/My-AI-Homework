import os
from sklearn.model_selection import KFold
from datetime import date
import dataset as ds
import model as mdl
import torch
import tqdm
import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import adjust_learning_rate


# Create one plot for loss and accuracy
# their curves will be plotted in the same plot
fig, ax = plt.subplots(2, 1, figsize=(16, 12), dpi=80)
ax[0].set_title("Loss")
ax[1].set_title("Accuracy")
ax[0].set_xlabel("Epoch")
ax[1].set_xlabel("Epoch")


# Get the current date
today = date.strftime(date.today(), "%m-%d-%H-%M")


def main():
    global ax

    config = utils.get_config()["train"]
    models = utils.get_config()["models"]

    # Select the loss function
    criterion = utils.get_criterion()

    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ds.ImageDataset(utils.get_config()["data"]["root"])
    batch_size = config["batch_size"]

    model_name = config["model"]
    model = mdl.get_model(model_name, models, 150)
    model.to(device)
    optimizer = utils.get_optimizer(model)

    train_dataset, test_dataset, val_dataset = utils.split_dataset(dataset)

    train_dataloader, val_dataloader, test_dataloader = utils.new_dataloaders(
        train_dataset, val_dataset, test_dataset, [batch_size, 1, 1]
    )

    loss_history = []
    val_loss_history = []
    auc_history = []

    for epoch in range(config["epochs"]):

        # Update the learning rate
        adjust_learning_rate(
            optimizer, epoch, config["lr_strategy"], config["lr_decay_step"]
        )

        # Train the model
        loss = train(
            model, train_dataloader, criterion, optimizer, device, loss_history
        )

        # Validate the model
        validate(model, val_dataloader, criterion, device, val_loss_history)

        # Test the model
        test(model, test_dataloader, device, auc_history)

        # Print the loss and accuracy
        print(
            f"Epoch:\t{epoch + 1}\nLoss:\t{loss}\nVal Loss:\t{val_loss_history[-1]}\n Accuracy:\t{auc_history[-1]}"
        )

        # Save the model
        checkpoint(
            model,
            model_name,
            auc_history[-1],
            loss_history,
            val_loss_history,
            auc_history,
        )

        # Visualize the loss and accuracy
        ax[0].cla()
        ax[0].plot(loss_history, label="Loss", color="blue")
        ax[0].plot(val_loss_history, label="Validation Loss", color="red")
        ax[0].legend()

        ax[1].cla()
        ax[1].plot(auc_history, label="Accuracy")
        ax[1].legend()

        plt.savefig(f"./checkpoints/{model_name}/{today}/loss_accuracy.png")
        plt.pause(0.01)

        # lr_scheduler.step()


def train(model, train_dataloader, criterion, optimizer, device, loss_history):
    model.train()
    losses = []
    for images, labels in tqdm.tqdm(train_dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    loss = sum(losses) / len(losses)
    loss_history.append(loss)
    return loss


def validate(model, val_dataloader, criterion, device, val_loss_history):
    model.eval()
    valid_loss = 0.00
    with torch.no_grad():
        for images, labels in tqdm.tqdm(val_dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            valid_loss += criterion(outputs, labels).item()
    val_loss_history.append(valid_loss / len(val_dataloader))


def test(
    model,
    test_dataloader,
    device,
    auc_history,
):
    model.eval()
    with torch.no_grad():
        total = 0.00
        correct = 0.00
        for images, labels in tqdm.tqdm(test_dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        auc_history.append(accuracy)


def checkpoint(
    model, model_name, accuracy, loss_history, val_loss_history, auc_history
):
    # Save the best model
    if accuracy == max(auc_history):

        # Create the checkpoint directory
        os.makedirs(f"./checkpoints/{model_name}/{today}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"./checkpoints/{model_name}/{today}/model_best_fold.pth",
        )

        # Save the model metrics
        np.save(f"./checkpoints/{model_name}/{today}/loss_history.npy", loss_history)
        np.save(
            f"./checkpoints/{model_name}/{today}/val_loss_history.npy", val_loss_history
        )
        np.save(f"./checkpoints/{model_name}/{today}/auc_history.npy", auc_history)


plt.show(block=False)


if __name__ == "__main__":
    main()
