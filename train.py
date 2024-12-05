import os
from sklearn.model_selection import KFold
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
fig, ax = plt.subplots(3, 1, figsize=(16, 12), dpi=80)
ax[0].set_title("Loss and Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss/Accuracy")
ax[1].set_title("Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[2].set_title("Accuracy")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Accuracy")


def main():
    global ax

    config = utils.get_config()["train"]
    models = utils.get_config()["models"]
    aug = utils.get_config()["data"]["use_augment"]
    model = mdl.get_model("densenet264", models, 150)

    # Select the loss function
    criterion = utils.get_criterion()

    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ds.ImageDataset("./data/")
    batch_size = config["batch_size"]

    model_name = config["model"]

    # K-fold cross validation
    kf = KFold(n_splits=config["kfolds"], shuffle=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        # Reset the model
        model = mdl.get_model(model_name, models, 150)
        model.to(device)
        optimizer = utils.get_optimizer(model)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=config["lr_decay_step"], gamma=config["lr_decay_rate"]
        # )

        # Create the train and test dataloaders
        train_dataset = ds.ImageSubset(dataset, train_idx, augmented=aug)
        test_dataset = ds.ImageSubset(dataset, test_idx, augmented=False)

        # Train the model
        train_dataloader = utils.new_train_dataloader(train_dataset, batch_size)
        test_dataloader = utils.new_test_dataloader(test_dataset, batch_size)

        loss_history = []
        auc_history = []

        best_accuracy = 0.0
        for epoch in range(config["epochs"]):
            
            # Update the learning rate
            adjust_learning_rate(optimizer, epoch, config['lr_strategy'], config['lr_decay_step'])
            
            # Train the model
            loss = train(
                model, train_dataloader, criterion, optimizer, device, loss_history
            )

            # Test the model
            validate(
                model_name,
                model,
                loss,
                test_dataloader,
                device,
                loss_history,
                auc_history,
                best_accuracy,
                fold,
                epoch,
            )

            # Visualize the loss and accuracy
            ax[0].cla()
            ax[0].plot(loss_history, label="Loss")
            ax[0].plot(auc_history, label="Accuracy")
            ax[0].legend()

            ax[1].cla()
            ax[1].plot(loss_history, label="Loss")
            ax[1].legend()

            ax[2].cla()
            ax[2].plot(auc_history, label="Accuracy")
            ax[2].legend()

            plt.savefig(f"./checkpoints/{model_name}/fold_{fold}/loss_accuracy.png")
            plt.pause(0.01)

            # lr_scheduler.step()


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
    loss = sum(losses) / len(losses)
    loss_history.append(loss)
    return loss


def validate(
    model_name,
    model,
    loss,
    test_dataloader,
    device,
    loss_history,
    auc_history,
    best_accuracy,
    fold,
    epoch,
):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in tqdm.tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        auc_history.append(accuracy)
        print(f"Fold {fold}, Epoch {epoch}, Loss {loss}, Accuracy: {accuracy}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # Create the checkpoint directory
            os.makedirs(f"./checkpoints/{model_name}/fold_{fold}", exist_ok=True)

            torch.save(
                model.state_dict(),
                f"./checkpoints/{model_name}/fold_{fold}/model_best_fold.pth",
            )

            # Save the model metrics
            np.save(
                f"./checkpoints/{model_name}/fold_{fold}/loss_history.npy", loss_history
            )
            np.save(
                f"./checkpoints/{model_name}/fold_{fold}/auc_history.npy", auc_history
            )


plt.show(block=False)


if __name__ == "__main__":
    main()
