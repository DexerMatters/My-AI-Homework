import dataset
import model as mdl
import torch
import tqdm
import matplotlib.pyplot as plt

def main():
    train_dataloader = dataset.TrainDataloader("./data/PokemonData", 32)
    test_dataloader = dataset.TestDataloader("./data/PokemonData", 1)
    model = mdl.DenseNet(150)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    losses = []
    auc = []
    for epoch in range(100):
        model.train()
        for _, (inputs, labels) in enumerate(tqdm.tqdm(train_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        lr_scheduler.step()
        model.eval()
        correct = 0
        total = 0
        for _, (inputs, labels) in enumerate(tqdm.tqdm(test_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        auc.append(accuracy)
        print(f"Epoch {epoch}, Loss {loss.item()}, Accuracy {accuracy}")
        # Save the best model
        if accuracy >= max(auc):
            torch.save(model.state_dict(), "model.pth")    
        
    # Visualize the training curve
    plt.plot(losses)
    plt.savefig("loss.png")
    plt.clf()
    plt.plot(auc)
    plt.savefig("accuracy.png")
    
    

if __name__ == "__main__":
    main()
    