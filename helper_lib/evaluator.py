import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Compute average loss and accuracy of a classifier on a dataset."""
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    print(f"Test loss={avg_loss:.4f}  accuracy={accuracy:.4f}")
    return avg_loss, accuracy
