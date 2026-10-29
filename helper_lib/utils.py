import torch


def save_model(model, path):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device="cpu"):
    """Load weights from disk into a model and switch it to eval mode."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
