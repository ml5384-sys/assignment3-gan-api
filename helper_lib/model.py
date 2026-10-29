import torch.nn as nn


class FCNN(nn.Module):
    """Fully connected network for 28x28 grayscale images."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Simple CNN: two conv blocks followed by a classifier."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 32 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EnhancedCNN(nn.Module):
    """CNN with BatchNorm and Dropout for better generalization."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- Module 6: GAN ----------------

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # Noise vector (B, 100) -> fully connected layer to 7*7*128
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 14 x 14 -> 1 x 28 x 28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),   # output pixels in range [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)   # reshape to (B, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 x 28 x 28 -> 64 x 14 x 14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14 -> 128 x 7 x 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),   # single output (logit)
        )

    def forward(self, x):
        # Returns a raw logit; torch.sigmoid() of it is the real/fake probability.
        # BCEWithLogitsLoss applies the sigmoid internally during training.
        return self.net(x)


class GAN(nn.Module):
    """Wrapper that holds both the generator and the discriminator."""
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()


def get_model(model_name):
    """Return a model by name: one of FCNN, CNN, EnhancedCNN, VAE, GAN."""
    if model_name == "FCNN":
        model = FCNN()
    elif model_name == "CNN":
        model = CNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "GAN":
        model = GAN(z_dim=100)
    # elif model_name == "VAE":   # VAE is from Module 5; add it here if you build it
    #     model = VAE()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model
