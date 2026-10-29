import torch


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Standard supervised training loop for classifiers."""
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  loss={running_loss/len(data_loader):.4f}")
    return model


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """
    Train a GAN.
    `optimizer` is a dict: {"generator": opt_G, "discriminator": opt_D},
    because the generator and discriminator are optimized separately.
    """
    G, D = model.generator, model.discriminator
    opt_G, opt_D = optimizer["generator"], optimizer["discriminator"]
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss_D, total_loss_G = 0.0, 0.0

        for real, _ in data_loader:          # labels are not needed for a GAN
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)    # labels for real images
            zeros = torch.zeros(b, 1, device=device)  # labels for fake images

            # ---- Train Discriminator: real -> 1, fake -> 0 ----
            z = torch.randn(b, model.z_dim, device=device)
            fake = G(z)
            # detach() so the generator is not updated in this step
            loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- Train Generator: make D classify fakes as real (1) ----
            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()

        n = len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}  loss_D={total_loss_D/n:.4f}  loss_G={total_loss_G/n:.4f}")

    return model
