
def train_unet(model, train_loader, val_loader, optimizer, criterion, eval_fn, epochs=10, device='cuda', verbose=True):
    def corrupt(x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
        return x * (1 - amount) + noise * amount

    train_eval_hist = []
    val_eval_hist = []
    loss_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    model.to(device)
    model.train()

    for epoch in rng:
        for x, y in train_loader:
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts

            noisy_x = corrupt(x, noise_amount)  # Create our noisy x
            pred = model(noisy_x)
            loss = criterion(pred, x)  # How close is the output to the true 'clean' x?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        model.eval()
        train_eval_hist.append(eval_fn(model, train_loader, device))
        val_eval_hist.append(eval_fn(model, val_loader, device))
        model.train()

    return loss_hist, train_eval_hist, val_eval_hist

def train_vae(model, train_loader, val_loader, optimizer, criterion, eval_fn, epochs=10, device='cuda', verbose=True):
    train_eval_hist = []
    val_eval_hist = []
    loss_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    model.to(device)
    model.train()

    for epoch in rng:
        for x, _ in train_loader:
            x = x.to(device)  # Data on the GPU
            pred, mean, log_var = model(x)
            loss = criterion(pred, x, mean, log_var)  # How close is the output to the true 'clean' x?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

        model.eval()
        train_eval_hist.append(eval_fn(model, train_loader, device))
        val_eval_hist.append(eval_fn(model, val_loader, device))
        model.train()

    return loss_hist, train_eval_hist, val_eval_hist

def train_gan(generator, discriminator, train_loader, optimizer_G, optimizer_D, criterion, device, epochs=10, verbose=True):
    train_G_hist = []
    train_D_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    for epoch in rng:
        for x, _ in train_loader:
            x = x.to(device)  # Data on the GPU
            # Train the discriminator
            optimizer_D.zero_grad()
            real_preds = discriminator(x)
            fake_preds = discriminator(generator(torch.randn(x.shape[0], generator.latent_dim).to(device)))
            loss_D = criterion(real_preds, torch.ones_like(real_preds)) + criterion(fake_preds, torch.zeros_like(fake_preds))
            loss_D.backward()
            optimizer_D.step()
            train_D_hist.append(loss_D.item())

            # Train the generator
            optimizer_G.zero_grad()
            fake_preds = discriminator(generator(torch.randn(x.shape[0], generator.latent_dim).to(device)))
            loss_G = criterion(fake_preds, torch.ones_like(fake_preds))
            loss_G.backward()
            optimizer_G.step()
            train_G_hist.append(loss_G.item())

        generator.eval()
        discriminator.eval()

    return train_G_hist, train_D_hist