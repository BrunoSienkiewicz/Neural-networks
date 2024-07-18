import torch
import numpy as np
from tqdm import tqdm
import optuna


def train_unet(model, train_loader, optimizer, criterion, epochs=10, device='cuda', verbose=True):
    def corrupt(x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
        return x * (1 - amount) + noise * amount

    loss_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    model.to(device)
    model.train()

    for epoch in rng:
        epoch_loss = []
        for x, _ in train_loader:
            x = x.to(device)  # Data on the GPU
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts

            noisy_x = corrupt(x, noise_amount)  # Create our noisy x
            pred = model(noisy_x)
            loss = criterion(pred, x)  # How close is the output to the true 'clean' x?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        epoch_loss = np.mean(np.array(epoch_loss))
        loss_hist.append(epoch_loss)

    return loss_hist

def train_conditioned_unet(model, train_loader, optimizer, criterion, epochs=10, max_steps=50, device='cuda', verbose=True, trial=None):
    def corrupt(x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
        return x * (1 - amount) + noise * amount

    loss_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    model.to(device)
    model.train()

    for epoch in rng:
        epoch_loss = []
        for x, y in train_loader:
            x = x.to(device)  # Data on the GPU
            y = y.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
            n_steps = np.random.randint(1, max_steps)
            for i in range(n_steps):
                noisy_x = corrupt(x, noise_amount)
            bs = x.shape[0]
            i = torch.tensor([i] * bs).to(device)
            pred = model(noisy_x, i, y)
            loss = criterion(pred, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        epoch_loss = np.mean(np.array(epoch_loss))
        loss_hist.append(epoch_loss)

        if trial is not None:
            trial.report(epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return loss_hist

def train_vae(model, train_loader, val_set, optimizer, loss_fn, eval_fn, epochs=10, device='cuda', verbose=True):
    eval_hist = []
    loss_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    model.to(device)
    model.train()

    for epoch in rng:
        epoch_loss = []
        for x, _ in train_loader:
            x = x.to(device)  # Data on the GPU
            out, means, log_var = model(x)
            loss = loss_fn(x, out, means, log_var) 
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(np.array(epoch_loss))
        loss_hist.append(epoch_loss)
        model.eval()
        eval_hist.append(eval_fn(model.decoder, val_set, device))
        model.train()

    return loss_hist, eval_hist

def train_gan(generator, discriminator, train_loader, val_set, val_loader, generator_optimizer, discriminator_optimizer, criterion, device, eval_fn, epochs=10, verbose=True):
    train_G_hist = []
    eval_G_hist = []
    train_D_hist = []
    D_fake_acc_hist = []
    D_real_acc_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    for epoch in rng:
        discriminator_fake_acc = []
        discriminator_real_acc = []
        epoch_G_loss = []
        epoch_D_loss = []
        for x, _ in train_loader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator_optimizer.zero_grad()
            # Format batch
            real_images = x.to(device)
            b_size = real_images.size(0)
            label = torch.ones((b_size,), dtype=torch.float, device=device) # Setting labels for real images
            # Forward pass real batch through D
            output = discriminator(real_images).view(-1)
            # Calculate loss on all-real batch
            error_discriminator_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            discriminator_real_acc.append(output.mean().item())

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, generator.latent_dim,device=device)
            # Generate fake image batch with Generator
            fake_images = generator(noise)
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=device)
            # Classify all fake batch with Discriminator
            output = discriminator(fake_images.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            error_discriminator_fake = criterion(output, label_fake)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            discriminator_fake_acc.append(output.mean().item())
            # Compute error of D as sum over the fake and the real batches
            error_discriminator = error_discriminator_real + error_discriminator_fake
            epoch_D_loss.append(error_discriminator.item())
            error_discriminator.backward()
            # Update D
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_optimizer.zero_grad()
            label = torch.ones((b_size,), dtype=torch.float, device=device)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake_images).view(-1)
            # Calculate G's loss based on this output
            error_generator = criterion(output, label)
            epoch_G_loss.append(error_generator.item())
            # Calculate gradients for G
            error_generator.backward()
            # Update G
            generator_optimizer.step()


        epoch_G_loss = np.mean(np.array(epoch_G_loss))
        train_G_hist.append(epoch_G_loss)
        epoch_D_loss = np.mean(np.array(epoch_D_loss))
        train_D_hist.append(epoch_D_loss)

        generator.eval()
        discriminator.eval()
        eval_G_hist.append(eval_fn(generator, val_set, device))
        D_fake_acc_hist.append(np.mean(np.array(discriminator_fake_acc)))
        D_real_acc_hist.append(np.mean(np.array(discriminator_real_acc)))
        generator.train()
        discriminator.train()

    return train_G_hist, train_D_hist, eval_G_hist, D_fake_acc_hist, D_real_acc_hist

def train_vaegan(vae, discriminator, train_loader, val_set, vae_optimizer, discriminator_optimizer, loss_fn, criterion, device, eval_fn, epochs=10, verbose=True, latent_dim=32):
    train_vae_hist = []
    train_D_hist = []
    eval_hist = []
    D_fake_acc_hist = []
    D_real_acc_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    vae.to(device)
    discriminator.to(device)
    vae.train()
    discriminator.train()

    for epoch in rng:
        discriminator_fake_acc = []
        discriminator_real_acc = []
        epoch_D_loss = []
        epoch_loss = []
        for x, _ in train_loader:
            x = x.to(device)  # Data on the GPU
            out, means, log_var = vae(x)
            loss = loss_fn(x, out, means, log_var) 
            epoch_loss.append(loss.item())
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            discriminator_optimizer.zero_grad()
            real_images = x.to(device)
            b_size = real_images.size(0)
            label = torch.ones((b_size,), dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            error_discriminator_real = criterion(output, label)
            discriminator_real_acc.append(output.mean().item())

            fake_images = vae.decoder(torch.randn([b_size, latent_dim]).to(device))
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=device)
            output = discriminator(fake_images.detach()).view(-1)
            error_discriminator_fake = criterion(output, label_fake)
            discriminator_fake_acc.append(output.mean().item())

            error_discriminator = error_discriminator_real + error_discriminator_fake
            epoch_D_loss.append(error_discriminator.item())
            error_discriminator.backward()
            discriminator_optimizer.step()

            fooling_loss = criterion(discriminator(fake_images), torch.ones_like(output))
            vae_optimizer.zero_grad()
            fooling_loss.backward()
            vae_optimizer.step()

        epoch_D_loss = np.mean(np.array(epoch_D_loss))
        train_D_hist.append(epoch_D_loss)

        epoch_loss = np.mean(np.array(epoch_loss))
        train_vae_hist.append(epoch_loss)
        vae.eval()
        eval_hist.append(eval_fn(vae.decoder, val_set, device))
        vae.train()

        discriminator.eval()
        D_fake_acc_hist.append(np.mean(np.array(discriminator_fake_acc)))
        D_real_acc_hist.append(np.mean(np.array(discriminator_real_acc)))
        discriminator.train()

    return train_vae_hist, train_D_hist, eval_hist, D_fake_acc_hist, D_real_acc_hist

def train_cvaegan(vae, discriminator, train_loader, val_set, vae_optimizer, discriminator_optimizer, loss_fn, criterion, device, eval_fn, evaluator, epochs=10, verbose=True, latent_dim=32, trial=None):
    train_vae_hist = []
    train_D_hist = []
    eval_hist = []
    D_fake_acc_hist = []
    D_real_acc_hist = []
    if verbose:
        rng = tqdm(range(epochs))
    else:
        rng = range(epochs)

    vae.to(device)
    discriminator.to(device)
    vae.train()
    discriminator.train()

    for epoch in rng:
        discriminator_fake_acc = []
        discriminator_real_acc = []
        epoch_D_loss = []
        epoch_loss = []
        for x, y in train_loader:
            x = x.to(device)  # Data on the GPU
            y = y.to(device)
            out, means, log_var = vae(x, y)
            loss = loss_fn(x, out, means, log_var) 
            epoch_loss.append(loss.item())
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            real_images = x.to(device)
            discriminator_optimizer.zero_grad()
            b_size = real_images.size(0)
            label = torch.ones((b_size,), dtype=torch.float, device=device)
            output = discriminator(real_images, y).view(-1)
            error_discriminator_real = criterion(output, label)
            discriminator_real_acc.append(output.mean().item())

            noise = torch.randn([b_size, latent_dim]).to(device)
            noise_y = torch.randint(0, vae.num_classes, (b_size,)).to(device)
            fake_images = vae.decoder(noise, noise_y)
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=device)
            output = discriminator(fake_images.detach(), noise_y).view(-1)
            error_discriminator_fake = criterion(output, label_fake)
            discriminator_fake_acc.append(output.mean().item())

            error_discriminator = error_discriminator_real + error_discriminator_fake
            epoch_D_loss.append(error_discriminator.item())
            error_discriminator.backward()
            discriminator_optimizer.step()

            fooling_loss = criterion(discriminator(fake_images, noise_y), torch.ones_like(output))
            vae_optimizer.zero_grad()
            fooling_loss.backward()
            vae_optimizer.step()

        epoch_D_loss = np.mean(np.array(epoch_D_loss))
        train_D_hist.append(epoch_D_loss)

        epoch_loss = np.mean(np.array(epoch_loss))
        train_vae_hist.append(epoch_loss)
        vae.eval()
        evaluation = eval_fn(evaluator, vae.decoder, val_set, device)
        eval_hist.append(evaluation)
        vae.train()

        discriminator.eval()
        D_fake_acc_hist.append(np.mean(np.array(discriminator_fake_acc)))
        D_real_acc_hist.append(np.mean(np.array(discriminator_real_acc)))
        discriminator.train()

        if trial is not None:
            trial.report(evaluation, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return train_vae_hist, train_D_hist, eval_hist, D_fake_acc_hist, D_real_acc_hist