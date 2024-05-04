import torch
import torch.nn as nn


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1, hidden_size=64, kernel_size=5, padding=2):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=padding),
                nn.Conv2d(32, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                nn.Conv2d(hidden_size, 32, kernel_size=kernel_size, padding=padding),
                nn.Conv2d(32, out_channels, kernel_size=kernel_size, padding=padding),
            ]
        )
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, hidden_layers=2):
        super(Encoder, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)

        _layer_list = []
        for _ in range(hidden_layers):
            _layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            _layer_list.append(nn.BatchNorm1d(hidden_dim))
            _layer_list.append(nn.Dropout(0.3))

        self.hidden_layers = nn.ModuleList(
            _layer_list
        )

        self.fc_mean  = nn.Linear(hidden_dim, latent_dim)
        self.fc_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x       = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))
        mean     = self.fc_mean(x)
        log_var  = self.fc_var(x)                      # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, image_shape, hidden_layers=2):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.fc = nn.Linear(latent_dim, hidden_dim)
        
        _layer_list = []
        for _ in range(hidden_layers):
            _layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            _layer_list.append(nn.BatchNorm1d(hidden_dim))
            _layer_list.append(nn.Dropout(0.3))

        self.hidden_layers = nn.ModuleList(
            _layer_list
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            h = self.LeakyReLU(layer(h))
        
        x_hat = torch.sigmoid(self.fc_out(h))
        x_hat = x_hat.view([-1] + self.image_shape)
        return x_hat

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
    
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, image_shape, hidden_layers=2):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, hidden_dim)
        
        _layer_list = []
        for _ in range(hidden_layers):
            _layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            _layer_list.append(nn.BatchNorm1d(hidden_dim))
            _layer_list.append(nn.Dropout(0.3))

        self.hidden_layers = nn.ModuleList(
            _layer_list
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            h = self.LeakyReLU(layer(h))
        
        x_hat = torch.sigmoid(self.fc_out(h))
        x_hat = x_hat.view([-1] + list(self.image_shape))
        return x_hat

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers=2):
        super(Discriminator, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        
        _layer_list = []
        for _ in range(hidden_layers):
            _layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            _layer_list.append(nn.BatchNorm1d(hidden_dim))
            _layer_list.append(nn.Dropout(0.3))

        self.hidden_layers = nn.ModuleList(
            _layer_list
        )
        self.fc_out  = nn.Linear(hidden_dim, 1)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))

        x = self.fc_out(x)
        return x
