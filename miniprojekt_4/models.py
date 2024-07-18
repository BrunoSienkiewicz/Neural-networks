import torch
import torch.nn as nn


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1, hidden_size=64, kernel_size=5, padding=2):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ),
            ]
        )
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = l(x)  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = l(x)  # Through the layer and the activation function

        return x
    
class LargeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_size=64, kernel_size=5, padding=2):
        super().__init__()
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ),
            ]
        )
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = l(x)
            if i < 4:
                h.append(x)
                x = self.downscale(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = l(x)

        return x
    

class ConditionedLargeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_size=64, kernel_size=5, padding=2, n_classes=10, n_time_steps=10):
        super().__init__()
        self.class_embedding = nn.Embedding(n_classes, 4)
        self.time_embedding = nn.Embedding(n_time_steps, 4)
        self.down_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels + 4 + 4, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_size, 32, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(32, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ),
            ]
        )
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x, t, y):
        bs, ch, w, h = x.shape
        class_emb = self.class_embedding(y)
        time_emb = self.time_embedding(t)
        class_emb = class_emb.view(bs, class_emb.shape[1], 1, 1).expand(bs, class_emb.shape[1], w, h)
        time_emb = time_emb.view(bs, time_emb.shape[1], 1, 1).expand(bs, time_emb.shape[1], w, h)
        x = torch.cat([x, class_emb, time_emb], dim=1)
        h = []
        for i, l in enumerate(self.down_layers):
            x = l(x)
            if i < 4:
                h.append(x)
                x = self.downscale(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = l(x)

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
        log_var  = self.fc_var(x)               
        
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
    
class ConvEncoder(Encoder):
    def __init__(self, input_dim, hidden_dim, latent_dim, image_shape, hidden_layers=2):
        super(ConvEncoder, self).__init__(input_dim, hidden_dim, latent_dim, hidden_layers)
        self.image_shape = image_shape
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128*4*4, hidden_dim)

    def forward(self, x):
        x = self.LeakyReLU(self.conv1(x))
        x = self.LeakyReLU(self.bn1(x))
        x = self.LeakyReLU(self.conv2(x))
        x = self.LeakyReLU(self.conv3(x))
        x = x.view(-1, 128*4*4)
        x = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))
        mean     = self.fc_mean(x)
        log_var  = self.fc_var(x)                    
        
        return mean, log_var
    
class ConvDecoder(Decoder):
    def __init__(self, latent_dim, hidden_dim, output_dim, image_shape, hidden_layers=2):
        super(ConvDecoder, self).__init__(latent_dim, hidden_dim, output_dim, image_shape, hidden_layers)
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, image_shape[0], kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        h = self.LeakyReLU(self.fc(x))
        h = h.view(-1, 128, 4, 4)
        h = self.LeakyReLU(self.conv1(h))
        h = self.LeakyReLU(self.bn1(h))
        h = self.LeakyReLU(self.conv2(h))
        h = self.LeakyReLU(self.conv3(h))
        h = torch.sigmoid(h)
        return h

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
    
class ConvGenerator(Generator):
    def __init__(self, latent_dim, hidden_dim, output_dim, image_shape, hidden_layers=2):
        super(ConvGenerator, self).__init__(latent_dim, hidden_dim, output_dim, image_shape, hidden_layers)
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, image_shape[0], kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(image_shape[0])

    def forward(self, x):
        h = self.LeakyReLU(self.fc(x))
        h = h.view(-1, 128, 4, 4)
        h = self.LeakyReLU(self.conv1(h))
        h = self.LeakyReLU(self.bn1(h))
        h = self.LeakyReLU(self.conv2(h))
        h = self.LeakyReLU(self.bn2(h))
        h = self.LeakyReLU(self.conv3(h))
        h = torch.sigmoid(self.bn3(h))
        return h
    
class ConvDiscriminator(Discriminator):
    def __init__(self, input_dim, hidden_dim, image_shape, hidden_layers=2):
        super(ConvDiscriminator, self).__init__(input_dim, hidden_dim, hidden_layers)
        self.image_shape = image_shape
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128*4*4, hidden_dim)

    def forward(self, x):
        x = self.LeakyReLU(self.conv1(x))
        x = self.LeakyReLU(self.bn1(x))
        x = self.LeakyReLU(self.conv2(x))
        x = self.LeakyReLU(self.bn2(x))
        x = self.LeakyReLU(self.conv3(x))
        x = self.LeakyReLU(self.bn3(x))
        x = x.view(-1, 128*4*4)
        x = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))

        x = self.fc_out(x)
        return x

class CDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers=2, num_classes=10):
        super(CDiscriminator, self).__init__()

        self.fc = nn.Linear(input_dim + num_classes, hidden_dim)
        self.num_classes = num_classes
        self.fc_out  = nn.Linear(hidden_dim, 1)
        
        _layer_list = []
        for _ in range(hidden_layers):
            _layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            _layer_list.append(nn.BatchNorm1d(hidden_dim))
            _layer_list.append(nn.Dropout(0.3))

        self.hidden_layers = nn.ModuleList(
            _layer_list
        )

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        x = torch.flatten(x, 1)
        x = torch.cat([x, y], dim=1)
        x = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))

        x = self.fc_out(x)
        return x


class ConvCEncoder(ConvEncoder):
    def __init__(self, input_dim, hidden_dim, latent_dim, image_shape, hidden_layers=2, num_classes=10):
        super().__init__(input_dim, hidden_dim, latent_dim, image_shape, hidden_layers)
        self.num_classes = num_classes
        self.fc = nn.Linear(128*4*4 + num_classes, hidden_dim)

    def forward(self, x, y):
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        x = self.LeakyReLU(self.conv1(x))
        x = self.LeakyReLU(self.bn1(x))
        x = self.LeakyReLU(self.conv2(x))
        x = self.LeakyReLU(self.conv3(x))
        x = x.view(-1, 128*4*4)
        x = torch.cat([x, y], dim=1)
        x = self.LeakyReLU(self.fc(x))
        for layer in self.hidden_layers:
            x = self.LeakyReLU(layer(x))
        mean     = self.fc_mean(x)
        log_var  = self.fc_var(x)                    
        
        return mean, log_var

class ConvCDecoder(ConvDecoder):
    def __init__(self, latent_dim, hidden_dim, output_dim, image_shape, hidden_layers=2, num_classes=10):
        super().__init__(latent_dim, hidden_dim, output_dim, image_shape, hidden_layers)
        self.num_classes = num_classes
        self.fc = nn.Linear(latent_dim + num_classes, 128*4*4)

    def forward(self, x, y):
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()
        x = torch.cat([x, y], dim=1)
        h = self.LeakyReLU(self.fc(x))
        h = h.view(-1, 128, 4, 4)
        h = self.LeakyReLU(self.conv1(h))
        h = self.LeakyReLU(self.bn1(h))
        h = self.LeakyReLU(self.conv2(h))
        h = self.LeakyReLU(self.conv3(h))
        h = torch.sigmoid(h)
        return h

class ConvCVAE(VAE):
    def __init__(self, encoder, decoder, num_classes=10):
        super().__init__(encoder, decoder)
        self.num_classes = num_classes

    def forward(self, x, y):
        mean, log_var = self.encoder(x, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) 
        x_hat = self.decoder(z, y)
        return x_hat, mean, log_var