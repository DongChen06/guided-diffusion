import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
# from torch.utils.tensorboard import SummaryWriter
import os
from numpy.random import randn
import argparse

torch.manual_seed(1)
device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='../datasets/CottonWeed_train', help="dir of training images")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--embedding_dim", type=int, default=100, help="dimensionality of the embedding space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(root=opt.data_dir, transform=transforms.Compose(
            [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(),
             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
        ), ), batch_size=opt.batch_size, shuffle=True,)

image_shape = (3, 256, 256)
image_dim = int(np.prod(image_shape))


# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_conditioned_generator = nn.Sequential(nn.Embedding(opt.n_classes, opt.embedding_dim),
                                                         nn.Linear(opt.embedding_dim, 64))

        self.latent = nn.Sequential(nn.Linear(opt.latent_dim, 8 * 8 * 512),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(nn.ConvTranspose2d(513, 64 * 8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 2, 64 * 1, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64 * 1, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 8, 8)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512, 8, 8)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        # print(image.size())
        return image


generator = Generator().to(device)
generator.apply(weights_init)
print(generator)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_condition_disc = nn.Sequential(nn.Embedding(opt.n_classes, opt.embedding_dim),
                                                  nn.Linear(opt.embedding_dim, 3 * 256 * 256))

        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 64 * 2, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 2, 64 * 4, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64 * 4, 64 * 8, 4, 3, 2, bias=False),
                                   nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Flatten(),
                                   nn.Dropout(0.4),
                                   nn.Linear(18432, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 256, 256)
        concat = torch.cat((img, label_output), dim=1)
        # print(concat.size())
        output = self.model(concat)
        return output


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator((z, labels))
    save_image(gen_imgs.data, "cgan/cGAN_samples/%d.png" % batches_done, nrow=n_row, normalize=True)


discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)
adversarial_loss = nn.BCELoss()


def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    # print(gen_loss)
    return gen_loss


def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


learning_rate = 0.0001
G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

os.makedirs("cgan/cGAN_samples", exist_ok=True)
os.makedirs("cgan/models", exist_ok=True)

D_loss_plot, G_loss_plot = [], []
for epoch in range(1, opt.n_epochs + 1):

    D_loss_list, G_loss_list = [], []
    for index, (real_images, labels) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long()

        real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))

        D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
        # print(discriminator(real_images))
        # D_real_loss.backward()

        noise_vector = torch.randn(real_images.size(0), opt.latent_dim, device=device)
        noise_vector = noise_vector.to(device)

        generated_image = generator((noise_vector, labels))
        output = discriminator((generated_image.detach(), labels))
        D_fake_loss = discriminator_loss(output, fake_target)

        # train with fake
        # D_fake_loss.backward()

        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)

        D_total_loss.backward()
        D_optimizer.step()

        # Train generator with real labels
        G_optimizer.zero_grad()
        G_loss = generator_loss(discriminator((generated_image, labels)), real_target)
        G_loss_list.append(G_loss)

        G_loss.backward()
        G_optimizer.step()

        print('Epoch: [%d/%d]: D_loss: %.3f, G_loss: %.3f' % (
            (epoch), opt.n_epochs, torch.mean(torch.FloatTensor(D_loss_list)), \
            torch.mean(torch.FloatTensor(G_loss_list))))


    if epoch % opt.sample_interval == 0:
        sample_image(n_row=10, batches_done=epoch)
        # save models
        torch.save(generator.state_dict(), 'cgan/models/generator_epoch_%d.pth' % (epoch))
        torch.save(discriminator.state_dict(), 'cgan/models/discriminator_epoch_%d.pth' % (epoch))

# generator.load_state_dict(torch.load('cgan/generator_epoch_1.pth'), strict=False)
# generator.eval()