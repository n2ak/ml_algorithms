import torch
import tqdm
import numpy as np
from torch import nn, optim
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


class Generator(nn.Module):
    def __init__(self, gray_shape, color_shape, z_shape, number_of_blocks=3) -> None:
        super().__init__()
        z_dim = np.prod(z_shape)
        gray_dim = np.prod(gray_shape)
        color_dim = np.prod(color_shape)

        self.out_size = color_shape

        def downsample(inn, out, kernel_size=3, stride=2):
            return nn.Sequential(
                nn.Conv2d(inn, out, kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(),

                # nn.MaxPool2d(2),
            )

        def upsample(inn, out, kernel_size=4, stride=2):
            return nn.Sequential(
                nn.ConvTranspose2d(inn, out, kernel_size,
                                   stride=stride, padding=1),
                nn.BatchNorm2d(out),
                nn.LeakyReLU()
            )
        self.downsamples = nn.ModuleList()
        inn, out = 2, 64
        for _ in range(number_of_blocks):
            self.downsamples.add_module(f"donw{_}",
                                        downsample(inn, out),
                                        )
            inn, out = out, out*2

        self.bneck = nn.Sequential(
            nn.Conv2d(inn, out, 3, stride=1, padding="same"),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(out, inn, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.upsamples = nn.ModuleList()
        inn = out
        for _ in range(number_of_blocks):
            out = inn//4
            self.upsamples.add_module(f"up{_}", upsample(inn, out))
            inn = inn//2

        self.last = nn.Sequential(
            nn.ConvTranspose2d(out, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, gray_image, z):
        x = torch.concatenate([gray_image, z], 1)
        conns = []
        for mod in self.downsamples:
            x = mod(x)
            conns.append(x)
        x = self.bneck(x)
        for conn, mod in zip(reversed(conns), self.upsamples):
            x = cat(conn, x)
            x = mod(x)
        x = self.last(x)
        from torchvision.transforms.functional import resize, normalize
        if self.out_size[1:] != x.shape[2:]:
            color_img_shape = self.out_size[1:]
            out = torch.empty((x.size(0), *self.out_size))
            for i in range(x.size(0)):
                img = x[i]
                # img = normalize(img, -1, 2)
                img = resize(img, color_img_shape, antialias=None)
                # img = normalize(img, .5, .5)
                out[i] = img
            return out
        return x


def cat(x1, x2):
    import torch.nn.functional as F
    diffY = x2.size(2) - x1.size(2)
    diffX = x2.size(3) - x1.size(3)
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    return torch.cat([x2, x1], dim=1)


class Discriminator(nn.Module):
    def __init__(self, gray_shape, color_shape) -> None:
        super().__init__()

        gray_dim = np.prod(gray_shape)
        color_dim = np.prod(color_shape)

        def conv_block(inn, out, kernel_size=3, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(inn, out, kernel_size, stride, padding),
                nn.BatchNorm2d(out),
                nn.ReLU(),
            )
        self.seq = nn.Sequential(
            conv_block(4, 64, stride=2),
            conv_block(64, 128, stride=2),
            conv_block(128, 256, stride=2),
            conv_block(256, 1, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, gray_image, color_image):
        x = torch.concatenate([gray_image, color_image], 1)
        x = self.seq(x)
        return x


def gen_train_tick(gen: Generator, disc: Discriminator, loss_fn, gray_images):
    gray_shape = gray_images.shape
    z = torch.from_numpy(np.random.normal(0, 1, gray_shape)).float().to(device)
    out = gen(gray_images, z)
    dis_out = disc(gray_images, out)
    loss = loss_fn(dis_out, torch.ones_like(dis_out, device=device))
    return loss, out


def disc_train_tick(disc: Discriminator, gray_images, colored_images, gen_out, loss_fn):
    dis_out1 = disc(gray_images, colored_images)
    loss1 = loss_fn(
        dis_out1,
        torch.ones_like(dis_out1, device=device)
    )
    dis_out2 = disc(gray_images, gen_out)
    loss2 = loss_fn(
        dis_out2,
        torch.zeros_like(dis_out2, device=device)
    )
    loss = (loss1 + loss2)/2
    return loss


def color_and_gray(batch):
    from torchvision.transforms.functional import rgb_to_grayscale
    b, (c, w, h) = len(batch), batch[0][0].shape
    colored = torch.empty((b, c, w, h)).to(device)
    grayed = torch.empty((b, 1, w, h)).to(device)
    for i, img in enumerate(batch):
        img = img[0]
        gray = rgb_to_grayscale(img)
        colored[i] = img
        grayed[i] = gray

    return colored, grayed


def test():
    lr = 3e-4
    gray_shape, color_shape, z_shape = (
        1, 300, 300), (3, 300, 300), (1, 300, 300)
    generator = Generator(gray_shape, color_shape, z_shape, 4)
    discriminator = Discriminator(gray_shape, color_shape)
    color_images, gray_images = torch.randn(
        (1, *color_shape)), torch.randn((1, *gray_shape))

    loss_fn = nn.BCELoss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    gen_out, gen_loss, dis_loss = train_tick(
        (color_images, gray_images),
        (generator, gen_optimizer),
        (discriminator, dis_optimizer),
        loss_fn,
    )


def main(
    args,
    dataset,
    gen_model_path,
    dis_model_path,
    load=True
):
    from torch.utils.data import DataLoader
    from torchvision.transforms import Normalize
    from paper_impl.utils import get_dataset, save_images, load_or_create_model

    dataloader = DataLoader(
        get_dataset(dataset, transform=[Normalize(.5, .5)]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=color_and_gray
    )
    ds_name = dataloader.dataset.__class__.__name__

    image_shape = c, w, h = dataloader.dataset[0][0].shape
    gray_shape = 1, w, h

    generator = load_or_create_model(lambda: Generator(
        gray_shape, image_shape, gray_shape), gen_model_path, load).to(device)
    discriminator = load_or_create_model(lambda: Discriminator(
        gray_shape, image_shape), dis_model_path, load).to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

    loss_fn = nn.BCELoss()

    for epoch in range(args.num_epochs):
        for batch_i, (color_images, gray_images) in (bar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
            color_images, gray_images = color_images.to(
                device), gray_images.to(device)
            gen_out, gen_loss, dis_loss = train_tick(
                (color_images, gray_images),
                (generator, gen_optimizer),
                (discriminator, dis_optimizer),
                loss_fn
            )
            bar.set_description(
                f"Epoch:{epoch}, Gen loss: {gen_loss.item():.4f}, Disc loss: {dis_loss.item():.4f}")
        to_be_saved = gen_out.reshape(-1, *image_shape)
        save_images(
            to_be_saved,
            f"epoch_{epoch:04d}.jpg",
            f"PIX2PIX/{ds_name}",
        )
    return generator, discriminator


def train_tick(images, gen, dis, loss_fn):
    color_images, gray_images = images
    generator, gen_optimizer = gen
    discriminator, dis_optimizer = dis
    gen_optimizer.zero_grad()
    gen_loss, gen_out = gen_train_tick(
        generator,
        discriminator,
        loss_fn,
        gray_images
    )

    gen_loss.backward()
    gen_optimizer.step()

    dis_optimizer.zero_grad()
    dis_loss = disc_train_tick(
        discriminator,
        gray_images,
        color_images,
        gen_out.detach(),
        loss_fn,
    )
    dis_loss.backward()
    dis_optimizer.step()

    return gen_out, gen_loss, dis_loss


if __name__ == "__main__":
    # test()
    import os
    import sys
    from torchvision.datasets import CIFAR10
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

    from paper_impl.utils import arg_parser, save_model
    args = arg_parser(
        lr=3e-4,
        num_epochs=1,
        batch_size=64*2,
    )
    gen_model_path = "pix2pix/generator.pt"
    dis_model_path = "pix2pix/discriminator.pt"
    generator, discriminator = main(
        args,
        CIFAR10,
        gen_model_path,
        dis_model_path,
        load=True
    )
    save_model(generator, gen_model_path)
    save_model(discriminator, dis_model_path)
