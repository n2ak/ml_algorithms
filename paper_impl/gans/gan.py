import torch
import tqdm
import numpy as np
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)


class Generator(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.seq(x)


def gen_train_tick(gen: Generator, disc: Discriminator, input_shape, loss_fn):
    z = torch.from_numpy(np.random.normal(
        0, 1, input_shape)).float().to(device)
    out = gen(z)
    loss = loss_fn(
        disc(out),
        torch.ones(out.size(0), 1).to(device)
    )
    return loss, out


def disc_train_tick(disc: Discriminator, images, gen_out, loss_fn):
    loss1 = loss_fn(
        disc(images),
        torch.ones(images.size(0), 1).to(device)
    )
    loss2 = loss_fn(
        disc(gen_out),
        torch.zeros(images.size(0), 1).to(device)
    )
    loss = (loss1 + loss2)/2
    return loss


def main(
    args,
    dataset,
):
    from torch.utils.data import DataLoader
    from paper_impl.utils import get_dataset, save_images
    from torchvision.transforms import Normalize

    dataloader = DataLoader(
        get_dataset(dataset, transform=[Normalize(.5, .5)]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    ds_name = dataloader.dataset.__class__.__name__

    image_shape = dataloader.dataset[0][0].shape
    generator = Generator(np.prod(image_shape)).to(device)
    discriminator = Discriminator(np.prod(image_shape)).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

    loss_fn = nn.BCELoss()

    for epoch in range(args.num_epochs):
        for batch_i, (images, _) in (bar := tqdm.tqdm(enumerate(dataloader), total=len(dataloader))):
            image_shape = images.shape[1:]
            images = images.view(images.size(0), -1).to(device)

            gen_optimizer.zero_grad()
            gen_loss, gen_out = gen_train_tick(
                generator, discriminator, (images.shape), loss_fn)

            gen_loss.backward()
            gen_optimizer.step()

            dis_optimizer.zero_grad()
            dis_loss = disc_train_tick(
                discriminator, images, gen_out.detach(), loss_fn)
            dis_loss.backward()
            dis_optimizer.step()

            bar.set_description(
                f"Epoch:{epoch}, Gen loss: {gen_loss.item():.4f}, Disc loss: {dis_loss.item():.4f}")
        to_be_saved = gen_out.reshape(-1, *image_shape)
        save_images(
            to_be_saved,
            f"epoch_{epoch:04d}.jpg",
            f"GAN/{ds_name}",
        )


if __name__ == "__main__":
    import os
    import sys
    from torchvision.datasets import MNIST
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

    from paper_impl.utils import arg_parser
    args = arg_parser(
        lr=3e-4,
        num_epochs=1,
        batch_size=64*2,
    )
    main(
        args,
        MNIST
    )
