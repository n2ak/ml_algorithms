import torchvision.transforms as T


def get_dataset(
    Dataset,
    transform=[],
    **kwargs
):

    transform.insert(0, T.ToTensor())
    return Dataset(
        path_to("../datasets"),
        transform=T.Compose(transform),
        download=True,
        **kwargs,
    )


def path_to(p):
    import pathlib
    path = pathlib.Path(__file__).parent.joinpath(p).resolve()
    path.mkdir(exist_ok=True, parents=True)
    return path


def save_images(
    imgs,
    path,
    task,
    **kwargs
):
    import numpy as np
    from torchvision.utils import save_image
    save_image(
        imgs,
        path_to(f"../output/{task}").joinpath(path).resolve(),
        nrow=int(np.sqrt(imgs.size(0))),
        normalize=True,
        **kwargs
    )


def arg_parser(**kwargs):
    import argparse
    parser = argparse.ArgumentParser()
    for a, dv in kwargs.items():
        parser.add_argument(f"--{a}", default=dv, required=False)
    return parser.parse_args()
