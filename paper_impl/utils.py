

import torchvision.transforms as T
PARENT_DIR = ".."
MODELS_PARENT_DIR = ".."


def path_relative_to_parent_dir(p, base):
    import pathlib
    path = pathlib.Path(__file__).parent.joinpath(base, p).resolve()
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_dataset(
    Dataset,
    transform=[],
    **kwargs
):

    transform.insert(0, T.ToTensor())
    return Dataset(
        path_relative_to_parent_dir("datasets", PARENT_DIR),
        transform=T.Compose(transform),
        download=True,
        **kwargs,
    )


def save_images(
    imgs,
    path,
    task,
    **kwargs
):
    import numpy as np
    from torchvision.utils import save_image
    path = path_relative_to_parent_dir(
        f"output/{task}", PARENT_DIR).joinpath(path).resolve()
    # print("Saving images to", path)
    save_image(
        imgs,
        path,
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


def save_model(model, path, base=MODELS_PARENT_DIR):
    import torch
    path = path_relative_to_parent_dir("models", base).joinpath(path).resolve()
    path.parent.mkdir(exist_ok=True, parents=True)
    print("Saved model to:", path)
    torch.save(model.state_dict(), path)


def load_or_create_model(create_fn, path, load=True, base=MODELS_PARENT_DIR):
    import torch
    import os
    model = create_fn()
    path = path_relative_to_parent_dir(
        "models", base).joinpath(path).resolve()
    if load and os.path.exists(path):
        print("Loading model from:", path)
        model.load_state_dict(torch.load(path))
    return model
