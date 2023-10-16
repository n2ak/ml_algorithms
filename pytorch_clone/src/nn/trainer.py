
from src._base import Optimizer, Loss
from src.nn import Module
from src.grad import grad_off
import numpy as np


class Dataset:
    def __init__(self, X, y) -> None:
        self.inner = X, y

    def next_batch(self):
        raise NotImplementedError()


class Trainer:
    def __init__(
        self,
        model: Module,
        lossfn: Loss,
        optimizer: Optimizer,
        metrics=[]
    ) -> None:
        self.model = model
        self.loss_fn = lossfn
        self.optimizer = optimizer
        self.metrics = [
            TrainingMetricResult("loss"),
            TrainingMetricResult("val_loss"),
        ]

        for m in metrics:
            self.metrics.append(TrainingMetricResult(m))

    def train(self, epochs, dataset: Dataset, val_dataset: Dataset):
        from tqdm import tqdm
        for epoch in range(epochs):
            bar = tqdm()
            results = []
            for batch_idx, batch in dataset:
                self.optimizer.zero_grad()
                loss, res = self.train_batch(batch)
                loss.backward()
                self.optimizer.step()

                results.append(res)
                bar.set_description(
                    f'Batch: {batch_idx}' + ', '.join(res.items())
                )
            self.update_metrics(results, True)

            if val_dataset is not None:
                with grad_off():
                    self.update_metrics(self.evaluate(val_dataset))

    def infer(self, X, y):
        pred_y = self.model(X)
        loss = self.loss_fn(pred_y, y)
        res = {"loss": loss}
        if "accuracy" in self._metrics:
            acc = self._calc_acc(pred_y, y)
            res["acc"] = acc
        return loss, res

    def evaluate(self, val_dataset: Dataset):
        self.model.evaluation_mode()
        X, y = val_dataset.inner
        _, res = self.infer(X, y)
        return res

    def update_metrics(self, means: dict[str, any], frombatches=False):
        if frombatches:
            names = means[0].keys()
            values = [m.values() for m in means]
            m = np.mean(means, axis=1)
            means = {k: v for k, v in zip(names, values)}
        for m in self.metrics:
            if m.name in means.keys():
                m.update(means[m.name])

    def train_batch(self, batch):
        X, y = batch
        self.model.training_mode()
        return self.infer(X, y)

    def plot_results(self):
        for m in self.metrics:
            m.plot()


class TrainingMetricResult:
    def __init__(self, name) -> None:
        self.name = name
        self.res = []

    def update(self, val):
        self.res.append(val)

    def plot(self):
        raise NotImplementedError()
