import torch

class EarlyStop():
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = None
        self.stop_training = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.wait += 1
            print(f"Early stopping counter: {self.wait}/{self.patience}, best_loss: {self.best_loss}, crr_loss: {current_loss}")
            if self.wait >= self.patience:
                self.stop_training = True
        else:
            self.best_loss = current_loss
            self.wait = 0

