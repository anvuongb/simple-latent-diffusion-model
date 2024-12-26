import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from typing import Callable
from ema_pytorch import EMA

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 start_epoch = 0,
                 best_loss = float("inf")):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.998)
        self.accelerator = Accelerator(mixed_precision = 'no')
        self.start_epoch = start_epoch
        self.best_loss = best_loss
            
    def train(self, dl : DataLoader, epochs : int, file_name : str, no_label : bool = False):
        self.model.train()
        model, optimizer, data_loader, scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dl, self.scheduler
            )
        ema = EMA(model).to(self.accelerator.device)
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                if no_label: 
                    if type(batch) == list:
                        x = batch[0].to(self.accelerator.device)
                    else:
                        x = batch.to(self.accelerator.device)
                else: x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                
                if no_label == True:
                    loss = self.loss_fn(x)
                else:
                    loss = self.loss_fn(x, y)

                optimizer.zero_grad()
                self.accelerator.backward(loss)
                optimizer.step()
                ema.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            scheduler.step()
            epoch_loss = epoch_loss / len(progress_bar)
            log_string = f"Loss at epoch {epoch}: {epoch_loss :.4f}"
            if self.accelerator.is_main_process:
                if self.best_loss > epoch_loss:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    self.best_loss = epoch_loss
                    torch.save({
                        "model_state_dict": unwrapped_model.state_dict(),
                        "ema_model_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "training_step": epoch * len(dl),
                        "best_loss": self.best_loss,
                        "batch_size": dl.batch_size,
                        "number_of_batches": len(dl)
                        }, file_name + '_epoch' + str(epoch) + '.pth')
                    log_string += " --> Best model ever (stored)"
                print(log_string)
            self.accelerator.wait_for_everyone()