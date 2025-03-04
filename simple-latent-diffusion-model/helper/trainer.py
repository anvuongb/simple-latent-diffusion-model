import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from typing import Callable
from helper.ema import EMA

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 ema: EMA = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 start_epoch = 0,
                 best_loss = float("inf"),
                 accumulation_steps: int = 1):
        self.accelerator = Accelerator(mixed_precision = 'no')
        self.model = model 
        if ema is None:
            self.ema = EMA(self.model).to(self.accelerator.device)            
        else:
            self.ema = ema.to(self.accelerator.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, min_lr=1e-6
            )
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.accumulation_steps = accumulation_steps
            
    def train(self, dl : DataLoader, epochs : int, file_name : str, no_label : bool = False):
        self.model.train()
        self.model, self.optimizer, data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dl, self.scheduler
            )
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                self.optimizer.zero_grad()
                if no_label: 
                    if type(batch) == list:
                        x = batch[0].to(self.accelerator.device)
                    else:
                        x = batch.to(self.accelerator.device)
                else: x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                
                if no_label == True:
                    loss = self.loss_fn(x0 = x)
                else:
                    loss = self.loss_fn(x0 = x, y = y)

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.ema.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                epoch_loss = epoch_loss / len(progress_bar)
                self.scheduler.step(epoch_loss)
                log_string = f"Loss at epoch {epoch}: {epoch_loss :.4f}"
                if self.best_loss > epoch_loss:
                    self.best_loss = epoch_loss
                    torch.save({
                        "model_state_dict": self.accelerator.get_state_dict(self.model),
                        "ema_state_dict": self.ema.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "training_step": epoch * len(dl),
                        "best_loss": self.best_loss,
                        "batch_size": dl.batch_size,
                        "number_of_batches": len(dl)
                        }, file_name + '.pth')
                    log_string += " --> Best model ever (stored)"
                print(log_string)
