from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable
from ema_pytorch import EMA

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 device: torch.device = torch.device("cpu")):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.995)                                                                           
        self.device = device
        self.ema = EMA(self.model).to(self.device)
        
        alpha_bars = self.model.sampler.alpha_bars
        snr = alpha_bars / (1 - alpha_bars)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max = 5)
        self.loss_weight_ = maybe_clipped_snr / snr
        
    def train(self, dl : DataLoader, epochs : int, file_name : str, no_label : bool = False):
        self.model.train()
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            for _, batch in enumerate(tqdm(dl, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500")):
                if no_label: 
                    if type(batch) == list:
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                else: x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                if no_label == True:
                    loss = self.loss_fn(x)
                else:
                    loss = self.loss_fn(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update()
                epoch_loss += loss.item()
                
            self.scheduler.step()
            log_string = f"Loss at epoch {epoch}: {epoch_loss / len(dl) :.3f}"
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save({
                    #"model_state_dict": self.model.state_dict(),
                    "ema_model_state_dict": self.ema.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "batch_size": dl.batch_size,
                    "number_of_batches": len(dl)
                    }, 'ig/check_points/' + file_name + '_epoch' + str(epoch) + '.pth')
                log_string += " --> Best model ever (stored)"
            print(log_string)
            
    def accelerated_train(self, dl : DataLoader, epochs : int, file_name : str, no_label : bool = False):
        self.model.train()
        best_loss = float("inf")
        accelerator = Accelerator(mixed_precision = 'no')
        model, optimizer, data_loader, scheduler = accelerator.prepare(
            self.model, self.optimizer, dl, self.scheduler
            )
        ema = EMA(model).to(accelerator.device)
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not accelerator.is_local_main_process)    
            for batch in progress_bar:
                if no_label: 
                    if type(batch) == list:
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                else: x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                if no_label == True:
                    loss = self.loss_fn(x)
                else:
                    loss = self.loss_fn(x, y)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                ema.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            scheduler.step()
            log_string = f"Loss at epoch {epoch}: {epoch_loss / len(progress_bar):.3f}"
            if accelerator.is_main_process:
                if best_loss > epoch_loss:
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_loss = epoch_loss
                    torch.save({
                        "model_state_dict": unwrapped_model.state_dict(),
                        "ema_model_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "batch_size": dl.batch_size,
                        "number_of_batches": len(dl)
                        }, file_name + '_epoch' + str(epoch) + '.pth')
                    log_string += " --> Best model ever (stored)"
                print(log_string)
            accelerator.wait_for_everyone()