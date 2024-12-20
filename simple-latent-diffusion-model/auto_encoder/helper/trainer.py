import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm
from accelerate import Accelerator

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 device: torch.device = torch.device("cpu"),
                 no_label : bool = True):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.995)                                                               
        self.loss_fn = loss_fn
        self.device = device
        self.no_label = no_label
         
    def accelerated_train(self, dl : DataLoader, epochs : int, file_name : str):
        self.model.train()
        accelerator = Accelerator(mixed_precision = 'no') 
        model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        self.model, self.optimizer, dl, self.scheduler)
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(training_dataloader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not accelerator.is_local_main_process)
            
            for batch in progress_bar:
                
                if self.no_label: 
                    if type(batch) == list:
                        x = batch[0].to(accelerator.device)
                    else:
                        x = batch.to(accelerator.device)
                else: x, y = batch[0].to(accelerator.device), batch[1].to(accelerator.device)
                
                loss = self.loss_fn(x)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / len(progress_bar))
                
            scheduler.step()
            log_string = f"Loss at epoch {epoch}: {epoch_loss / len(dl):.3f}"
            if accelerator.is_main_process:
                if best_loss > epoch_loss:
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_loss = epoch_loss
                    accelerator.save({
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "batch_size": dl.batch_size,
                        "number_of_batches": len(dl),
                        }, file_name + "_epoch" + str(epoch) + '.pth')
                    log_string += " --> Best model ever (stored)"
                print(log_string)
            accelerator.wait_for_everyone()
        