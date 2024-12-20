import torch
import torch.nn as nn
from accelerate import Accelerator
from ema_pytorch import EMA

class Loader2():
    def __init__(self, device = None):
        self.accelerator = Accelerator()
        self.device = device
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Best loss: " + str(check_point["best_loss"]))
        print("Batch size: " + str(check_point["batch_size"]))
        print("Number of batches: " + str(check_point["number_of_batches"]))
        print("Optimizer: " + str(check_point["optimizer_state_dict"]["param_groups"]))
        
    def load(self, file_name : str, model : nn.Module, print_dict : bool = True, ema : bool = False):
        check_point = torch.load(file_name + ".pth", map_location=self.accelerator.device)
        if print_dict: self.print_model(check_point)
        model = EMA(model)
        model.load_state_dict(check_point["ema_model_state_dict"]) 
        if ema == False:
            model = model.ema_model
        model.eval()
        print("===Model loaded!===")
        return model