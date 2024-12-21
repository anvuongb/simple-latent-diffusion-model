import torch
import torch.nn as nn
from ema_pytorch import EMA

class Loader():
    def __init__(self, device = None):
        self.device = device
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Epoch: " + str(check_point["training_step"]))
        print("Best loss: " + str(check_point["best_loss"]))
        print("Batch size: " + str(check_point["batch_size"]))
        print("Number of batches: " + str(check_point["number_of_batches"]))
        
    def load(self, file_name : str, model : nn.Module, 
             print_dict : bool = True, ema : bool = False):
        check_point = torch.load(file_name + ".pth", map_location=self.device)
        if print_dict: self.print_model(check_point)
        if ema:
            model = EMA(model)
            model.load_state_dict(check_point["ema_model_state_dict"]) 
        else:
            model.load_state_dict(check_point["model_state_dict"]) 
        model.eval()
        print("===Model loaded!===")
        return model