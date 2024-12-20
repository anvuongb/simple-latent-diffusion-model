import torch
import torch.nn as nn
from accelerate import Accelerator

class Loader():
    def __init__(self):
        self.accelerator = Accelerator()
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Best loss: " + str(check_point["best_loss"]))
        print("Batch size: " + str(check_point["batch_size"]))
        print("Number of batches: " + str(check_point["number_of_batches"]))
        print("Optimizer: " + str(check_point["optimizer_state_dict"]["param_groups"]))
    
    def load_with_acc(self, file_name : str, model : nn.Module, print_dict : bool = True):
        unwrap_model = self.accelerator.unwrap_model(model)
        check_point = torch.load(file_name + ".pth", map_location = self.accelerator.device)
        if print_dict: self.print_model(check_point)
        unwrap_model.load_state_dict(check_point["model_state_dict"])
        unwrap_model.eval()
        return unwrap_model