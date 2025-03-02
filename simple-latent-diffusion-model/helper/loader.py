import torch
import torch.nn as nn
from helper.ema import EMA

class Loader():
    def __init__(self, device = None):
        self.device = device
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Training step: " + str(check_point["training_step"]))
        print("Best loss: " + str(check_point["best_loss"]))
        print("Batch size: " + str(check_point["batch_size"]))
        print("Number of batches: " + str(check_point["number_of_batches"]))
        
    def model_load(self, file_name : str, model : nn.Module, 
             print_dict : bool = True, is_ema: bool = True):
        check_point = torch.load(file_name + ".pth", map_location=self.device,
                                 weights_only=True)
        if print_dict: self.print_model(check_point)
        if is_ema:
            model = EMA(model)
            model.load_state_dict(check_point['ema_state_dict'])
            model = model.ema_model
        else:
            model.load_state_dict(check_point['model_state_dict'])
        model.eval()
        print("===Model loaded!===")
        return model
        
    def load_for_training(self, file_name: str, model: nn.Module, print_dict: bool = True):
        check_point = torch.load(file_name + ".pth", map_location=self.device)
        if print_dict: self.print_model(check_point)
        model = EMA(model)
        model.load_state_dict(check_point['ema_state_dict'])
        model = model.ema_model
        for param in model.parameters():
            param.requires_grad_(True)
        model.train()
        epoch = check_point["epoch"]
        loss = check_point["best_loss"]
        print("===Model/Epoch/Loss loaded!===")
        return model, epoch, loss