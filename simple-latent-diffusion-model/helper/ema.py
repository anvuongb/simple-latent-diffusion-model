import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        """
        Implements Exponential Moving Average (EMA) of model parameters.

        Args:
            model: The model to apply EMA to.
            decay: The decay rate (beta).
            device: Device to use ('cpu' or 'cuda').  If None, uses the model's device.
        """
        super().__init__()
        self.model = model
        self.decay = decay
        self.device = device  # Store the device

        if self.device is not None:
            self.model.to(self.device)

        # Create a copy of the model parameters for EMA
        self.shadow_params = [p.clone().detach() for p in model.parameters()]

        # for inference
        self.collected_params = []

    def update(self):
        """
        Updates the EMA parameters.  Call this after optimizer.step().
        """
        for s_param, param in zip(self.shadow_params, self.model.parameters()):
            if self.device is not None:
                param = param.to(self.device)
            s_param.data.copy_(self.decay * s_param.data + (1. - self.decay) * param.data) #EMA update

    def copy_to(self, model):
        """Copies the EMA parameter to model"""
        for s_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(s_param.data)

    def forward(self, *args, **kwargs):
        """Use EMA parameters for inference"""
        self.collected_params = [param.clone() for param in self.model.parameters()]
        self.copy_to(self.model)
        output = self.model(*args, **kwargs)
        for i, collected_param in enumerate(self.collected_params):
            self.model.parameters()[i].data.copy_(collected_param.data) #copy back the weight
        return output