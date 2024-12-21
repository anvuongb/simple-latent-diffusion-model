from PIL import Image as im
import matplotlib.pyplot as plt
import torch
import numpy as np

class Painter(object):
    def __init__(self) :
        pass

    def show_images(self, images, title : str = '', index : bool = False, cmap = None):
        images = images.permute(0, 2, 3, 1) 
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        images = np.clip(images / 2 + 0.5, 0, 1)
        
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        idx = 0
        for _ in range(rows):
            for _ in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx], cmap = cmap)
                    if index :
                        plt.title(idx + 1)
                    plt.axis('off')
                    idx += 1
        fig.suptitle(title, fontsize=30)
        plt.show()

    def show_first_batch(self, loader):
        for batch in loader:
            self.show_or_save_images(images = batch, title = "Images in the first batch")
            break