from PIL import Image as im
import matplotlib.pyplot as plt
from tqdm import tqdm
import io
import torch
import numpy as np

class Painter():
    def __init__(self):
        pass
    
    def show_or_save_images(self, images, show : bool = True, title : str = '', save : bool = False, 
                            file_name : str = 'result', index : bool = False, cmap = None):
        """Shows the provided images as sub-pictures in a square"""
        """
        This function takes a set of images as arguments and displays them as appropriate sub-pictures.
        show --> If True, it displays the images; if False, it does not.
        save --> If True, it saves the images.
        file_name --> The file name used when saving the images.
        index --> If True, a number appears below each sub-picture.
        cmap --> Specify 'gray' for 1-dimensional images.
        """

        # Converting images to CPU numpy arrays
        images = images.permute(0, 2, 3, 1) # Orginial shape (bs, ch, w, h) --> (bs, w, h, ch)
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        images = np.clip(images / 2 + 0.5, 0, 1)
        
        # Defining number of rows and columns
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

        # Populating figure with sub-plots
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)

                if idx < len(images):
                    plt.imshow(images[idx], cmap = cmap)
                    if index :
                        plt.title(idx + 1)
                    plt.axis('off')
                    idx += 1
        fig.suptitle(title, fontsize=30)

        # Save the figure
        if save :
            plt.savefig(file_name + '.png')
        # Showing the figure
        if show :
            plt.show()
        