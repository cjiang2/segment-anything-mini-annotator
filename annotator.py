from typing import List
import os
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import torch
import torchvision

from segment_anything import sam_model_registry, SamPredictor

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

class Annotator:
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str,
        device: str = "cuda",
        out_path: str = "out/",
        vis_path: str = "vis/",
        show_points_in_vis: bool = True,
        ):
        self.out_path = out_path
        create_folder(out_path)
        self.vis_path = vis_path
        if vis_path is not None:
            create_folder(vis_path)
        self.show_points_in_vis = show_points_in_vis
        self.done = False

        # Load Segment-anything
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.label_mode = 1

    def annotate_image(
        self,
        img_path: str, 
        ):
        self.done = False
        self.img = cv2.imread(img_path)
        
        # Setup GUI
        self.coords = []
        self.labels = []
        self.fig = plt.figure()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)

        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.show()

        # Figure exit success
        if self.done:
            img_filename = img_path.split(os.sep)[-1]

            # Segment-anything
            self.predictor.set_image(self.img)
            masks, _, _ = self.predictor.predict(
                point_coords=np.array(self.coords),
                point_labels=np.array(self.labels),
                multimask_output=False,
                )

            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            show_mask(masks[0], plt.gca())
            # Mask-only visualization
            if self.vis_path is not None:
                plt.savefig(os.path.join(self.vis_path, img_filename), dpi=300)

            # Also visualize the clicks
            if self.show_points_in_vis:
                show_points(np.array(self.coords), np.array(self.labels), plt.gca())
                if self.vis_path is not None:
                    plt.savefig(os.path.join(self.vis_path, "with_points_" + img_filename), dpi=300)

            # Save mask
            plt.show()
            mask = 255 * masks.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(self.out_path, img_filename), mask)

            

    def onclick(self, event):
        x, y = int(event.xdata), int(event.ydata)

        # Left mouse click saves coordinates
        if event.button is MouseButton.LEFT:
            self.coords.append([x, y])
            self.labels.append(self.label_mode)

            print(self.coords, self.labels)
            show_points(np.array(self.coords), np.array(self.labels), plt.gca())
            self.fig.canvas.draw()

        # While right mouse click clears last point
        elif event.button is MouseButton.RIGHT:
            self.coords.pop()
            self.labels.pop()

            # Refresh
            plt.clf()
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            if len(self.coords) > 0:
                show_points(np.array(self.coords), np.array(self.labels), plt.gca())
            self.fig.canvas.draw()


    def onkey(self, event):
        # Press "c" to reset
        if event.key == 'c':
            self.coords = []
            self.labels = []

            # Refresh
            plt.clf()
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            self.fig.canvas.draw()

        # Done current image, go next
        elif event.key == "enter":
            self.done = True
            self.fig.canvas.mpl_disconnect(self.cid_key)
            self.fig.canvas.mpl_disconnect(self.cid_click)
            plt.close()

        # SAM
        # Press "x" to switch between 1 - foreground, and 0 - background
        elif event.key == "x":
            if self.label_mode == 1:
                self.label_mode = 0
                print("Switched to background mode.")
            else:
                self.label_mode = 1
                print("Switched to foreground mode.")

    def run(
        self,
        img_paths: List[str],
        ):
        for img_path in img_paths:
            self.annotate_image(img_path)


# Grab all images to annotate
workspace_path = "imgs/"
img_paths = glob.glob(os.path.join(workspace_path, "*.png"))

# Load SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# Start the annotator
annotator = Annotator(sam_checkpoint, model_type, device)
annotator.run(img_paths)
