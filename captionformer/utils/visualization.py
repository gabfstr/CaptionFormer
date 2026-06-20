# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import torch
from torchvision.utils import draw_segmentation_masks 
import cv2

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import seaborn as sns



def _nop(arg):
    return arg


class VisualizationMethod(ABC):
    """Abstract base class of a visualization method."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Comput visualization output.

        A visualization method takes some inputs and returns a Visualization.
        """
        pass


class Segmentation(VisualizationMethod):
    """Segmentaiton visualization."""

    def __init__(
        self,
        n_instances: int = 8,
        denormalization: Optional[tuple] = None,
        bgr_to_rgb: Optional[bool] = False,
        to_onehot: Optional[bool] = False,
    ):
        """Initialize segmentation visualization.

        Args:
            n_instances: Number of masks to visualize
            denormalization: Function to map from normalized inputs to unnormalized values
        """
        self.n_instances = n_instances
        self.denormalization = denormalization
        self.bgr_to_rgb = bgr_to_rgb
        self.to_onehot = to_onehot
        self._cmap_cache: Dict[int, List[Tuple[int, int, int]]] = {}

    def _get_cmap(self, num_classes: int) -> List[Tuple[int, int, int]]:
        if num_classes in self._cmap_cache:
            return self._cmap_cache[num_classes]

        #Put red if only one class
        if num_classes == 1:
            mpl_cmap = plt.get_cmap("autumn", num_classes)(range(num_classes))
        elif num_classes <= 20:
            mpl_cmap = plt.get_cmap("tab20", num_classes)(range(num_classes))
        else:
            #mpl_cmap = cm.get_cmap("turbo", num_classes)(range(num_classes))
            mpl_cmap = plt.get_cmap("turbo", num_classes)(range(num_classes))

        cmap = [tuple((255 * cl[:3]).astype(int)) for cl in mpl_cmap]
        self._cmap_cache[num_classes] = cmap
        return cmap

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor, plot_name: str, metrics: Optional[Dict[str, float]] = None, contours: Optional[bool] = False,
    ) -> Optional[torch.Tensor]:
        """Visualize segmentation overlaying original image.

        Args:
            image: Image to overlay
            mask: Masks of individual objects
        """
        if self.n_instances != -1:
            image = image[: self.n_instances].cpu()
            mask = mask[: self.n_instances].cpu().contiguous()

        if mask.shape[0] == 0:
            print("No masks to visualize")
            return None


        assert image.dim() == 4 # Only support image data at the moment.
        assert mask.dim() == 4 # Only support mask data at the moment.

        if image.dtype in [torch.long, torch.int64, torch.int32, torch.int16, torch.int8]:
            image = image.float() / 255.0
        
        if self.denormalization:
            pixel_mean = self.denormalization[0]
            pixel_std = self.denormalization[1]
            image = ((image) * pixel_std.view(-1, 1, 1) + pixel_mean.view(-1, 1, 1)).long()

        if image.dtype in [torch.long, torch.int64, torch.int32, torch.int16, torch.int8]:
            image = image.float() / 255.0

        if self.bgr_to_rgb:
            image = torch.tensor(cv2.cvtColor(image[0].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            image = image.unsqueeze(0)

        if image.shape[-2:] != mask.shape[-2:]:
            from torch.nn import functional as F
            image = F.interpolate(image, size=mask.shape[-2:], mode='bilinear', align_corners=False)

        if image.dtype == torch.float:
            image = (image * 255).to(torch.uint8)

        n_objects = mask.shape[1]
        if self.to_onehot:
            masks_argmax = mask.argmax(dim=1)[:, None]
            classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)
            masks_one_hot = masks_argmax == classes
        else:
            masks_one_hot = mask == 1

        cmap = self._get_cmap(n_objects)
        masks_on_image = torch.stack(
            [
                draw_segmentation_masks(
                    img, mask, alpha=0.4, colors=cmap
                )
                for img, mask in zip(image.to("cpu"), masks_one_hot.to("cpu"))
            ]
        )

        for img, mask_instance in zip(masks_on_image, masks_one_hot):
            if contours:
                img_np = np.ascontiguousarray(img.permute(1, 2, 0).numpy(), dtype=np.uint8)
                for mask_np in mask_instance:
                    mask_np = (mask_np.numpy() * 255).astype(np.uint8)  # Ensure single-channel, uint8 format
                    contours_list, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_np, contours_list, -1, (0, 255, 0), 2)
                img = torch.tensor(img_np).permute(2, 0, 1)

            if metrics is not None:
                # Add a small vertical white band to the right of the image
                if img.shape[1] < 256:
                    # Width of the white band in pixels
                    white_band_width = 90
                    # Font for metrics
                    font=22
                elif img.shape[1] < 320:
                    white_band_width = 150
                    font=23
                elif img.shape[1] < 640:
                    white_band_width = 250  
                    font=28
                else:
                    white_band_width = 350
                    font=32
                white_band=torch.ones_like(img[:,:,:white_band_width])*255
                img = torch.cat([img,white_band ], dim=2)

                # Create a string representation of the metrics
                metrics_str = '\n'.join([f"{key}: {value}" for key, value in metrics.items()])
                # add '----------------' after the 3rd line and 5th line:
                metrics_str = '\n'.join([metrics_str.split('\n')[0], metrics_str.split('\n')[1], metrics_str.split('\n')[2], 
                                            '----------------', 
                                            metrics_str.split('\n')[3], metrics_str.split('\n')[4], metrics_str.split('\n')[5],
                                            '----------------', 
                                            metrics_str.split('\n')[6], metrics_str.split('\n')[7]])
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(img.permute(1, 2, 0))
            if metrics is not None:
                plt.text(img.shape[-1] * (7 / 10), (img.shape[-2] * 1) // 20, metrics_str, fontsize=font, color='black', ha='left', va='top')
            fig.savefig(plot_name)
            plt.close(fig)

        return masks_on_image


    def plot_bbox(
        self, image: torch.Tensor, boxes: torch.Tensor, plot_name: str, metrics: Optional[Dict[str, float]] = None, box_mode: str = "xywh"
    ) -> Optional[torch.Tensor]:
        """Visualize bounding boxes overlaying original image.

        Args:
            image: Image to overlay
            boxes: Bounding boxes of individual objects in xyxy format
        """
        image = image.cpu()
        boxes = boxes.cpu()

        assert box_mode in ["xyxy", "xywh"]; "Only support xyxy or xywh box format"

        # Check that boxes is not empty
        if boxes.shape[0] == 0:
            print("No boxes to visualize")
            return None

        assert image.dim() == 4 # Only support image data at the moment.
        assert boxes.dim() == 2

        if self.denormalization != None:
            pixel_mean = self.denormalization[0]*255.0
            pixel_std = self.denormalization[1]*255.0 
            image = ((image)*pixel_std.view(-1,1,1) + pixel_mean.view(-1,1,1)).long()
        
         # if image is of type long
        if image.dtype == torch.long:
            image = image.float()
            image = image / 255.0
        if self.bgr_to_rgb:
            image=torch.tensor(cv2.cvtColor(image[0].permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)).permute(2,0,1)
            image=image.unsqueeze(0)
        

         # if image is of type float
        if image.dtype == torch.float:
            image = (image * 255).to(torch.uint8)

        n_objects = boxes.shape[0]

        bb_color=(0,255,0)
        
        image_np = image[0].permute(1,2,0).numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(image_np)

        for i in range(n_objects):
            if box_mode == "xywh":
                x0, y0, w, h = boxes[i]
                x0, y0, w, h = int(x0), int(y0), int(w), int(h)
                rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='r', facecolor='none')
            else : 
                x0, y0, x1, y1 = boxes[i]
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.axis('off')
        plt.savefig(plot_name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        return torch.from_numpy(image_np).permute(2, 0, 1)


        


    def plot_metrics_dev(
        self, image: torch.Tensor, mask: torch.Tensor, plot_name: str, metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        """Visualize segmentation overlaying original image with multiple metrics  (dev set).

        Args:
            image: Image to overlay
            mask: Masks of individual objects
        """
        image = image[: self.n_instances].cpu()
        mask = mask[: self.n_instances].cpu().contiguous()

        # Check that mask is not empty
        if mask.shape[0] == 0:
            print("No masks to visualize")
            return None


        assert image.dim() == 4 # Only support image data at the moment.
        assert mask.dim() == 4 # Only support mask data at the moment.


        if self.denormalization != None:
            pixel_mean = self.denormalization[0]*255.0
            pixel_std = self.denormalization[1]*255.0 
            image = ((image)*pixel_std.view(-1,1,1) + pixel_mean.view(-1,1,1)).long()
        
         # if image is of type long
        if image.dtype == torch.long:
            image = image.float()
            image = image / 255.0
        if self.bgr_to_rgb:
            image=torch.tensor(cv2.cvtColor(image[0].permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)).permute(2,0,1)
            image=image.unsqueeze(0)
        
        if image.shape[-2:]!=mask.shape[-2:]:
            from torch.nn import functional as F
            image = F.interpolate(image, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        
         # if image is of type float
        if image.dtype == torch.float:
            image = (image * 255).to(torch.uint8)
        
        n_objects = mask.shape[1]

        if self.to_onehot:
            masks_argmax = mask.argmax(dim=1)[:, None]
            classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)
            masks_one_hot = masks_argmax == classes
        else:
            masks_one_hot = mask==1
        
        cmap = self._get_cmap(n_objects)
        masks_on_image = torch.stack(
            [
                draw_segmentation_masks(
                    img, mask, alpha=0.4, colors=cmap
                )
                for img, mask in zip(image.to("cpu"), masks_one_hot.to("cpu"))
            ]
        )

        for img in masks_on_image:
            
            
            if metrics is not None:
                # Add a small vertical white band to the right of the image
                # if img.shape[1] < 256:
                #     # Width of the white band in pixels
                #     white_band_width = 90*4
                #     # Font for metrics
                #     font=22
                # elif img.shape[1] < 320:
                #     white_band_width = 150*4
                #     font=23
                # elif img.shape[1] < 640:
                #     white_band_width = 250*4  
                #     font=28
                # else:
                #     white_band_width = 350*4
                #     font=32
                white_band_width = img.shape[1]*3 + 50
                font = 16
                white_band=torch.ones_like(img[:,:,:white_band_width])*255
                img = torch.cat([img,white_band ], dim=2)

                # Create a string representation of the metrics
                metrics_keys = list(metrics.keys())
                metrics_values = list(metrics.values())
                
                
                # Split the metrics into two columns
                metrics_str1 = '\n'.join([f"{key}: {value}" for key, value in zip(metrics_keys[:8], metrics_values[:8])])
                metrics_str2 = '\n'.join([f"{key}: {value}" for key, value in zip(metrics_keys[8:], metrics_values[8:])])

                # Add '----------------' after the 3rd line and 5th line in each column:
                metrics_str1 = '\n'.join([metrics_str1.split('\n')[0], metrics_str1.split('\n')[1], metrics_str1.split('\n')[2], 
                                        '----------------', 
                                        metrics_str1.split('\n')[3], metrics_str1.split('\n')[4], metrics_str1.split('\n')[5],
                                        '----------------', 
                                        metrics_str1.split('\n')[6], metrics_str1.split('\n')[7]])
                metrics_str2 = '\n'.join([metrics_str2.split('\n')[0], metrics_str2.split('\n')[1], metrics_str2.split('\n')[2], 
                                        '----------------', 
                                        metrics_str2.split('\n')[3], metrics_str2.split('\n')[4], metrics_str2.split('\n')[5]])
                
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            ax.imshow(img.permute(1,2,0))
            if metrics is not None:
                plt.text(img.shape[1]+3, (img.shape[1]*1)//20, metrics_str1, fontsize=font, color='black', ha='left', va='top')
                plt.text(img.shape[1]+img.shape[1]//2, (img.shape[1]*1)//20, metrics_str2, fontsize=font, color='black', ha='left', va='top')
            fig.savefig(plot_name)
            plt.close(fig)
        
        return masks_on_image
    

    def plot_image(
            self, image: torch.Tensor, plot_name: str
    )-> Optional[torch.Tensor]:
        
        image = image.cpu()
        
        if image.dim() == 4:
            image = image[0]
        assert image.dim() == 3 ; "Only support image data at the moment."
        if self.denormalization != None:
            pixel_mean = self.denormalization[0]*255.0
            pixel_std = self.denormalization[1]*255.0 
            image = ((image)*pixel_std.view(-1,1,1) + pixel_mean.view(-1,1,1)).long()
         # if image is of type long
        if image.dtype == torch.long:
            image = image.float()
            image = image / 255.0
        if self.bgr_to_rgb:
            image=torch.tensor(cv2.cvtColor(image.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))
            image=image.permute(2,0,1)

         # if image is of type float
        if image.dtype == torch.float:
            image = (image * 255).to(torch.uint8)
    
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(image.permute(1,2,0))
        fig.savefig(plot_name)
        plt.close(fig)

        return
    
    def slot_heatmap(
            self, image: torch.Tensor, masks: torch.Tensor, plot_dir: str,
        ) -> Optional[torch.Tensor]:

        image = image.cpu()
        
        if image.dim() == 4:
            image = image[0]
        assert image.dim() == 3 ; "Only support image data at the moment."
        if self.denormalization != None:
            pixel_mean = self.denormalization[0]*255.0
            pixel_std = self.denormalization[1]*255.0 
            image = ((image)*pixel_std.view(-1,1,1) + pixel_mean.view(-1,1,1)).long()
         # if image is of type long
        if image.dtype == torch.long:
            image = image.float()
            image = image / 255.0
        image=image.permute(1,2,0)
        if self.bgr_to_rgb:
            image=torch.tensor(cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2RGB))

        if image.shape[:2]!=masks.shape[:2]:
            from torch.nn import functional as F
            # image = F.interpolate(image, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            masks = F.interpolate(masks.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=False).squeeze(0)

         # if image is of type float
        if image.dtype == torch.float:
            image = (image * 255).to(torch.uint8)

        image=image.cpu().numpy()

        assert masks.dim() == 3 # Only support mask data at the moment.
        assert masks.max() <= 1.0 # Only support mask data at the moment.
        assert masks.min() >= 0.0 # Only support mask data at the moment.
        masks = masks.detach().cpu().numpy()

        # Number of heatmaps to overlay (K)
        K = len(masks)
        # Convert back to RGB format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("image shape", image.shape)
        print("masks shape", masks.shape)
        # Create a custom color map (blue to red)
        heatmap_color_map = plt.get_cmap('coolwarm')

        # Iterate through the first K rows of "pixel"
        for k in range(K):
            # Apply the color map to the k-th row of "pixel"
            heatmap = heatmap_color_map(masks[k])

            # Convert the heatmap to BGR format
            heatmap_bgr = (heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Define the transparency level for the heatmap
            alpha = 0.5  # Adjust this value as needed
            # Superimpose the k-th heatmap on the image
            result = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
            
            
            # Display the superimposed image
            plt.figure()
            plt.imsave(plot_dir+'heatmap_{}.png'.format(k), result)
            plt.axis('off')  # Optional: Turn off axis labels
            plt.close()
            
        return

    def self_attention(self, attention: torch.Tensor, plot_dir: str) -> None:
        """
        This function visualizes self-attention weights using a heatmap.

        Parameters:
        attention (torch.Tensor): The self-attention weights tensor of shape (n_seq, n_seq).
        plot_dir (str): The directory where the plot will be saved.
        """
        
        assert attention.dim() == 2, 'The attention tensor should have 3 dimensions: (n_seq, n_seq).'
        attention = attention.cpu().detach().numpy()
        plt.figure()

        sns.heatmap(attention, cmap='Blues')
        plt.title('Self-attention heatmap')
        # Save the figure
        plt.savefig(f'{plot_dir}/self_attention_heatmap.png')
        plt.close()
        
        return