# Copyright Gabriel Fiastre — CaptionFormer (https://github.com/gabfstr/CaptionFormer)

import torch

from detectron2.structures import Instances

class Captions:
    """
    A lightweight class to store captions for instances.
    """

    def __init__(self, captions=None):
        """
        Args:
            captions (list[str], optional): A list of captions, one for each instance.
                Defaults to None.
        """
        if captions is None:
            self.captions = []
        else:
            if not isinstance(captions, list) or not all(isinstance(cap, (str, type(None))) for cap in captions):
                raise ValueError("Captions must be a list of strings.")
            self.captions = captions

    def set(self, captions):
        """
        Sets the captions for all instances.

        Args:
            captions (list[str]): A list of captions, one for each instance.
        """
        if not isinstance(captions, list) or not all(isinstance(cap, str) for cap in captions):
            raise ValueError("Captions must be a list of strings.")
        self.captions = captions

    def get(self):
        """
        Returns the list of captions.

        Returns:
            list[str]: A list of captions, one for each instance.
        """
        return self.captions

    def __len__(self):
        """
        Returns the number of captions (number of instances).

        Returns:
            int: The number of captions.
        """
        return len(self.captions)

    def __getitem__(self, item):
        """
        Retrieves captions for a single instance, slice of instances, or boolean mask.

        Args:
            item (int, slice, torch.Tensor): Index, slice, or boolean tensor to retrieve captions.

        Returns:
            str or list[str]: A single caption (for integer indexing) or a list of captions
                              (for slice or boolean indexing).
        """
        if isinstance(item, int) or isinstance(item, slice):
            return self.captions[item]
        elif isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return [cap for cap, keep in zip(self.captions, item.tolist()) if keep]
        else:
            raise TypeError("Invalid index type. Must be int, slice, or boolean tensor.")
    
    def __str__(self):
        """
        Returns a string representation of the captions.

        Returns:
            str: A string representation of the captions.
        """
        str="["
        for i,cap in enumerate(self.captions):
            str+="'"+cap+"'"
            if i<len(self.captions)-1:
                str+=", "
        str+="]"
        return str

    def __repr__(self):
        """
        Returns a detailed string representation of the captions.

        Returns:
            str: A detailed string representation of the captions.
        """
        return "Captions object of lentgh {} : {}".format(len(self.captions), self.captions)


class CaptionedInstances(Instances):
    def __init__(self, image_size, fields=None):
        super().__init__(image_size, fields=fields)
        self.captions = Captions()

    def set(self, field, value):
        if field == "captions":
            self.captions.set(value)
        else:
            super().set(field, value)

    def get(self, field):
        if field == "captions":
            return self.captions.get()
        else:
            return super().get(field)

    def __getitem__(self, item):
        ret = super().__getitem__(item)
        if isinstance(ret, torch.Tensor):
            return ret
        ret.captions = Captions(self.captions[item]) # Crucial change here
        return ret
    
    def to(self, *args, **kwargs):
        """Override the to method to handle captions"""
        ret = super().to(*args, **kwargs)
        ret.captions = Captions(self.captions.get())
        return ret