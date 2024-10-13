import numpy as np
from torch.utils.data import IterableDataset


class IterableImageDataset(IterableDataset):
    """
    Cada vez que iteramos por el dataset, se aplican las transformaciones por la imagen
    """

    def __init__(
        self,
        data: IterableDataset,
        images_colname: str,
        id_colname: str,
        label_colname: str,
    ):
        self.data = data
        self.images_colname = images_colname
        self.id_colname = id_colname
        self.label_colname = label_colname

    def __iter__(self):
        for example in self.data:
            # Sample a single image from the list of images
            image = example[self.images_colname][
                np.random.randint(0, len(example[self.images_colname]))
            ]
            yield example[self.id_colname], example[self.label_colname], image
