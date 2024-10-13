import numpy as np
from torch.utils.data import Dataset


class SemiIterableImageDataset(Dataset):
    """
    Este dataset es un "tramposo" porque haces get_item, pero realmente estas iterando de un dataset en streaming.
    Como tenemos un parquet con la longitud del dataset (numero de productos), podemos
    """

    def __init__(
        self,
        data,
        data_length,
        images_colname,
        id_colname,
        label_colname,
    ):
        self.data_iterator = iter(data)
        self.data_length = data_length
        self.images_colname = images_colname
        self.id_colname = id_colname
        self.label_colname = label_colname

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        example = next(self.data_iterator)
        # Sample a single image from the list of images
        image = example[self.images_colname][
            np.random.randint(0, len(example[self.images_colname]))
        ]
        return example[self.id_colname], example[self.label_colname], image
