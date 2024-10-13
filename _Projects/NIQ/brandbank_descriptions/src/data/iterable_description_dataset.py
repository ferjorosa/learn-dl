import numpy as np
from torch.utils.data import IterableDataset


class IterableDescriptionDataset(IterableDataset):
    """
    Cada vez que iteramos por el dataset, se tokeniza la descripción seleccionada
    """

    def __init__(
        self,
        data: IterableDataset,
        descriptions_colname: str,
        id_colname: str,
        label_colname: str,
    ):
        self.data = data
        self.descriptions_colname = descriptions_colname
        self.id_colname = id_colname
        self.label_colname = label_colname

    def __iter__(self):
        for example in self.data:
            # Sample a single description from the list of descriptions
            description = example[self.descriptions_colname][
                np.random.randint(0, len(example[self.descriptions_colname]))
            ]
            yield example[self.id_colname], example[
                self.label_colname
            ], description
