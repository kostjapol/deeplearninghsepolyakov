import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    result_batch["data_object"] = pad_sequence(
        [elem["data_object"] for elem in dataset_items],
        batch_first=True,
        padding_value=0,
    )

    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
