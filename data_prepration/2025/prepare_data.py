"""Utilities for loading Brain-to-Text 2025 HDF5 data."""

from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BrainToText2025Dataset(Dataset):
    """Iterates over all trials contained in Brain-to-Text 2025 sessions."""

    def __init__(self, data_dir):
        self.session_paths = sorted(Path(data_dir).glob("hdf5_data_final/*/*.hdf5"))
        self.trial_index = []
        for session_path in self.session_paths:
            with h5py.File(session_path, "r") as f:
                for trial_key in f.keys():
                    self.trial_index.append((session_path, trial_key))

    def __len__(self):
        return len(self.trial_index)

    def __getitem__(self, idx):
        session_path, trial_key = self.trial_index[idx]
        with h5py.File(session_path, "r") as f:
            g = f[trial_key]
            sample = {
                "input_features": torch.from_numpy(g["input_features"][:]),
                "seq_class_ids": torch.from_numpy(g["seq_class_ids"][:]),
                "transcription": "".join(map(chr, g["transcription"][:])),
                "n_time_steps": int(g.attrs["n_time_steps"]),
                "seq_len": int(g.attrs["seq_len"]),
                "block_num": int(g.attrs["block_num"]),
                "trial_num": int(g.attrs["trial_num"]),
            }
        return sample


def collate_batch(batch):
    features = [b["input_features"] for b in batch]
    labels = [b["seq_class_ids"] for b in batch]
    transcriptions = [b["transcription"] for b in batch]
    n_time_steps = [b["n_time_steps"] for b in batch]
    seq_lens = [b["seq_len"] for b in batch]
    block_nums = [b["block_num"] for b in batch]
    trial_nums = [b["trial_num"] for b in batch]

    return {
        "input_features": pad_sequence(features, batch_first=True, padding_value=0),
        "seq_class_ids": pad_sequence(labels, batch_first=True, padding_value=0),
        "transcriptions": transcriptions,
        "n_time_steps": torch.tensor(n_time_steps),
        "seq_lens": torch.tensor(seq_lens),
        "block_nums": torch.tensor(block_nums),
        "trial_nums": torch.tensor(trial_nums),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load Brain-to-Text 2025 data and print a summary."
    )
    parser.add_argument(
        "data_dir", help="Directory containing the hdf5_data_final folder"
    )
    args = parser.parse_args()

    dataset = BrainToText2025Dataset(args.data_dir)
    print(f"Found {len(dataset)} trials across {len(dataset.session_paths)} sessions.")
    sample = dataset[0]
    print("Sample feature shape:", sample["input_features"].shape)
    print("Sample label length:", len(sample["seq_class_ids"]))
    print("Sample transcription:", sample["transcription"])
    print("Sample n_time_steps:", sample["n_time_steps"])
