from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVSpoofDataset(BaseDataset):
    def __init__(
        self,
        audio_dir,
        protocol_path,
        name="train",
        *args,
        **kwargs,
    ):
        index_path = ROOT_PATH / "data" / "asvspoof" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(
                audio_dir=audio_dir,
                protocol_path=protocol_path,
                index_path=index_path,
            )

        super().__init__(index, *args, **kwargs)

    def _create_index(self, audio_dir, protocol_path, index_path):
        index = []
        audio_dir = Path(audio_dir)

        with open(protocol_path, "r") as protocol_file:
            for line in protocol_file:
                parts = line.split()

                audio_id = parts[1]
                label_name = parts[-1]

                if label_name == "bonafide":
                    label = 1
                else:
                    label = 0

                audio_path = audio_dir / f"{audio_id}.flac"

                index.append(
                    {
                        "path": str(audio_path),
                        "label": label,
                    }
                )

        index_path.parent.mkdir(exist_ok=True, parents=True)
        write_json(index, str(index_path))

        return index

    def load_object(self, path):
        audio, sample_rate = torchaudio.load(path)
        return audio.squeeze(0)
