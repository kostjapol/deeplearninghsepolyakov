import torch
from torch import nn


class LogPowerSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=1724,
        win_length=1724,
        hop_length=130,
        max_frames=600,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.max_frames = max_frames

        window = torch.blackman_window(win_length)
        self.register_buffer("window", window)

    def forward(self, audio):
        target_length = self.hop_length * (self.max_frames - 1)
        current_length = audio.shape[-1]

        if current_length < target_length:
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            audio = audio[..., :target_length]

        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        power_spectrum = spectrum.abs().pow(2)
        log_power_spectrum = torch.log(power_spectrum + 1e-6)

        return log_power_spectrum.unsqueeze(1)
