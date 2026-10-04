from __future__ import annotations
import torch


def _autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype in (torch.bfloat16, torch.float16),
    )


class Code2WavDecoderGraph:
    """Fixed-window Code2Wav convolution decoder graph."""

    def __init__(
        self,
        code2wav: torch.nn.Module,
        *,
        window_frames: int,
        dtype: torch.dtype,
    ) -> None:
        self.code2wav = code2wav
        self.window_frames = int(window_frames)
        self.dtype = dtype
        device = next(code2wav.parameters()).device
        hidden_size = int(code2wav.config.hidden_size)
        self.transformed = torch.zeros(
            (1, self.window_frames, hidden_size),
            device=device,
            dtype=dtype,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.waveform: torch.Tensor | None = None
        self._capture()

    def _forward(self) -> torch.Tensor:
        hidden = self.transformed.permute(0, 2, 1)
        for blocks in self.code2wav.upsample:
            for block in blocks:
                hidden = block(hidden)
        waveform = hidden
        for block in self.code2wav.decoder:
            waveform = block(waveform)
        return waveform.clamp(min=-1, max=1)

    @torch.no_grad()
    def _capture(self) -> None:
        with _autocast(self.transformed.device, self.dtype):
            self._forward()
        torch.cuda.synchronize(self.transformed.device)
        with (
            _autocast(self.transformed.device, self.dtype),
            torch.cuda.graph(
                self.graph,
                capture_error_mode="thread_local",
            ),
        ):
            self.waveform = self._forward()
        if self.waveform is None:
            raise RuntimeError("Code2Wav decoder graph captured no waveform")

    @torch.no_grad()
    def replay(self, transformed: torch.Tensor) -> torch.Tensor:
        if tuple(transformed.shape) != tuple(self.transformed.shape):
            raise ValueError(
                f"Code2Wav decoder graph {self.window_frames} received {tuple(transformed.shape)}"
            )
        self.transformed.copy_(transformed)
        self.graph.replay()
        return self.waveform
