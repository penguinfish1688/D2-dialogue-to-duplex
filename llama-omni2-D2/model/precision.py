from __future__ import annotations
import torch


def promote_aut_masters(student: torch.nn.Module) -> None:
    """Preserve original FP32 trainables and use native BF16 operations.

    Call before loading production weights, never after a blanket BF16 cast.
    Native Whisper Linear/Conv1d cast weights differentiably to their inputs.
    """
    encoder = student.speech_encoder
    for frontend in (encoder.conv1, encoder.conv2):
        if (type(frontend).__module__, type(frontend).__name__) != ("whisper.model", "Conv1d"):
            raise TypeError("FP32 AuT requires released Whisper convolution weight casting")
    frontend = encoder.conv1
    frontend._d2_aut_compute_dtype = torch.bfloat16
    if not hasattr(frontend, "_d2_aut_compute_dtype_hook"):

        def preserve_compute(module, arguments):
            if not arguments or not torch.is_tensor(arguments[0]):
                raise TypeError("Whisper frontend requires positional tensor input")
            return (arguments[0].to(dtype=module._d2_aut_compute_dtype), *arguments[1:])

        frontend._d2_aut_compute_dtype_hook = frontend.register_forward_pre_hook(preserve_compute)
    for parameter in student.parameters():
        dtype = torch.float32 if parameter.requires_grad else torch.bfloat16
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.detach().to(dtype=dtype)
    for module in student.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point() and buffer.dtype != torch.bfloat16:
                setattr(module, name, buffer.to(dtype=torch.bfloat16))
