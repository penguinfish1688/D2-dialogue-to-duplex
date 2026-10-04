"""Reuse a captured native estimator across the ten diffusion steps."""

from collections import OrderedDict

import torch


class EstimatorGraph:
    def __init__(self, forward):
        self.forward = forward
        self.cache = OrderedDict()

    @torch.inference_mode()
    def __call__(self, *args):
        key = tuple((tuple(value.shape), value.dtype, value.device) for value in args)
        if key not in self.cache:
            # Prefix lengths grow during speech. Bound retained graphs instead
            # of accumulating one allocation for every length ever observed.
            while len(self.cache) >= 4:
                self.cache.popitem(last=False)
            inputs = tuple(value.clone() for value in args)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                expected = self.forward(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward(*inputs)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
            self.cache[key] = inputs, graph, output
        inputs, graph, output = self.cache[key]
        self.cache.move_to_end(key)
        for target, source in zip(inputs, args, strict=True):
            target.copy_(source)
        graph.replay()
        return output.clone()
