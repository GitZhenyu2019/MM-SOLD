"""
Build the TensorRT fp16 engine for the fixed pretrained DiffAE UNet.
"""
import time

import tensorrt as trt

import config as cfg

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, logger)
assert parser.parse_from_file(cfg.DIFFAE_UNET_ONNX), \
    [parser.get_error(i) for i in range(parser.num_errors)]
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
config.builder_optimization_level = 3
t0 = time.time()
eb = builder.build_serialized_network(network, config)
assert eb is not None
open(cfg.DIFFAE_TRT_ENGINE, "wb").write(memoryview(eb))
print(f"ENGINE_BUILT in {time.time()-t0:.0f}s -> {cfg.DIFFAE_TRT_ENGINE}", flush=True)
