#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# Native Mooncake AscendDirectTransport/FindAdxlListenPort constants.
MOONCAKE_ADXL_DEFAULT_BASE_PORT = 20000
MOONCAKE_ADXL_PORT_STRIDE = 100
MOONCAKE_ADXL_MIN_PORT = 1024
MOONCAKE_ADXL_MAX_PORT = 65535

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # Existing Mooncake Ascend-direct native setting, not a new variable.
    # Unadjusted base defaults to 20000. The optional worker startup control
    # requires an integer in [1024, 65535], with the full device window in range.
    # This setting is not sensitive. Reading it never modifies the environment.
    "ASCEND_BASE_PORT": lambda: os.getenv("ASCEND_BASE_PORT", str(MOONCAKE_ADXL_DEFAULT_BASE_PORT)),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Experimental A5 MLA decode optimization. Split each request's Block Table
    # while broadcasting Query, then merge Local FIA outputs with AttentionUpdate.
    # It is disabled by default until the end-to-end accuracy/performance gates pass.
    "VLLM_ASCEND_MLA_FIA_SPLIT": lambda: bool(int(os.getenv("VLLM_ASCEND_MLA_FIA_SPLIT", "0"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_ffn_combine/mega_moe`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine/mega_moe` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # `mega_moe` can be used only for moe layer with W8A8/W4A8/bf16(none quant), EP<=64, non-dynamic-eplb.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())


def get_mooncake_adxl_base_port() -> int:
    """Read the existing native base without applying a device offset."""
    value = env_variables["ASCEND_BASE_PORT"]()
    if not value.isascii() or not value.isdecimal():
        raise ValueError("ASCEND_BASE_PORT must be an unsigned decimal integer.")
    base_port = int(value)
    if not MOONCAKE_ADXL_MIN_PORT <= base_port <= MOONCAKE_ADXL_MAX_PORT:
        raise ValueError("ASCEND_BASE_PORT must be in [1024, 65535].")
    return base_port


def configure_mooncake_disjoint_adxl_ports(
    physical_device_id: int,
    previous_assignment: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Set the native base once and return (original base, device, native base).

    Mooncake selects inclusively from [native_base + 100*d,
    native_base + 100*(d+1)]. Adding d to the original base separates these
    ranges. The worker owns the immutable assignment; this module keeps no
    process-global assignment state and never retries native initialization.
    """
    if type(physical_device_id) is not int or physical_device_id < 0:
        raise ValueError("Mooncake physical device ID must be a nonnegative integer.")
    if previous_assignment is not None:
        base_port, previous_device_id, native_base = previous_assignment
        if physical_device_id != previous_device_id:
            raise RuntimeError("Mooncake ADXL physical device changed after port assignment.")
        if os.environ.get("ASCEND_BASE_PORT") != str(native_base):
            raise RuntimeError("ASCEND_BASE_PORT changed after the worker's ADXL port assignment.")
        return previous_assignment
    base_port = get_mooncake_adxl_base_port()
    native_base = base_port + physical_device_id
    max_port = native_base + MOONCAKE_ADXL_PORT_STRIDE * (physical_device_id + 1)
    if max_port > MOONCAKE_ADXL_MAX_PORT:
        raise ValueError("Mooncake ADXL physical device range exceeds TCP port 65535.")
    assignment = (base_port, physical_device_id, native_base)
    os.environ["ASCEND_BASE_PORT"] = str(native_base)
    return assignment
