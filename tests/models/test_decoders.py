from fms.models.hf.utils import AutoConfig
from fms.utils import serialization
import pytest
from fms.models import get_model
from fms.utils.generation import pad_input_ids
import itertools
import warnings
import re
import torch
from torch import distributed as dist
from aiu_fms_testing_utils.testing.validation import (
    extract_validation_information,
    LogitsExtractorHook,
    GoldenTokenHook,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    get_default_validation_prefix,
    load_validation_information,
    validate_level_0,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils import (
    warmup_model,
    sample_sharegpt_requests,
    sample_granite_3_3_long_answerable_requests,
)
import json
from transformers import AutoTokenizer

from aiu_fms_testing_utils.utils.aiu_setup import dprint, aiu_dist_setup

import os

try:
    from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear  # noqa: F401

    GPTQ_ENABLED = True
except ImportError:
    GPTQ_ENABLED = False

MICRO_MODELS_HOME = os.environ.get(
    "FMS_TEST_SHAPES_MICRO_MODELS_HOME", "/mnt/home/models/tiny-models"
)

# Add models to test here
LLAMA_3p1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
GRANITE_3p2_8B_INSTRUCT = "ibm-granite/granite-3.2-8b-instruct"
GRANITE_3p3_8B_INSTRUCT = "ibm-granite/granite-3.3-8b-instruct"
GRANITE_20B_CODE_INSTRUCT_8K = "ibm-granite/granite-20b-code-instruct-8k"
LLAMA_3p1_70B_INSTRUCT = "meta-llama/Llama-3.1-70B-Instruct"

micro_model_mapping = {
    LLAMA_3p1_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "llama-3.1-8b-layers-3-step-24000"
    ),
    GRANITE_3p2_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "granite-3.2-8b-layers-3-step-100000"
    ),
    # FIXME: Because this uses the same config as 3.2, re-using here, but should update
    GRANITE_3p3_8B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "granite-3.3-8b-layers-3-step-100000"
    ),
    LLAMA_3p1_70B_INSTRUCT: os.path.join(
        MICRO_MODELS_HOME, "llama-3.1-70b-layers-3-step-24000"
    ),
}

DATASET_PATH = os.environ.get("DATASET_PATH", os.path.expanduser("~/share_gpt.json"))
USE_MICRO_MODELS = os.environ.get("FMS_TEST_SHAPES_USE_MICRO_MODELS", "1") == "1"
USE_DISTRIBUTED = os.environ.get("FMS_TEST_SHAPES_DISTRIBUTED", "0") == "1"
TIMING = os.environ.get("TIMING", "")
DATASET = os.environ.get("DATASET", "sharegpt")
ALLOWED_DATASET = {"sharegpt", "long-ctx"}
if DATASET not in ALLOWED_DATASET:
    raise ValueError(
        f"{DATASET} is not a valid environment for DATASET. Please use {ALLOWED_DATASET}"
    )
if "SHARE_GPT_DATASET_PATH" in os.environ:
    warnings.warn(
        "SHARE_GPT_DATASET_PATH is deprecated, please use DATASET_PATH environment instead"
    )

ATTN_TYPE = os.environ.get("FMS_TEST_SHAPES_ATTN_TYPE", "sdpa")
attention_map = {
    "sdpa": "sdpa_causal",
    "paged": "spyre_paged_attn",
    "math_fp8": "math_fp8",
    "paged_fp8": "spyre_paged_attn_fp8",
}
ATTN_NAME = attention_map[ATTN_TYPE]

FORCE_VALIDATION_LEVEL_1 = (
    os.environ.get("FMS_TEST_SHAPES_FORCE_VALIDATION_LEVEL_1", "0") == "1"
)
skip_assertions = os.environ.get("FMS_TEST_SHAPES_SKIP_ASSERTIONS", {})
validation_info_dir = os.environ.get(
    "FMS_TEST_SHAPES_VALIDATION_INFO_DIR", "/tmp/models/validation_info"
)
common_model_paths = os.environ.get(
    "FMS_TEST_SHAPES_COMMON_MODEL_PATHS",
    [
        LLAMA_3p1_8B_INSTRUCT,
        GRANITE_3p2_8B_INSTRUCT,
        GRANITE_3p3_8B_INSTRUCT,
        GRANITE_20B_CODE_INSTRUCT_8K,
        LLAMA_3p1_70B_INSTRUCT,
    ],
)
model_configuration_path = os.environ.get(
    "FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION", ""
)
model_configuration_frequency = os.environ.get(
    "FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION_FREQUENCY", "0"
)

# for validation level 1, the default is a failure rate of 1%
# set this environment variable if you would like to relax that threshold
failure_rate_threshold = os.environ.get("FMS_TEST_SHAPES_FAILURE_THRESHOLD", 0.01)
default_metrics_threshold = os.environ.get(
    "FMS_TEST_SHAPES_METRICS_THRESHOLD", (3.0, 0.001)
)
save_validation_info_outputs = (
    os.environ.get("FMS_TEST_SHAPES_SAVE_VALIDATION_INFO_OUTPUTS", "0") == "1"
)
common_batch_sizes = os.environ.get("FMS_TEST_SHAPES_COMMON_BATCH_SIZES", [1, 2, 4, 8])
common_seq_lengths = os.environ.get("FMS_TEST_SHAPES_COMMON_SEQ_LENGTHS", [64, 2048])
common_max_new_tokens = os.environ.get("FMS_TEST_SHAPES_COMMON_MAX_NEW_TOKENS", [128])

if USE_DISTRIBUTED:
    dist.init_process_group()
    aiu_dist_setup(dist.get_rank(), dist.get_world_size())
    save_validation_info_outputs = save_validation_info_outputs and (
        dist.get_rank() == 0
    )

if USE_MICRO_MODELS:
    validation_info_dir = os.path.join(validation_info_dir, "tiny_models")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if isinstance(common_model_paths, str):
    common_model_paths = common_model_paths.split(",")

# pass custom failure rate threshold as float
if isinstance(failure_rate_threshold, str):
    failure_rate_threshold = float(failure_rate_threshold)

# pass custom default metrics threshold as a comma separated str of floats <cross-entropy threshold>,<mean diff threshold>
if isinstance(default_metrics_threshold, str):
    default_metrics_threshold = tuple(
        [float(m) for m in default_metrics_threshold.split(",")]
    )

# pass custom common batch sizes as a comma separated str of ints
if isinstance(common_batch_sizes, str):
    common_batch_sizes = [int(bs) for bs in common_batch_sizes.split(",")]

# pass custom common seq lengths as a comma separated str of ints
if isinstance(common_seq_lengths, str):
    common_seq_lengths = [int(sl) for sl in common_seq_lengths.split(",")]

# pass custom common max new tokens as a comma separated str of ints
if isinstance(common_max_new_tokens, str):
    common_max_new_tokens = [int(mnt) for mnt in common_max_new_tokens.split(",")]

# pass metrics to skip as a comma separated list (ce,mean_diff)
if isinstance(skip_assertions, str):
    _skip_assertions = []
    for metric in skip_assertions.split(","):
        metric = metric.lower()
        if metric not in {"ce", "mean_diff"}:
            pytest.fail(
                "FMS_TEST_SHAPES_SKIP_ASSERTIONS can only accept metrics ce and mean_diff"
            )
        _skip_assertions.append(metric)
    skip_assertions = set(_skip_assertions)

compile_dynamic_sendnn = ATTN_TYPE == "paged"

if compile_dynamic_sendnn:
    import bisect

    # the compiler supports certain max context lengths (VLLM_DT_MAX_CONTEXT_LEN)
    # this will ensure that we select smallest supported VLLM_DT_MAX_CONTEXT_LEN that fits the largest possible context (prompt size + max_new_tokens)
    __largest_context = max(common_seq_lengths) + max(common_max_new_tokens)
    __supported_context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
        __supported_context_lengths[
            bisect.bisect_left(__supported_context_lengths, __largest_context)
        ]
    )
    os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(max(max(common_batch_sizes), 2))

# thresholds are chosen based on 1024 tokens per sequence
# 1% error threshold rate between cpu fp32 and cuda fp16
# if a models failure thresholds do not exist in this dict, default to the default_metrics_threshold defined above
# threshold key is (model_id, is_tiny_model)
fail_thresholds = {
    (LLAMA_3p1_8B_INSTRUCT, False): (
        2.6994638133048965,
        0.00047589250549208347,
    ),
    (GRANITE_3p2_8B_INSTRUCT, False): (
        2.3919514417648315,
        0.0005767398688476533,
    ),
    (GRANITE_3p3_8B_INSTRUCT, False): (
        2.4444521379470827,
        0.0004970188625156878,
    ),
    (GRANITE_20B_CODE_INSTRUCT_8K, False): (
        2.640706129074097,
        0.00034344267623964697,
    ),
    (LLAMA_3p1_70B_INSTRUCT, False): (
        2.841279556751251,
        0.0044301633024588115,
    ),
}

if model_configuration_path != "":
    print(
        "ignoring FMS_TEST_SHAPES_COMMON_MODEL_PATHS, FMS_TEST_SHAPES_USE_MICRO_MODELS as configuration will be set by FMS_TEST_SHAPES_FROM_MODEL_CONFIGURATION"
    )
    USE_MICRO_MODELS = False
    common_model_paths = []
    frequency = int(model_configuration_frequency)
    with open(model_configuration_path, "r") as f:
        for line in f:
            try:
                model_config = json.loads(line)
                if model_config["frequency"] <= frequency:
                    common_model_paths.append(model_config["model_id"])
                    # assume fullsize models
                    fail_thresholds[(model_config["model_id"], USE_MICRO_MODELS)] = (
                        model_config["ce"],
                        model_config["mean_diff"],
                    )
            except json.JSONDecodeError:
                print(f"config contained an improper json line: {line.strip()}")

common_shapes = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
    )
)

# custom weight adaptation to be used in future. For instance if we would like to add some other adaptation, we can register it with this custom adapter
# and provide it when converting from an aiu fms model's weights to a cpu fms model's weights. Currently this is only done for gptq, but may be done for other
# formats in the future
# note: llama already has many adapters for aiu and they are the same for all models, so just use llama. This way we don't need to re-register a new architecture / adapter step (we can just re-use)
__custom_adapter = {"architecture": "llama", "source": "fms_aiu"}


@pytest.fixture(autouse=True)
def reset_compiler():
    yield  # run the test
    if not compile_dynamic_sendnn:
        torch.compiler.reset()
        torch._dynamo.reset()
        os.environ.pop("COMPILATION_MODE", None)


# TODO: Currently, gptq does not have the same level of support as non-gptq models for get_model. This method provides the extra requirements for gptq for get_model,
#  however ideally, these fixes should be done in foundation-model-stack.
def __maybe_get_gptq_kwargs(model_path):
    gptq_adapter_step = []
    gptq_kwargs_aiu = {}
    gptq_kwargs_cpu = {}
    if GPTQ_ENABLED:
        # TODO: hf_configured/hf_pretrained options in get_model should be inferring the linear_config based on the hf quantization_config attribute
        config = AutoConfig.from_pretrained(model_path)
        if (
            hasattr(config, "quantization_config")
            and config.quantization_config["quant_method"] == "gptq"
        ):
            gptq_adapter_step.append("gptq_qweights_transpose_aiu")
            group_size = config.quantization_config["group_size"]
            desc_act = config.quantization_config["desc_act"]
            linear_config = {"group_size": group_size, "desc_act": desc_act}
            if USE_MICRO_MODELS:
                micro_aiu_kwargs = {"nlayers": 3}
                micro_cpu_kwargs = {"nlayers": 3}
            else:
                # TODO: infer the source based on the device for get_model when using gptq
                micro_aiu_kwargs = {"model_path": model_path, "source": "hf_gptq_aiu"}
                micro_cpu_kwargs = {"model_path": model_path, "source": "hf"}

            # TODO: infer the linear_type based on the device for get_model when using gptq
            gptq_kwargs_aiu = {
                "linear_config": {"linear_type": "gptq_aiu", **linear_config},
                "architecture": "hf_configured",
                "variant": model_path,
                **micro_aiu_kwargs,
            }
            gptq_kwargs_cpu = {
                "linear_config": {"linear_type": "gptq_cpu", **linear_config},
                "architecture": "hf_configured",
                "variant": model_path,
                **micro_cpu_kwargs,
            }
    try:
        # llama already has this adapter and it is the same for all models, so just use llama
        serialization.register_adapter(
            **__custom_adapter, adapter_steps=gptq_adapter_step
        )
    except KeyError:
        pass
    return gptq_kwargs_aiu, gptq_kwargs_cpu


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    if "paged" in ATTN_NAME:
        if DATASET == "sharegpt":
            prompts_and_sizes = sample_sharegpt_requests(
                DATASET_PATH,
                batch_size,
                tokenizer,
                32,
                seq_length * 2,  # this ensures we get sequences back
                seed,
                enforce_heterogeneous=False,
                enforce_sizes=[
                    seq_length
                ],  # ensure at least the max seq length is sampled
                pad_multiple=64,
                truncation=True,
            )
        elif DATASET == "long-ctx":
            dataset_lowerbound = 6336
            dataset_upperbound = 31744
            if seq_length > dataset_upperbound or seq_length < dataset_lowerbound:
                warnings.warn(
                    f"You have chosen {seq_length=} which isn't within the range of {dataset_lowerbound}-{dataset_upperbound} of the dataset",
                    stacklevel=2,
                )
            prompts_and_sizes = sample_granite_3_3_long_answerable_requests(
                DATASET_PATH,
                batch_size,
                tokenizer,
                4096,
                32768,  # for this dataset has no prompts longer or equal to 32768
                seed,
                enforce_heterogeneous=False,
                enforce_sizes=[
                    seq_length
                ],  # ensure at least the max seq length is sampled
                pad_multiple=64,
                truncation=False,
            )
        else:
            raise ValueError(f"env DATASET must be from the option: {ALLOWED_DATASET}")
    else:
        if DATASET == "sharegpt":
            prompts_and_sizes = sample_sharegpt_requests(
                DATASET_PATH,
                batch_size,
                tokenizer,
                seq_length // 2,
                seq_length,
                seed,
            )
        elif DATASET == "long-ctx":
            dataset_lowerbound = 6336
            dataset_upperbound = 31744
            if seq_length > dataset_upperbound or seq_length < dataset_lowerbound:
                warnings.warn(
                    f"You have chosen {seq_length=} which isn't within the range of {dataset_lowerbound}-{dataset_upperbound} of the dataset",
                    stacklevel=2,
                )
            prompts_and_sizes = sample_granite_3_3_long_answerable_requests(
                DATASET_PATH,
                batch_size,
                tokenizer,
                seq_length // 2,
                seq_length,
                seed,
            )
        else:
            raise ValueError(f"env DATASET must be from the option: {ALLOWED_DATASET}")

    prompt_list = []
    size_list = []
    for prompt, size in prompts_and_sizes:
        encoded = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        if size > seq_length:
            encoded = encoded[:seq_length]

        size_list.append(encoded.size(0))
        prompt_list.append(encoded)

    dprint(f"found prompts with sizes: {size_list}")
    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, extra_kwargs


def __find_eos_index(reference_tokens, eos_token_id, seq_length, max_new_tokens):
    result = []
    for sentence in reference_tokens:
        found_eos = False
        for token_idx, token in enumerate(sentence[seq_length:]):
            if token.item() == eos_token_id:
                found_eos = True
                result.append(token_idx)
                break
        if not found_eos:
            result.append(max_new_tokens)
    return result


def __filter_before_eos(metrics, filter_indexes):
    from itertools import groupby

    filtered_results = [
        list(g)[: filter_indexes[k]] for k, g in groupby(metrics, key=lambda x: x[0])
    ]
    return [item for sublist in filtered_results for item in sublist]


def __get_validation_info_full_path(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    seed,
    attn_type: str,
    device_type="cpu",
):
    validation_file_name = f"{get_default_validation_prefix(model_path, max_new_tokens, batch_size, seq_length, 'fp16', attn_type)}.{device_type}_validation_info.{seed}.out"
    full_path = os.path.join(validation_info_dir, validation_file_name)
    return full_path


def __load_validation_info(
    model_path, batch_size, seq_length, max_new_tokens, tokenizer, seed, attn_type: str
):
    # if path doesn't exist and paged isn't in the attention name, remove `attn_type` and recheck again, warn that we will no longer in the future have paths without 'attn_type'
    full_path = __get_validation_info_full_path(
        model_path, batch_size, seq_length, max_new_tokens, seed, attn_type
    )

    if os.path.exists(full_path):
        dprint(f"cpu validation info found for seed={seed} -- loading it")
        return load_validation_information(full_path, "logits", batch_size, tokenizer)
    elif "paged" not in attn_type:
        # This regex applies to a very specific file name format
        modified_full_path = re.sub(r"_attn-type[^.]*", "", full_path)

        if os.path.exists(modified_full_path):
            warnings.warn(
                f"All future paths should contain attn_type prefix information in path name, please modify {full_path=} to {modified_full_path=}",
                stacklevel=2,
            )
            dprint(f"cpu validation info found for seed={seed} -- loading it")
            return load_validation_information(
                modified_full_path, "logits", batch_size, tokenizer
            )
    return None


class PersistentModel:
    """This class will either get a model that is pre-compiled (if compile_dynamic_sendnn) or re-create the model for each test"""

    def __init__(self):
        self.model = None

    def get_or_create(self, is_gptq, is_fp8, **kwargs):
        if self.model is None:
            model = get_model(
                device_type="cpu",
                data_type=None if is_fp8 or is_gptq else torch.float16,
                fused_weights=False,
                **kwargs,
            )
            self.__maybe_reset_model(model, is_gptq)

            self.__maybe_prepare_fp8_weights(model, is_fp8)

            model.eval()
            model.compile(
                backend="sendnn", options={"sendnn.dynamic": compile_dynamic_sendnn}
            )

            if compile_dynamic_sendnn:
                self.model = model

            return model
        else:
            return self.model

    @staticmethod
    def __maybe_prepare_fp8_weights(model, is_fp8):
        if is_fp8:
            for name, param in model.named_parameters():
                if param.dtype == torch.bfloat16:
                    if param.max() > torch.finfo(torch.float16).max:
                        dprint(
                            f"[WARNING] You are casting param {name} to fp16, which will cause loss of accuracy. You can ignore this warning if this is intended."
                        )
                    param.data = param.data.to(dtype=torch.float16)

    # TODO: This was added as we require a special reset for gptq models. Ideally, we would be able to do something like this reset when calling reset_parameters() on the model
    #  however the gptq modules are yet to support this
    @staticmethod
    def __maybe_reset_model(model, is_gptq):
        if USE_MICRO_MODELS and is_gptq:
            sd = model.state_dict()
            for key, param in sd.items():
                if "qweight" in key:
                    res = torch.randint(
                        low=0,
                        high=torch.iinfo(torch.int32).max,
                        size=param.shape,
                        dtype=torch.int32,
                    )
                    sd[key].copy_(res)
                elif "qzeros" in key:
                    res = torch.ones(param.shape, dtype=torch.int32) * 8
                elif "g_idx" in key:
                    res = param
                else:
                    res = torch.randn_like(param)
                    res -= 0.5
                    res /= 20.0
                param.copy_(res)


@pytest.fixture
def persistent_model():
    return PersistentModel()


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens", common_shapes
)
def test_common_shapes(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    persistent_model,
    record_property,
):
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}, micro_model={USE_MICRO_MODELS}, attn_type={ATTN_TYPE}"
    )

    # we don't currently support inferring gptq from get_model, so we must use an adapter with hf_configured
    gptq_kwargs_aiu, gptq_kwargs_cpu = __maybe_get_gptq_kwargs(model_path)
    is_gptq = len(gptq_kwargs_aiu) != 0
    is_fp8 = "fp8" in ATTN_NAME

    micro_model_path = micro_model_mapping.get(model_path, None)
    if USE_MICRO_MODELS and micro_model_path is None:
        dprint("using randomly initialized model")
        micro_model_kwargs = {"architecture": "hf_configured", "nlayers": 3}
    else:
        dprint("using trained model")
        micro_model_kwargs = {"architecture": "hf_pretrained"}

    if not USE_MICRO_MODELS and os.path.exists(model_path):
        model_path_kwargs = {"model_path": model_path}
    elif USE_MICRO_MODELS and micro_model_path is not None:
        model_path_kwargs = {"model_path": micro_model_path}
    else:
        model_path_kwargs = {"variant": model_path}

    distributed_kwargs = {}
    if USE_DISTRIBUTED:
        distributed_kwargs["distributed_strategy"] = "tp"
        distributed_kwargs["group"] = dist.group.WORLD

    get_model_kwargs = {}
    if not is_gptq:
        get_model_kwargs = {
            **model_path_kwargs,
            **micro_model_kwargs,
            **distributed_kwargs,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # prepare the AIU model
    model = persistent_model.get_or_create(
        is_gptq, is_fp8, **gptq_kwargs_aiu, **get_model_kwargs
    )

    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=None if is_fp8 or is_gptq else torch.float32,
        fused_weights=False,
        **gptq_kwargs_cpu,
        **get_model_kwargs,
    )

    if USE_MICRO_MODELS:
        serialization.load_state_dict_into_model(
            validation_model, model.state_dict(), **__custom_adapter
        )

    # prepare input_ids
    input_ids, extra_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)
    extra_kwargs["attn_name"] = ATTN_NAME

    # warmup aiu model
    warmup_model(
        model, input_ids, max_new_tokens, compile_dynamic_sendnn, **extra_kwargs
    )

    # generate cpu validation info
    cpu_validation_info = __load_validation_info(
        model_path, batch_size, seq_length, max_new_tokens, tokenizer, 0, ATTN_NAME
    )
    if cpu_validation_info is None:
        cpu_validation_info = extract_validation_information(
            validation_model,
            input_ids,
            max_new_tokens,
            LogitsExtractorHook(),
            attn_algorithm="math",
            timing=TIMING,
            **extra_kwargs,
        )

        if save_validation_info_outputs:
            cpu_validation_info.save(
                __get_validation_info_full_path(
                    model_path, batch_size, seq_length, max_new_tokens, 0, ATTN_NAME
                )
            )
    cpu_static_tokens = cpu_validation_info.get_info("tokens")
    eos_indexes = __find_eos_index(
        cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens
    )
    dprint(
        "cpu validation info extracted for validation level 0 and validation level 1 (iter=0)"
    )

    # first test validation level 0
    aiu_validation_info = extract_validation_information(
        model,
        input_ids,
        max_new_tokens,
        None,
        only_last_token="paged" not in ATTN_NAME,
        timing=TIMING,
        **extra_kwargs,
    )
    dprint("aiu validation info extracted for validation level 0")

    # validate level 0
    failed_responses = validate_level_0(
        aiu_validation_info.get_info("tokens"), cpu_static_tokens
    )

    failed_validation_level_0 = len(failed_responses) != 0

    # if level 0 fails validation, validate level 1
    if FORCE_VALIDATION_LEVEL_1 or failed_validation_level_0:
        if failed_validation_level_0:
            dprint("failed validation level 0, testing validation level 1")
        else:
            dprint("passed validation level 0, testing validation level 1")

        # metric calculator based on the cross-entropy and mean diff for each decode step
        def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
            cross_entropy = torch.nn.CrossEntropyLoss()(
                r, t.softmax(dim=1).to(dtype=torch.float32)
            )
            diff = torch.mean(
                torch.abs(
                    r.softmax(dim=1).to(dtype=torch.float32)
                    - t.softmax(dim=1).to(dtype=torch.float32)
                )
            )
            return (cross_entropy, diff)

        iters = 1024 // max_new_tokens
        ce_fail_responses_list = []
        diff_fail_responses_list = []
        total_tokens = 0
        for i in range(iters):
            # for iteration 0, we have computed the cpu validation info in the prior step for seed=0, so skip
            if i != 0:
                input_ids, extra_kwargs = __prepare_inputs(
                    batch_size, seq_length, tokenizer, seed=i
                )
                extra_kwargs["attn_name"] = ATTN_NAME
                cpu_validation_info = __load_validation_info(
                    model_path,
                    batch_size,
                    seq_length,
                    max_new_tokens,
                    tokenizer,
                    i,
                    ATTN_NAME,
                )
                if cpu_validation_info is None:
                    cpu_validation_info = extract_validation_information(
                        validation_model,
                        input_ids,
                        max_new_tokens,
                        LogitsExtractorHook(),
                        attn_algorithm="math",
                        timing=TIMING,
                        **extra_kwargs,
                    )
                    dprint(
                        f"cpu validation info extracted for validation level 1 - iter={i}"
                    )
                    if save_validation_info_outputs:
                        cpu_validation_info.save(
                            __get_validation_info_full_path(
                                model_path,
                                batch_size,
                                seq_length,
                                max_new_tokens,
                                i,
                                ATTN_NAME,
                            )
                        )
                cpu_static_tokens = cpu_validation_info.get_info("tokens")
                eos_indexes = __find_eos_index(
                    cpu_static_tokens,
                    tokenizer.eos_token_id,
                    seq_length,
                    max_new_tokens,
                )

            # generate aiu validation info
            aiu_validation_info = extract_validation_information(
                model,
                input_ids,
                max_new_tokens,
                GoldenTokenHook(cpu_static_tokens),
                only_last_token=ATTN_TYPE != "paged",
                timing=TIMING,
                **extra_kwargs,
            )
            dprint(f"aiu validation info extracted for validation level 1 - iter={i}")
            if save_validation_info_outputs:
                aiu_validation_info.save(
                    __get_validation_info_full_path(
                        model_path,
                        batch_size,
                        seq_length,
                        max_new_tokens,
                        i,
                        ATTN_NAME,
                        "aiu",
                    )
                )

            # capture all level 1 metrics
            level_1_metrics = capture_level_1_metrics(
                cpu_validation_info.get_info("logits"),
                aiu_validation_info.get_info("logits"),
                top_k_loss_calculator(20, _metric_calculator),
            )
            # only consider those metrics captured prior to the eos
            level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

            # if we do not have real model weights, use a default_metrics_threshold
            if USE_MICRO_MODELS and micro_model_path is None:
                ce_threshold, diff_threshold = default_metrics_threshold
            # if we have real weights, try and get the proper validation metrics threshold
            else:
                # if we have a micro model with real weights, but no real thresholds, default to the full model thresholds
                if USE_MICRO_MODELS:
                    ce_threshold, diff_threshold = fail_thresholds.get(
                        (model_path, True),
                        fail_thresholds.get(
                            (model_path, False), default_metrics_threshold
                        ),
                    )
                else:
                    ce_threshold, diff_threshold = fail_thresholds.get(
                        (model_path, False), default_metrics_threshold
                    )

            # get all failed responses for each metric
            ce_fail_responses = filter_failed_level_1_cases(
                level_1_metrics, lambda m: m[0] >= ce_threshold
            )
            diff_fail_responses = filter_failed_level_1_cases(
                level_1_metrics,
                lambda m: m[1] >= diff_threshold,
            )

            ce_fail_responses_list.extend(ce_fail_responses)
            diff_fail_responses_list.extend(diff_fail_responses)
            total_tokens += len(level_1_metrics)

        # test the failure rates for across all tokens
        diff_failure_rate = len(diff_fail_responses_list) / total_tokens
        ce_failure_rate = len(ce_fail_responses_list) / total_tokens
        dprint(f"mean diff failure rate: {diff_failure_rate}")
        dprint(f"cross entropy loss failure rate: {ce_failure_rate}")
        # Add failure rates to xml report
        record_property("mean_diff_failure_rate", diff_failure_rate)
        record_property("cross_entropy_loss_failure_rate", ce_failure_rate)
        if "mean_diff" not in skip_assertions:
            assert diff_failure_rate < failure_rate_threshold, (
                f"failure rate for mean diff was too high: {diff_failure_rate}"
            )
        if "ce" not in skip_assertions:
            assert ce_failure_rate < failure_rate_threshold, (
                f"failure rate for cross entropy loss was too high: {ce_failure_rate}"
            )

        print("passed validation level 1")
    else:
        print("passed validation level 0")
