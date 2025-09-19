import pytest
import os
from subprocess import Popen, PIPE
from pathlib import Path
import itertools
import math

FMS_DIR = Path(__file__).parent
AIU_FMS_DIR = os.path.join(FMS_DIR, "../../../aiu-fms-testing-utils/")
INFERENCE_FILE_PATH = os.path.join(AIU_FMS_DIR, "scripts", "inference.py")

common_model_paths = os.environ.get("FMS_TESTING_COMMON_MODEL_PATHS", "")

# pass custom model path list for eg: EXPORT FMS_TESTING_COMMON_MODEL_PATHS="/tmp/models/granite-3-8b-base,/tmp/models/granite-7b-base"
if common_model_paths == "":
    common_model_paths = ["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
else:
    common_model_paths = common_model_paths.split(",")

common_batch_sizes = [1, 4]
common_seq_lengths = [64]
common_max_new_tokens = [8]
common_attn_types = ["sdpa", "paged"]
common_allow_symbolic_shapes = [None]

common_params = list(
    itertools.product(
        common_model_paths,
        common_batch_sizes,
        common_seq_lengths,
        common_max_new_tokens,
        common_attn_types,
        common_allow_symbolic_shapes,
    )
)

current_env = os.environ.copy()


def execute_script(execute_cmd):
    with Popen(
        execute_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        env=current_env,
    ) as p:
        output, error = p.communicate()
        if p.returncode == 0:
            return output
        else:
            raise Exception(error)


def execute_inference(
    model_path, batch_size, seq_length, max_new_tokens, attn_type, allow_symbolic_shapes
):
    extra_args = []
    if attn_type == "paged":
        # paged needs symbolic shapes
        extra_args.append("--attention_type=paged")
        # using these options temporarily
        current_env["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = "16384"
        current_env["VLLM_DT_MAX_BATCH_SIZE"] = "4"
        current_env["VLLM_DT_MAX_CONTEXT_LEN"] = "4096"
    else:
        # added in case symbolic shapes used with sdpa
        current_env["_PROMPT_LEN"] = "64"
        current_env["_MAX_DECODE_TOKENS"] = "8"
        current_env["_MAX_CONTEXT_LEN"] = "71"

    if allow_symbolic_shapes is not None and allow_symbolic_shapes:
        extra_args.append("--compile_dynamic_sendnn")

    execute_cmd = [
        "python3",
        INFERENCE_FILE_PATH,
        "--architecture=hf_pretrained",
        f"--variant={model_path}",
        f"--tokenizer={model_path}",
        f"--max_new_tokens={max_new_tokens}",
        f"--min_pad_length={seq_length}",
        f"--batch_size={batch_size}",
        "--unfuse_weights",
        "--no_early_termination",
        "--compile_dynamic",
        "--compile",
        "--device_type=aiu",
        "--default_dtype=fp16",
    ]
    return execute_script(execute_cmd + extra_args)


common_asserts = [
    "### Response:\n\n1.\n\nThe following",
    "### Response:\n\n1.\n\nI am",
    "### Response:\n\nI am not sure what you",
    "### Response:\n\nI have just come into a",
]


def __repeat_batch_asserts(bs: int) -> list[str]:
    n_repeats = int(math.ceil(bs / len(common_asserts)))
    return (common_asserts * n_repeats)[:bs]


# add the asserts based on batch size
# for batches greater than common_asserts, repeat common_asserts since this follows inference behavior
common_inference_params = [
    common_param + (__repeat_batch_asserts(common_param[1]),)
    for common_param in common_params
]
# adding special case where we allow symbolic shapes for batch size 1 using sdpa
common_inference_params.append(
    (common_model_paths[0], 1, 64, 8, "sdpa", [common_asserts[0]], True)
)


@pytest.mark.parametrize(
    "model_path,batch_size,seq_length,max_new_tokens,attn_type,asserts,allow_symbolic_shapes",
    common_inference_params,
)
def test_inference_script(
    model_path,
    batch_size,
    seq_length,
    max_new_tokens,
    attn_type,
    asserts,
    allow_symbolic_shapes,
):
    # force symbolic shapes if paged
    if "paged" in attn_type:
        allow_symbolic_shapes = True
    result_text = execute_inference(
        model_path,
        batch_size,
        seq_length,
        max_new_tokens,
        attn_type,
        allow_symbolic_shapes,
    )

    for common_assert in asserts:
        assert common_assert in result_text
