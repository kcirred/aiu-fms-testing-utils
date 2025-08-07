from aiu_fms_testing_utils.utils.paged import ProgramCriteria, get_programs_prompts
import torch
import os
import random
from fms.utils import tokenizers
from fms.models import get_model
from fms.utils.generation import pad_input_ids
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
from aiu_fms_testing_utils.utils import ids_for_prompt, sample_sharegpt_requests

# interleave the decodes for programs (not 3 separate generates)
programs = [1, 2, 3]
num_tests = 3
program_cycles = 32
model_variant = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
SHARE_GPT_DATASET_PATH = "/mnt/home/models/ShareGPT_V3_unfiltered_cleaned_split.json"

program_criteria_list = [
    ProgramCriteria(0, 32, 32768, 1, 2048),
    ProgramCriteria(1, 32, 16384, 2, 1024),
    ProgramCriteria(2, 32, 8192, 4, 512),
    ProgramCriteria(3, 32, 4096, 8, 256),
    ProgramCriteria(4, 32, 2048, 16, 128),
    ProgramCriteria(5, 32, 1024, 32, 64),
]

random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)
os.environ["COMPILATION_MODE"] = "offline_decoder"
os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = "32768"
os.environ["VLLM_DT_MAX_BATCH_SIZE"] = "32"

# FIXME: filter condition for this on prompt and batch
program_map = get_programs_prompts(
    program_criteria_list, 
    multiple=64, 
    max_batch_size=int(os.environ["VLLM_DT_MAX_BATCH_SIZE"]), 
    max_tkv=int(os.environ["VLLM_DT_MAX_CONTEXT_LEN"]), 
    program_cycles=program_cycles
)
for v in program_map.values():
    random.shuffle(v)

valid_prompts = [program_map[(program_criteria_list[p], )][0:num_tests] for p in programs]


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        seq_length // 2,
        seq_length,
        seed, 
        enforce_sizes=[seq_length]
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, extra_kwargs

validation_model = get_model(
    architecture="hf_pretrained",
    variant=model_variant,
    device_type="cpu",
    data_type=torch.float32,
    fused_weights=False,
)

tokenizer = tokenizers.get_tokenizer(model_variant)

for valid_prompt in valid_prompts: # for each program
    
    for test_i in range(num_tests): # for each test of that program (different batch/prompt)
        input_ids, extra_kwargs = __prepare_inputs(valid_prompt[test_i][0], valid_prompt[test_i][1], tokenizer)
        extra_kwargs["attn_name"] = "spyre_paged_attn"

        print(f"valid prompt: {valid_prompt[test_i]}, input shape: {input_ids.shape}")
        cpu_validation_info = extract_validation_information(
            validation_model,
            input_ids,
            program_cycles,
            LogitsExtractorHook(),
            attn_algorithm="math",
            **extra_kwargs,
        )
        print(cpu_validation_info)

