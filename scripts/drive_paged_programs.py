from aiu_fms_testing_utils.utils.paged import ProgramCriteria, get_programs_prompts
import torch
import os
import random
from fms.models import get_model
from fms.utils.generation import pad_input_ids
from aiu_fms_testing_utils.testing.validation import (
    extract_validation_information,
    LogitsExtractorHook,
    GoldenTokenHook,
    capture_level_1_metrics,
    top_k_loss_calculator,
)
from torch import distributed as dist
from aiu_fms_testing_utils.utils import sample_sharegpt_requests, warmup_model
from transformers import AutoTokenizer
import json
import argparse

parser = argparse.ArgumentParser(
    description="Script which will drive paged programs for debugging"
)
parser.add_argument(
    "--programs",
    metavar='N',
    type=int,
    nargs='*',
    default=[],
    help="select what programs to drive. If not specified, will drive all combinations of programs. Note: This will only account for the first decode program -- check max_new_tokens for details",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=8,
    help="set this if you want to change the number of tokens generated per sequence (1 prefill + max_new_tokens-1 decodes). Note: If this value is larger than 64, this may result in switching decode programs mid generation"
)
parser.add_argument(
    '--batch_sizes',
    metavar='N',
    type=int,
    nargs='*',
    default=[],
    help='list of integers which denotes which batch sizes to potentially run per program. Note: if the batch size does not exist for a given program, it will be skipped and a warning will be given that another batch size was chosen'
)

parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--model_variant",
    type=str,
    default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    help="The model id or path to use for this test. Note: must be a huggingface format",
)
parser.add_argument(
    "--timing",
    type=str,
    choices=["e2e", "per-token"],
    default="",
    help="if set, how to time the generation of tokens, e2e or per-token",
)
parser.add_argument(
    "--program_criteria_json_path",
    type=str,
    help="path to json file containing the program criteria list",
)
parser.add_argument(
    "--share_gpt_path",
    type=str,
    help="path to share gpt file",
)

args = parser.parse_args()

# interleave the decodes for programs (not 3 separate generates)
max_new_tokens = args.max_new_tokens
batch_sizes = args.batch_sizes
model_variant = args.model_variant
SHARE_GPT_DATASET_PATH = args.share_gpt_path
USE_DISTRIBUTED = args.distributed
TIMING = args.timing
warmed_up = False

with open(args.program_criteria_json_path, 'r') as f:
    program_criteria_json_list = json.load(f)["programs"]
    program_criteria_list = []
    for i, d in enumerate(program_criteria_json_list):
        program_criteria_list.append(ProgramCriteria(i, d["max_batch"], d["max_tkv"], d["batch_granularity"], d["tkv_granularity"]))
    
    programs = [p.program_id for p in program_criteria_list] if len(args.programs) == 0 else args.programs

random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)
os.environ["COMPILATION_MODE"] = "offline_decoder"
if "VLLM_DT_MAX_CONTEXT_LEN" not in os.environ or "VLLM_DT_MAX_BATCH_SIZE" not in os.environ:
    print("Please specify VLLM_DT_MAX_CONTEXT_LEN and VLLM_DT_MAX_BATCH_SIZE environment variables")
    exit()

max_batch_size = int(os.environ["VLLM_DT_MAX_BATCH_SIZE"])
max_tkv = int(os.environ["VLLM_DT_MAX_CONTEXT_LEN"])

# FIXME: filter condition for this on prompt and batch
program_map = get_programs_prompts(
    program_criteria_list, 
    multiple=64, 
    max_batch_size=max_batch_size, 
    max_tkv=max_tkv,
    program_cycles=max_new_tokens
)
for v in program_map.values():
    random.shuffle(v)

# select prompts that fit the batch size criteria
valid_prompts = []
for program in programs:
    valid_prompt_shapes = program_map[(program_criteria_list[program], )]
    unpicked_prompt_shapes = []
    valid_program_prompts = []
    not_found_batch_sizes = set(batch_sizes)
    for valid_prompt_shape in valid_prompt_shapes:
        
        # check if we have found all of our batch sizes
        if len(not_found_batch_sizes) == 0:
            break

        # if the batch size exists, append it and remove from not found set
        if valid_prompt_shape[0] in not_found_batch_sizes:
            valid_program_prompts.append(valid_prompt_shape)
            not_found_batch_sizes.remove(valid_prompt_shape[0])
        else:
            unpicked_prompt_shapes.append(valid_prompt_shape)
    
    # in case we need to pick up a random one if we have not found all of the batch sizes
    if len(not_found_batch_sizes) > 0:
        print(f"need to select {len(not_found_batch_sizes)} prompts that do not satisfy given batch sizes: {not_found_batch_sizes} for program: {program}")
        valid_program_prompts.extend(unpicked_prompt_shapes[:len(not_found_batch_sizes)])
    # if the user didn't specify batch sizes, just pick one
    elif len(batch_sizes) == 0:
        valid_program_prompts.append(valid_prompt_shapes[0])

    valid_prompts.append(valid_program_prompts)


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        SHARE_GPT_DATASET_PATH,
        batch_size,
        tokenizer,
        32,
        seq_length,
        seed, 
        enforce_sizes=[seq_length]
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(tokenizer.encode(prompt, return_tensors="pt").squeeze(0))

    input_ids, extra_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, extra_kwargs

# metric calculator based on the cross-entropy and mean diff for each decode step
def __metric_calculator(r: torch.Tensor, t: torch.Tensor):
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

model_path_kwargs = {}
if os.path.exists(model_variant):
    model_path_kwargs = {"model_path": model_variant}
else:
    model_path_kwargs = {"variant": model_variant}

distributed_kwargs = {}
if USE_DISTRIBUTED:
    distributed_kwargs["distributed_strategy"] = "tp"
    distributed_kwargs["group"] = dist.group.WORLD

model = get_model(
    architecture="hf_pretrained",
    device_type="cpu",
    data_type=torch.float16,
    fused_weights=False,
    **model_path_kwargs,
    **distributed_kwargs
)

model.eval()
model.compile(
    backend="sendnn", options={"sendnn.dynamic": True}
)

validation_model = get_model(
    architecture="hf_pretrained",
    device_type="cpu",
    data_type=torch.float32,
    fused_weights=False,
    **model_path_kwargs,
    **distributed_kwargs
)

tokenizer = AutoTokenizer.from_pretrained(model_variant)

for program_id, valid_program_prompt_list in zip(programs, valid_prompts): # for each program
    print(f"*** testing program {program_id} ***")
    
    for valid_prompt in valid_program_prompt_list: # for each test of that program (different batch/prompt)
        input_ids, extra_kwargs = __prepare_inputs(valid_prompt[0], valid_prompt[1], tokenizer)
        extra_kwargs["attn_name"] = "spyre_paged_attn"
        print(f"program id: {program_id}, valid prompt: {valid_prompt}, input shape: {input_ids.shape}")

        # warmup aiu model
        if not warmed_up:
            warmup_model(
                model, 
                input_ids,
                max_new_tokens=max_new_tokens, 
                compile_dynamic_sendnn=True, 
                **extra_kwargs
            )
            warmed_up = True

        cpu_validation_info = extract_validation_information(
            validation_model,
            input_ids,
            max_new_tokens,
            LogitsExtractorHook(),
            attn_algorithm="math",
            **extra_kwargs,
        )

        aiu_validation_info = extract_validation_information(
            model,
            input_ids,
            max_new_tokens,
            GoldenTokenHook(cpu_validation_info.get_info("tokens")),
            only_last_token=False,
            timing=TIMING,
            **extra_kwargs,
        )

        # capture all level 1 metrics
        level_1_metrics = capture_level_1_metrics(
            cpu_validation_info.get_info("logits"),
            aiu_validation_info.get_info("logits"),
            top_k_loss_calculator(20, __metric_calculator),
        )

        for sentence_idx, token_idx, metrics_value in level_1_metrics:
            print(
                f"For Program {program_id} in sentence {sentence_idx + 1}, the metric for token {token_idx} is {metrics_value}"
            )


