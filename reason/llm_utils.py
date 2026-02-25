import os
import time
import openai
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, llm_mode="vllm"):
    """
    llm_mode:
      - "vllm": local vLLM chat (optional dependency; may be broken on some systems)
      - "openai": OpenAI-compatible chat via openai SDK
      - "hf": HuggingFace Transformers local inference (recommended on HPC for Llama)
    """
    if llm_mode == "hf":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(model_name, token=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        def hf_chat(messages, **kwargs):
            # messages: list of {"role": "...", "content": "..."} (OpenAI style)
            input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

            gen = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=(temperature is not None and float(temperature) > 0),
                temperature=float(temperature) if temperature is not None else 0.0,
                pad_token_id=tok.eos_token_id,
            )
            # Decode only the newly generated part
            out = tok.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True)
            return out

        return hf_chat

    # ---- default: vllm vs openai based on model_name ----
    if "gpt" not in model_name:
        from vllm import LLM, SamplingParams
        client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_seq_len_to_capture=max_seq_len_to_capture)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    else:
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    return llm


def get_outputs(outputs, model_name):
    # HF backend returns a raw string
    if isinstance(outputs, str):
        return outputs
    if "gpt" not in model_name:
        return outputs[0].outputs[0].text
    else:
        return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)
