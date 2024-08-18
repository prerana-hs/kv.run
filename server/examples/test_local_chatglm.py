import enum
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# config_dict, _ = PretrainedConfig.get_config_dict(
#     "THUDM/glm-4-9b-chat", trust_remote_code=True
# )

# print(config_dict)

# config_dict2, _ = PretrainedConfig.get_config_dict(
#     "THUDM/chatglm3-6b", trust_remote_code=True
# )



device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

query = "你好"

# inputs_not_tokenized = tokenizer.apply_chat_template([{"role": "user", "content": query}],
#                                        add_generation_prompt=True,
#                                        tokenize=False,
#                                        return_tensors="pt",
#                                        return_dict=False
#                                        )

# print(inputs_not_tokenized)

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

print(model.can_generate())

gen_kwargs = {"max_length": 25, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# with torch.no_grad():
#     inputs_for_gen = model.prepare_inputs_for_generation(**inputs, **gen_kwargs)
#     outputs = model.forward(**inputs_for_gen)
#     outputs = outputs['logits'][:, inputs['input_ids'].shape[1]:]
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))