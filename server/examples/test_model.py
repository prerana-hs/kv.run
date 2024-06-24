# Load model directly
import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# model_path = '/gpfsnyu/scratch/yx2432/models/qwen2-7b-instruct'
# model_path = '/gpfsnyu/scratch/yx2432/models/chatglm3-6b'
model_path = "/gpfsnyu/scratch/yx2432/models/chatglm4-9b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.float16)

model.to('cuda')

# # Generate a prompt
# model_inputs = tokenizer(prompt, return_tensors="pt")

# # Generate a token
# output = model.generate(**model_inputs, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.9)


# prompt = "why is deep learning so popular these days?"
prompt = "给我讲个故事"

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
model_inputs = tokenizer([prompt], return_tensors="pt").to('cuda').to(torch.float16)

# input ids: [29680,  1709,  4797,  4697,  1893,  4929,  2271,  3206, 35392]
# hidden_states shape: [1, 9, 4096]
# (Pdb) hidden_states[0][0][:10]
# tensor([-0.0159,  0.0065,  0.0116,  0.0014, -0.0004, -0.0076, -0.0013,  0.0055,
#         -0.0060, -0.0010], device='cuda:0')


generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)


# (Pdb) model.model.layers[0].self_attn.q_proj.weight[0][:10]
# tensor([-9.6436e-03, -6.6528e-03,  5.1260e-05, -5.6839e-04,  5.8365e-04,
#         -3.7384e-03,  2.7466e-03, -2.7161e-03, -2.8687e-03, -8.5831e-04],
#        device='cuda:0', grad_fn=<SliceBackward0>)