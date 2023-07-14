import torch
import time
time1 = time.time()
"""
应加载原始精度模型到 CPU 后再开始量化,耗时很长
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc\Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("baichuan-inc\Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc\Baichuan-13B-Chat", torch_dtype=torch.float16, trust_remote_code=True)

#使用 int8 量化：15.8G
model = model.quantize(8).cuda() 

#使用 int4 量化：9.7G
#model = model.quantize(4).cuda()

model.generation_config = GenerationConfig.from_pretrained("baichuan-inc\Baichuan-13B-Chat")

messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
response = model.chat(tokenizer, messages)
print(response)
time2 = time.time()
print(f'耗时2{round(time2-time1, 2)}s') #耗时291.37s


