import os
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model_name = "/data/user-data/haitian.fan/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name,
                                        load_in_8bit = False,
                                        torch_dtype = torch.float16,device_map='cpu',
                                        low_cpu_mem_usage = True)
tokenizer = LlamaTokenizer.from_pretrained(model_name)


prefix = """
Please focus on the conversation below while considering the following task.
You are a reviewer in call center who is able to only focus on AGENT's conversation
your task is to assess whether the agent greeted the caller/customer at the beginning of the call.
The call script is provided below:

###conversation###


"""

suffix = """

###Explanation###
The output will be presented in JSON format, containing three key-value pairs: 'Result', 'Explanation', and 'Sentences'.
'Result' will contain either 'yes' if the agent greeted the caller,or 'No' if the agent did not greet the caller.
'Sentences' :First you need to clearly identify which dialogue belongs to AGENT, then only export the AGENT's greeting sentence to caller
###Output Format###
The output will be presented in JSON format, containing three key-value pairs:

{
  "Call Opening": {
    "Result": "Yes",
    "Sentences": "only export the AGENT's greeting sentence to caller"
  }
}

If 'Result' is 'No', the 'Sentences' key will be empty. 
Result in given JSON format only 
"""


folder_path = os.path.join(os.getcwd(), "symTrainConversation")


for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        with open(os.path.join(folder_path, file_name), 'r', encoding="utf-8") as f:
            content = f.read()
        
        
        full_input = prefix + content + suffix
        
        
        input_ids = tokenizer.encode(full_input, return_tensors="pt")
        output_ids = model.generate(input_ids,temperature = 0.1,top_k=30,top_p=0.95)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        
        with open(os.path.join(folder_path, "output_" + file_name), 'w', encoding="utf-8") as f:
            f.write(output_text)
            print(output_text,"completed!!")

print("处理完成!")
