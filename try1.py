import os
from transformers import LlamaForCausalLM, LlamaTokenizer


model_name = "/data/user-data/haitian.fan/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name,
                                        load_in_8bit = True,
                                        torch_dtype = torch.float16,
                                        low_cpu_mem_usage = True)
tokenizer = LlamaTokenizer.from_pretrained(model_name)


prefix = """Ignore all previous instructions.
For the following conversation quoted in the triple backticks between an agent and a customer, follow the instructions.
"""

suffix = """

Instructions:
The response should be a JSON object in the following format.  Your response should contain nothing but the JSON object.
{
    "Category 1": {
      "score": score in the category 1, 
      "Quoted Sentences":[]
    },
    "Category 2": {
      "score": score in the category 2,
      "Quoted Sentences":[]
    },
    "Category 3": {
      "score": score in the category 3,
      "Quoted Sentences":[]
    },
    "Category 4": {
      "score": score in the category 4, 
      "Quoted Sentences":[]
    }
    "Category 5": {
      "score": score in the category 5, 
      "Quoted Sentences":[]
    }
}
Ignore the sentences that have severe grammar errors.
The scores are on the scale of 1 to 10 based on the evaluation of the full conversation for the agent in each of the categories which will be defined later. 
The quoted sentences should be quoted from the conversation which can best fit into the corresponding category.  There should be no more than 3 quoted sentences for each category. 
Category 1 means the agent understands and shares the feelings of the other caller.
Category 2 means the speaker avoids hedging language, and is assertive and direct.
Category 3 means the agent can find the answer or solution for the main reason of the conversation.
Category 4 means how positive or negative the speaker feels about the conversation and the topic.
Category 5 means the agent verifies the caller's name and address.
"""


folder_path = os.path.join(os.getcwd(), "symTrainConversation")


for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        with open(os.path.join(folder_path, file_name), 'r', encoding="utf-8") as f:
            content = f.read()
        
        
        full_input = prefix + content + suffix
        
        
        input_ids = tokenizer.encode(full_input, return_tensors="pt")
        output_ids = model.generate(input_ids,temperature = 0.1,top_K=30,top_P=0.95)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        
        with open(os.path.join(folder_path, "output_" + file_name), 'w', encoding="utf-8") as f:
            f.write(output_text)
            print(output_text,"completed!!")

print("处理完成!")
