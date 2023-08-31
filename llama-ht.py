from typing import List, Literal, Optional, Tuple, TypedDict, Union
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

import torch
import os
from typing import List


Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]

        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def read_dialogs_from_file(file_path):
    with open(file_path, "r") as file:
        dialogs = json.load(file)
    return dialogs


# Function to load the main model for text generation
def load_model(model_name, quantization, device):
    if device == "cpu":
        device_map = {"": "cpu"}
    else:
        device_map = "auto"
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    return model


# Set the seeds for reproducibility
model_name = "/data/user-data/haitian.fan/Llama-2-7b-hf"
quantization = False
seed = 42
device = "cpu"
if device != "cpu":
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

model = load_model(model_name, quantization, device)

tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)


DEFAULT_SYSTEM_PROMPT = """Ignore all previous instructions.
For the following conversation quoted in the triple backticks between an agent and a customer, follow the instructions.
"""

conversation_prompt = """
```
{conversation}
```
"""

end_prompt = """

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


From: Anup Kumar <anup.kumar@electrifai.com> 


"""

conversation_path = os.path.join(os.getcwd(), "symTrainConversation")
if not os.path.exists(os.path.join(conversation_path, "symTrainConversationOutput")):
    os.mkdir(os.path.join(conversation_path, "symTrainConversationOutput"))

cnv_list = [os.path.join(conversation_path, x) for x in os.listdir(conversation_path)]
cnv_list = [x for x in cnv_list if os.path.isfile(x)]
print(len(cnv_list))
for cnv in cnv_list:
    with open(cnv, "r") as f:
        file = f.read()
        temp_prompt = conversation_prompt.format(conversation=file)
        temp_prompt += end_prompt
        # temp_prompt
        dialogs = [[{"role": "user", "content": temp_prompt}]]
        chats = format_tokens(dialogs, tokenizer)
        use_cache = True

        with torch.no_grad():
            for idx, chat in enumerate(chats):
                tokens = torch.tensor(chat).long()
                tokens = tokens.unsqueeze(0)
                if device == "cpu":
                    tokens = tokens.to("cpu")
                else:
                    tokens = tokens.to("cuda")
                outputs = model.generate(
                    tokens,
                    max_new_tokens=2048,
                    do_sample=False,
                    top_p=0.1,
                    temperature=0.1,
                    top_k=50,
                    use_cache=use_cache,
                    repetition_penalty=1.0,
                    length_penalty=1,
                )

                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("User input and model output deemed safe.")
                print(f"Model output:\n{output_text}")
                print("\n==================================\n")
                with open(
                    os.path.join(
                        os.path.sep.join(cnv.split(os.path.sep)[:-1]),
                        "symTrainConversationOutput",
                        cnv.split(os.path.sep)[-1],
                    ),
                    "w",
                ) as f:
                    f.write(output_text)
    # break
