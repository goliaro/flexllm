import json
import os
from tqdm import tqdm
from transformers import pipeline

os.chdir(os.path.dirname(os.path.realpath(__file__)))

prompts = [
 "A person is locked out of their house with no key. What are some possible solutions?",
 "Should social media platforms be allowed to collect personal data from users? Explain your reasoning.",
 "What might happen if humans could suddenly read each other's thoughts?",
 "What does the phrase 'actions speak louder than words' mean to you?",
 "How might artificial intelligence change education in the next 20 years?",
 "Someone feels overwhelmed by their daily responsibilities. What strategies might help them manage their stress?",
 "Write a short story that begins with: 'The last person on Earth sat alone in a room. There was a knock on the door.'",
 "What are the potential benefits and drawbacks of remote work becoming the norm?",
 "What makes a life meaningful?",
 "How would you design a city that prioritizes both environmental sustainability and human happiness?"
]
model_name = "meta-llama/Llama-3.1-8B-Instruct"


pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = pipe.tokenizer


results = []
for prompt in tqdm(prompts, desc="Processing prompts"):
    # Calculate the prompt length
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=100,
    )
    response = outputs[0]["generated_text"][-1]["content"]
    
    # Calculate response token count
    response_length = len(tokenizer.encode(response))
    
    # Create a dictionary for this prompt-response pair
    results.append({
        "prompt": prompt,
        "response": response,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "arrival_time": 0.0,
    })

json_contents = {"entries": results}
# Save the results to a JSON file
with open("../traces/kickstart.json", "w+") as f:
    json.dump(json_contents, f, indent=2)

os._exit(0)
