from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat_with_gpt2(prompt):
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Encode the user input and generate a response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example interaction
user_input = "Tell me a story about a lost treasure"
response = chat_with_gpt2(user_input)
print(response)