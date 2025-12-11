from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"

print("Loading model and tokenizer...")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!\n")

# Mathematical problem prompt - format for base model (completion, not instruction)
prompt = "Q: What is 127 + 348?\nA:"

# Prepare input - don't use chat template for base models
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

# Generate response
print(f"Prompt: {prompt}\n")
print("Generating answer...")

# Add newline tokens as stop condition
stop_strings = ["\nQ:", "\n\n"]
stop_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
# Flatten the list
stop_token_ids = [token_id for sublist in stop_token_ids for token_id in sublist]

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,
    temperature=None,
    top_p=None
)

# Decode output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

# Clean up response - take only the answer before any new question
response = response.split('\n')[0].strip()

print(f"\nModel response: {response}")
