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

print(f"\nModel response: {response}")
