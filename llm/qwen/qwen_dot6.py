from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

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

def generate_response(conversation_history):
    """Generate a response for the given conversation history."""
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # Decode the response
    content = tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
    content_clean = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    return content, content_clean

def main():
    """Interactive chat loop."""
    print("=" * 60)
    print("Qwen3-0.6B Chat - Interactive CLI")
    print("=" * 60)
    print("Type your message (multiple lines supported).")
    print("Press Enter 4 times (3 empty lines) to send your message.")
    print("Type 'exit', 'quit', or 'q' on a new line to end.")
    print("Type 'clear' to reset conversation history.")
    print("=" * 60)
    print()
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        try:
            # Get multi-line user input
            print("You: ", end="", flush=True)
            lines = []
            empty_count = 0
            while True:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                    if empty_count >= 3 and lines:  # 3 empty lines after content means send
                        break
                    lines.append(line)
                else:
                    empty_count = 0
                    if line.strip().lower() in ['exit', 'quit', 'q'] and not lines:
                        print("\nGoodbye!")
                        return
                    if line.strip().lower() == 'clear' and not lines:
                        conversation_history = []
                        print("\nConversation history cleared!\n")
                        print("-" * 60)
                        print()
                        break
                    lines.append(line)
            
            user_input = "\n".join(lines).strip()
            
            # Skip if somehow empty or if we just cleared history
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nGenerating response...\n")
            response, response_clean = generate_response(conversation_history)
            
            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response_clean})
            
            # Display response
            print(f"Assistant: {response}\n")
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

if __name__ == "__main__":
    main()
