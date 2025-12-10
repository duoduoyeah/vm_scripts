from safetensors import safe_open
import os

model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
output_file = "model_tensors_info.txt"

with open(output_file, "w") as out:
    for file in os.listdir(model_path):
        if file.endswith('.safetensors'):
            header = f"\n=== {file} ==="
            print(header)
            out.write(header + "\n")
            
            with safe_open(os.path.join(model_path, file), "pt", "cpu") as f:
                for weight_name in f.keys():
                    tensor = f.get_tensor(weight_name)
                    line = f"{weight_name}: {tensor.shape}"
                    print(line)
                    out.write(line + "\n")

print(f"\nResults saved to {output_file}")

