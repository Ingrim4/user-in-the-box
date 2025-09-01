import copy
import itertools
import json
import os
import yaml

def set_nested(config, key_path, value, append=False):
    keys = key_path.split(".")
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})

    if append and value is None:
        return
    if append and isinstance(d.get(keys[-1]), dict) and isinstance(value, dict):
        d[keys[-1]].update(value)
    else:
        d[keys[-1]] = value

# Load template YAML config
with open("template.yaml", "r") as file:
    template_config = yaml.safe_load(file)

# Load parameters to vary
with open("parameters.json", "r") as f:
    parameter_groups = json.load(f)

# Create output directory
output_dir = "generated"
os.makedirs(output_dir, exist_ok=True)

for group_idx, parameters in enumerate(parameter_groups):
    # Generate all combinations
    param_paths = []
    param_values = []
    
    for param in parameters:
        path = param["path"]
        append = param.get("append", False)
        values = param["values"]
        # Each value gets name + value + append flag
        param_paths.append({
            "path": path,
            "append": append
        })
        param_values.append(values)
    
    combinations = list(itertools.product(*param_values))
    
    group_dir = os.path.join(output_dir, f"mirror_{group_idx+1:02d}")
    os.makedirs(group_dir, exist_ok=True)
    
    # Generate and save configs
    for idx, combo in enumerate(combinations):
        config = copy.deepcopy(template_config)
        name_parts = []
    
        for param_meta, value_item in zip(param_paths, combo):
            path = param_meta["path"]
            append = param_meta["append"]
            value = value_item["value"]
            name = value_item["name"]
            name_parts.append(name)
    
            set_nested(config, path, value, append)
    
        name = f"mirror_{group_idx+1:02d}_{idx+1:03d}__{'__'.join(name_parts)}"
        config["simulator_name"] = name
    
        with open(os.path.join(group_dir, f"{name}.yaml"), "w") as f:
            yaml.dump(config, f)
        
        print(f"\"{name}\"")
