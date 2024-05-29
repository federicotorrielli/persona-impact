import torch
import ujson as json
from vllm import LLM, SamplingParams


def prompt_extractor(file_name: str):
    entries = []

    # Read the .jsonl file line by line
    with open(file_name, "r") as file:
        for line in file:
            # Parse the JSON line and append to the list
            entry = json.loads(line.strip())
            # Use 'ind' if it exists, otherwise fallback to 'filename'
            entry_key = entry.get("ind", entry.get("filename"))
            entry["sort_key"] = entry_key
            entries.append(entry)

    # Sort the list of dictionaries by the 'sort_key' field
    sorted_entries = sorted(entries, key=lambda x: x["sort_key"])

    # Create a dictionary with 'sort_key' values as keys
    ordered_dict = {entry["sort_key"]: entry for entry in sorted_entries}

    return ordered_dict


def filenames_in_a_folder(folder_path: str):
    import os

    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".jsonl")
    ]


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="VLLM Inference")
    parser.add_argument("--input", type=str, help="Input folder path")
    parser.add_argument("--output", type=str, help="Output folder path")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of .jsonl files in the input folder
    input_files = filenames_in_a_folder(input_folder)

    print("Input files: ", input_files)

    llm = LLM(
        model="yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling_params = SamplingParams(
        max_tokens=128, top_k=10, top_p=0.95, temperature=0.69
    )

    # Iterate through the input files
    for file in input_files:
        # Check if the output file already exists
        output_file = os.path.join(output_folder, os.path.basename(file))
        if os.path.exists(output_file):
            print(f"Skipping {file} as the output file {output_file} already exists.")
            continue

        # Extract the prompts from the .jsonl file
        prompts = prompt_extractor(file)

        # Create a list of prompt_texts using the sorted keys from the prompts dictionary
        sorted_keys = sorted(prompts.keys())
        prompt_texts = [prompts[key]["llm_instruction"] for key in sorted_keys]

        # Generate the continuation
        responses = llm.generate(prompt_texts, sampling_params)

        # Add the continuation to the prompt dictionary using the correct keys
        for key, response in zip(sorted_keys, responses):
            prompts[key]["llm_response"] = response.outputs[0].text

        # Write the prompt dictionary to a .jsonl file
        with open(output_file, "w") as file:
            for key in sorted_keys:  # Ensure the ordering is consistent
                file.write(json.dumps(prompts[key]) + "\n")


main()
