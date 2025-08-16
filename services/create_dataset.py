import json
import os
from .openrouter_api import generate
from icecream import ic

def create_dataset(
    samples_file: str = 'samples_request.md',
    system_prompt_file: str = 'system_prompt.md',
    output_dataset_file: str = 'dataset.json',
    model_name: str = 'google/gemma-3-27b-it:free'
):
    dataset = []

    # Load system prompt
    try:
        with open(system_prompt_file, 'r', encoding='utf8') as f:
            system_prompt_content = f.read()
    except FileNotFoundError:
        ic(f"Error: System prompt file '{system_prompt_file}' not found.")
        return False, f"Error: System prompt file '{system_prompt_file}' not found."
    except Exception as e:
        ic(f"Error reading system prompt file '{system_prompt_file}': {e}")
        return False, f"Error reading system prompt file '{system_prompt_file}': {e}"

    # Load user samples
    try:
        with open(samples_file, 'r', encoding='utf8') as f:
            user_samples = f.readlines()
    except FileNotFoundError:
        ic(f"Error: User samples file '{samples_file}' not found.")
        return False, f"Error: Sample query file '{samples_file}' not found."
    except Exception as e:
        ic(f"Error reading user samples file '{samples_file}': {e}")
        return False, f"Error reading sample queries file'{samples_file}': {e}"

    ic(f"Starting dataset generation from {len(user_samples)} samples...")

    for i, sample_line in enumerate(user_samples):
        user_message = sample_line.strip()
        if not user_message:
            continue

        ic(f"Processing sample {i+1}/{len(user_samples)}): '{user_message}'")

        # Generate messages for LLM
        messages = [
            {'role': 'user', 'content': user_message}
        ]

        # Generate response
        assistant_response = generate(
            messages=messages,
            prompt_name=os.path.splitext(os.path.basename(system_prompt_file))[0],
            model=model_name
        )

        if assistant_response:
            dataset_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            dataset.append(dataset_entry)
            ic(f"Added entry to dataset for sample {i+1}.")
        else:
            ic(f"Failed to get response for sample {i+1}: '{user_message}'. Skipping.")

        # Save the dataset
        try:
            with open(output_dataset_file, 'w', encoding='utf8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            ic(f"Dataset successfully saved to '{output_dataset_file}'. Total records: {len(dataset)}")
        except Exception as e:
            ic(f"Error saving dataset to '{output_dataset_file}': {e}")
            return False, f"Ошибка при сохранении датасета в '{output_dataset_file}': {e}"
    
    return True, f"Dataset successfully created and saved to '{output_dataset_file}'. Total entries: {len(dataset)}"
