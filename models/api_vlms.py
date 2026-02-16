import os
import base64
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import subprocess

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import save_classification_results, fuzzy_match_label
from utils.mcqa import get_mcqa_choices, load_all_dataset_classes


class SafeFormatDict(dict):
    """Dict that returns empty string for missing keys in format_map."""
    def __missing__(self, key):
        return ""

def encode_image_base64(image_path: str) -> tuple[str, str]:
    """
    Encode image to base64. Converts TIFF to PNG if needed.
    Returns (base64_string, mime_type).
    """
    import io
    from PIL import Image
    
    # Check if it's a TIFF file
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        # Convert TIFF to PNG in memory
        img = Image.open(image_path)
        
        # Convert to RGB if needed (some TIFFs have different color modes)
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        # Save as PNG to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_str, 'image/png'
    else:
        # For other formats, read directly
        with open(image_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine MIME type
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(ext, 'image/jpeg')
        
        return base64_str, mime_type


def test_openai(args: dict, model_type: str, dataset: str, output_dir: str):
    
    INPUT_TOKEN_PRICE = 0.1
    # INPUT_TOKEN_PRICE = 1.25
    OUTPUT_TOKEN_PRICE = 0.4
    # OUTPUT_TOKEN_PRICE = 10.0

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from bashrc
        import subprocess
        result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $OPENAI_API_KEY'], 
                            capture_output=True, text=True)
        api_key = result.stdout.strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("Loaded API key from ~/.bashrc")
        else:
            raise ValueError("OPENAI_API_KEY not found. Please set it in ~/.bashrc or export it in terminal")
    
    client = OpenAI(api_key=api_key)
    
    # load dataset
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a subset
    sample_limit = args.get("sample_limit", None)  # Default to full dataset
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)
    
    # prepare data
    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    
    # build prompt
    candidate_labels = class_names
    classes_text = ", ".join(candidate_labels)
    prompt_template = args.get("prompt_template", 
        "Classify this image into one of the following categories: {classes}. "
        "Respond with ONLY the category name, nothing else.")

    # Check if MCQA mode is enabled
    mcqa_options = args.get("mcqa_options", None)
    all_dataset_classes = None
    answer_included_list = []
    mcqa_correct_answers = []  # Track the correct answer for each sample
    mcqa_choices_list = []  # Track the choices for each sample
    
    if mcqa_options:
        print(f"\nMCQA Mode Enabled:")
        print(f"  Options within dataset: {mcqa_options.get('options_within_dataset', True)}")
        print(f"  Number of choices: {mcqa_options.get('mcqa_num_choices', 4)}")
        
        # Load all dataset classes if needed for cross-dataset sampling
        if not mcqa_options.get('options_within_dataset', True):
            all_dataset_classes = load_all_dataset_classes()
            print(f"  Loaded {len(all_dataset_classes)} datasets for cross-dataset sampling")

    # Safely format the prompt_template if not using MCQA (MCQA will format per-sample)
    if not mcqa_options:
        fmt_map = {"classes": classes_text}
        user_prompt = prompt_template.format_map(SafeFormatDict(fmt_map))
    else:
        user_prompt = None  # Will be set per-sample
    
    # Get parallel processing settings
    max_workers = args.get("max_workers", 1)  # Default to sequential (1 worker)
    use_parallel = max_workers > 1
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    chosen_options = []  # Track which option number (1, 2, 3, ...) was chosen
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Thread-safe lock for updating token counts
    token_lock = Lock()
    
    def process_single_image(image_path, sample_idx=0):
        """Process a single image and return prediction results."""
        # Generate MCQA choices if enabled
        if mcqa_options:
            true_label = df.iloc[sample_idx]["label"]
            choices, correct_answer, answer_included, correct_answer_index = get_mcqa_choices(
                true_label=true_label,
                all_classes=candidate_labels,
                options_within_dataset=mcqa_options.get('options_within_dataset', True),
                mcqa_num_choices=mcqa_options.get('mcqa_num_choices', 4),
                all_dataset_classes=all_dataset_classes,
                current_dataset=dataset,
                answer_included_ratio=0.7,
                sample_index=sample_idx,
                print_sample=(sample_idx == 0)  # Print first sample only
            )
            sample_choices_str = ", ".join(choices)
            current_prompt = prompt_template.format(classes=sample_choices_str)
        else:
            current_prompt = user_prompt
            answer_included = None
            correct_answer = None
            choices = None
        
        # encode image (converts TIFF to PNG if needed)
        base64_image, mime_type = encode_image_base64(image_path)
        
        # call API
        response = client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": current_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50
        )
        
        # parse response
        prediction = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # For MCQA, match against the choices for this sample
        if mcqa_options and choices:
            # fuzzy matching to find the predicted class from the MCQA choices
            predicted_class, match_score, matched_label = fuzzy_match_label(
                prediction, choices, threshold=0.6
            )
            
            # Track which option was chosen (1-indexed)
            if predicted_class is not None:
                chosen_option = predicted_class + 1  # Convert to 1-indexed
                # Map the choice back to the original candidate_labels for storing pred_id
                matched_choice = choices[predicted_class]
                if matched_choice in candidate_labels:
                    predicted_class = candidate_labels.index(matched_choice)
                else:
                    # It's "None of the above" or not in original labels
                    predicted_class = None
            else:
                chosen_option = None
                match_score = 0.0
        else:
            # Standard fuzzy matching to find the predicted class
            predicted_class, match_score, matched_label = fuzzy_match_label(
                prediction, candidate_labels, threshold=0.6
            )
            chosen_option = None  # Not applicable for non-MCQA
            
            # If no match found, keep as None (for open-ended evaluation)
            if predicted_class is None:
                match_score = 0.0
        
        # create one-hot encoded probabilities
        probs = [0.0] * len(candidate_labels)
        if predicted_class is not None:
            probs[predicted_class] = 1.0
        
        return predicted_class, probs, input_tokens, output_tokens, prediction, match_score, answer_included, correct_answer, chosen_option
    
    # Process images
    if use_parallel:
        print(f"Processing images with {max_workers} parallel workers...")
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_image, path, idx): idx 
                           for idx, path in enumerate(paths)}
            
            # Collect results as they complete
            results = [None] * len(paths)  # Preserve order
            
            for future in tqdm(as_completed(future_to_idx), total=len(paths), desc="Testing"):
                idx = future_to_idx[future]
                try:
                    predicted_class, probs, input_tokens, output_tokens, prediction, match_score, answer_included, correct_answer, chosen_option = future.result()
                    results[idx] = (predicted_class, probs, prediction, match_score, answer_included, correct_answer, chosen_option)
                    
                    # Thread-safe token counting
                    with token_lock:
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        
                    # Print warning for non-matches
                    if predicted_class is None:
                        print(f"\nWARNING: No match found for: '{prediction}'")
                        
                except Exception as e:
                    print(f"\nWarning: Error processing image {idx}: {e}")
                    # Default to None on error
                    results[idx] = (None, [0.0] * len(candidate_labels), "", 0.0, None, None, None)
            
            # Extract predictions in correct order
            for predicted_class, probs, prediction, match_score, answer_included, correct_answer, chosen_option in results:
                preds_ids.append(predicted_class)
                probs_rows.append(probs)
                generated_texts.append(prediction)
                match_scores.append(match_score)
                if mcqa_options:
                    if answer_included is not None:
                        answer_included_list.append(answer_included)
                    if correct_answer is not None:
                        mcqa_correct_answers.append(correct_answer)
                    if chosen_option is not None:
                        chosen_options.append(chosen_option)
    
    else:
        # Sequential processing (original implementation)
        print("Processing images sequentially...")
        
        for idx, image_path in enumerate(tqdm(paths, desc="Testing")):
            try:
                predicted_class, probs, input_tokens, output_tokens, prediction, match_score, answer_included, correct_answer, chosen_option = process_single_image(image_path, idx)
                
                preds_ids.append(predicted_class)
                probs_rows.append(probs)
                generated_texts.append(prediction)
                match_scores.append(match_score)
                if mcqa_options:
                    if answer_included is not None:
                        answer_included_list.append(answer_included)
                    if correct_answer is not None:
                        mcqa_correct_answers.append(correct_answer)
                    if chosen_option is not None:
                        chosen_options.append(chosen_option)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Print warning for non-matches
                if predicted_class is None:
                    print(f"\nWARNING: No match found for: '{prediction}'")
                
            except Exception as e:
                print(f"\nWarning: Error processing {image_path}: {e}")
                # Default to None on error
                preds_ids.append(None)
                probs_rows.append([0.0] * len(candidate_labels))
                generated_texts.append("")
                match_scores.append(0.0)
                if mcqa_options:
                    answer_included_list.append(None)
            
    
    # calculate costs
    total_price = (total_input_tokens * INPUT_TOKEN_PRICE + total_output_tokens * OUTPUT_TOKEN_PRICE) / 1e6
    print("\nToken Usage Summary:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total cost: ${total_price:.6f}")
    
    # save token usage to JSON file
    token_usage = {
        "model": model_type,
        "dataset": dataset,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_token_price_per_million": INPUT_TOKEN_PRICE,
        "output_token_price_per_million": OUTPUT_TOKEN_PRICE,
        "total_cost_usd": round(total_price, 6),
        "num_images": len(df)
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    token_usage_path = os.path.join(output_dir, "token_usage.json")
    with open(token_usage_path, 'w') as f:
        json.dump(token_usage, f, indent=2)
    print(f"Token usage saved to: {token_usage_path}")
    
    # save metrics
    extra_cols = {}
    if mcqa_options:
        if answer_included_list:
            extra_cols['answer_included'] = answer_included_list
        if mcqa_correct_answers:
            extra_cols['mcqa_correct_answer'] = mcqa_correct_answers
        if chosen_options:
            extra_cols['chosen_option'] = chosen_options
    
    # For MCQA, we need to adjust y_true to reflect the correct answer
    # (which may be "None of the above" when answer is not included)
    y_true_adjusted = y_true.copy() if not mcqa_options else None
    if mcqa_options:
        y_true_adjusted = []
        for i, correct_ans in enumerate(mcqa_correct_answers):
            if correct_ans in candidate_labels:
                y_true_adjusted.append(candidate_labels.index(correct_ans))
            else:
                # "None of the above" - use a special marker (-1 will be handled in save_classification_results)
                y_true_adjusted.append(None)
        y_true_adjusted = [y if y is not None else -1 for y in y_true_adjusted]
    
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true_adjusted,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores,
        **extra_cols
    )


def test_gemini(args: dict, model_type: str, dataset: str, output_dir: str):
    """
    Test using Google Gemini API for image classification.
    
    Supports both real-time and batch API:
    - Real-time: Immediate results, standard pricing
    - Batch: Submit jobs, poll for completion, 50% discount
    
    Pricing (as of Dec 2024):
    Real-time:
    - Gemini 2.5 Flash: $0.075/1M input tokens, $0.30/1M output tokens
    - Gemini 1.5 Flash: $0.075/1M input tokens, $0.30/1M output tokens
    - Gemini 1.5 Pro: $1.25/1M input tokens, $10/1M output tokens
    
    Batch (50% discount):
    - Gemini 2.5 Flash: $0.0375/1M input tokens, $0.15/1M output tokens
    - Gemini 1.5 Flash: $0.0375/1M input tokens, $0.15/1M output tokens
    - Gemini 1.5 Pro: $0.625/1M input tokens, $5/1M output tokens
    """
    
    # Check if batch mode is enabled
    use_batch = args.get("use_batch", False)
    
    if use_batch:
        return test_gemini_batch(args, model_type, dataset, output_dir)
    
    # Default pricing for Gemini 2.5 Flash / 1.5 Flash (real-time)
    INPUT_TOKEN_PRICE = 0.15
    OUTPUT_TOKEN_PRICE = 0.50
    
    # Adjust pricing if using Pro models
    if 'pro' in model_type.lower():
        INPUT_TOKEN_PRICE = 1.00
        OUTPUT_TOKEN_PRICE = 6.0
    
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Google GenAI package not installed. Run: pip install google-genai")
    
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try loading from bashrc
        import subprocess
        result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $GEMINI_API_KEY'], 
                            capture_output=True, text=True)
        api_key = result.stdout.strip()
        if not api_key:
            result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $GOOGLE_API_KEY'], 
                                capture_output=True, text=True)
            api_key = result.stdout.strip()
        
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            print("Loaded API key from ~/.bashrc")
        else:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found. "
                "Please set it in ~/.bashrc or export it in terminal"
            )
    
    client = genai.Client(api_key=api_key)
    
    # load dataset
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a subset
    sample_limit = args.get("sample_limit", None)
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)
    
    # prepare data
    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    
    # build prompt
    candidate_labels = class_names
    classes_text = ", ".join(candidate_labels)
    prompt_template = args.get("prompt_template", 
        "Classify this image into one of the following categories: {classes}. "
        "Respond with ONLY the category name, nothing else.")
    
    fmt_map = {"classes": classes_text}
    user_prompt = prompt_template.format_map(SafeFormatDict(fmt_map))
    
    # Get parallel processing settings
    max_workers = args.get("max_workers", 1)
    use_parallel = max_workers > 1
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Thread-safe lock for updating token counts
    token_lock = Lock()
    
    def process_single_image(image_path):
        """Process a single image and return prediction results."""
        # Read image bytes
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Determine MIME type from file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(ext, 'image/jpeg')
        
        # Call Gemini API
        response = client.models.generate_content(
            model=model_type,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                user_prompt
            ]
        )
        
        # Parse response
        prediction = response.text
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        # Fuzzy matching to find the predicted class
        predicted_class, match_score, matched_label = fuzzy_match_label(
            prediction, candidate_labels, threshold=0.6
        )
        
        # If no match found, keep as None
        if predicted_class is None:
            match_score = 0.0
        
        # create one-hot encoded probabilities
        probs = [0.0] * len(candidate_labels)
        if predicted_class is not None:
            probs[predicted_class] = 1.0
        
        return predicted_class, probs, input_tokens, output_tokens, prediction, match_score
    
    # Process images
    if use_parallel:
        print(f"Processing images with {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_image, path): idx 
                           for idx, path in enumerate(paths)}
            
            # Collect results as they complete
            results = [None] * len(paths)
            
            for future in tqdm(as_completed(future_to_idx), total=len(paths), desc="Testing"):
                idx = future_to_idx[future]
                try:
                    predicted_class, probs, input_tokens, output_tokens, prediction, match_score = future.result()
                    results[idx] = (predicted_class, probs, prediction, match_score)
                    
                    # Thread-safe token counting
                    with token_lock:
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        
                    # Print warning for non-matches
                    if predicted_class is None:
                        print(f"\nWARNING: No match found for: '{prediction}'")
                        
                except Exception as e:
                    print(f"\nWarning: Error processing image {idx}: {e}")
                    results[idx] = (None, [0.0] * len(candidate_labels), "", 0.0)
            
            # Extract predictions in correct order
            for predicted_class, probs, prediction, match_score in results:
                preds_ids.append(predicted_class)
                probs_rows.append(probs)
                generated_texts.append(prediction)
                match_scores.append(match_score)
    
    else:
        # Sequential processing
        print("Processing images sequentially...")
        
        for image_path in tqdm(paths, desc="Testing"):
            try:
                predicted_class, probs, input_tokens, output_tokens, prediction, match_score = process_single_image(image_path)
                
                preds_ids.append(predicted_class)
                probs_rows.append(probs)
                generated_texts.append(prediction)
                match_scores.append(match_score)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Print warning for non-matches
                if predicted_class is None:
                    print(f"\nWARNING: No match found for: '{prediction}'")
                
            except Exception as e:
                print(f"\nWarning: Error processing {image_path}: {e}")
                preds_ids.append(None)
                probs_rows.append([0.0] * len(candidate_labels))
                generated_texts.append("")
                match_scores.append(0.0)
    
    # calculate costs
    total_price = (total_input_tokens * INPUT_TOKEN_PRICE + total_output_tokens * OUTPUT_TOKEN_PRICE) / 1e6
    print("\nToken Usage Summary:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total cost: ${total_price:.6f}")
    
    # save token usage to JSON file
    token_usage = {
        "model": model_type,
        "dataset": dataset,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_token_price_per_million": INPUT_TOKEN_PRICE,
        "output_token_price_per_million": OUTPUT_TOKEN_PRICE,
        "total_cost_usd": round(total_price, 6),
        "num_images": len(df)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    token_usage_path = os.path.join(output_dir, "token_usage.json")
    with open(token_usage_path, 'w') as f:
        json.dump(token_usage, f, indent=2)
    print(f"Token usage saved to: {token_usage_path}")
    
    # save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores
    )

def test_gemini_batch(args: dict, model_type: str, dataset: str, output_dir: str):
    """
    Test using Google Gemini Batch API for image classification.
    
    Batch API offers 50% cost savings but requires:
    1. Uploading images to Google Cloud Storage
    2. Submitting batch job
    3. Polling for completion
    4. Downloading results
    
    Batch Pricing (50% discount):
    - Gemini 2.5 Flash: $0.0375/1M input tokens, $0.15/1M output tokens
    - Gemini 1.5 Flash: $0.0375/1M input tokens, $0.15/1M output tokens
    - Gemini 1.5 Pro: $0.625/1M input tokens, $5/1M output tokens
    """
    import time
    import tempfile
    
    # Default pricing for Gemini 2.5 Flash / 1.5 Flash (batch - 50% off)
    INPUT_TOKEN_PRICE = 0.15
    OUTPUT_TOKEN_PRICE = 1.25
    
    # Adjust pricing if using Pro models
    if 'pro' in model_type.lower():
        INPUT_TOKEN_PRICE = 1.00
        OUTPUT_TOKEN_PRICE = 6.0
    
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Google GenAI package not installed. Run: pip install google-genai")
    
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try loading from bashrc
        import subprocess
        result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $GEMINI_API_KEY'], 
                            capture_output=True, text=True)
        api_key = result.stdout.strip()
        if not api_key:
            result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $GOOGLE_API_KEY'], 
                                capture_output=True, text=True)
            api_key = result.stdout.strip()
        
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            print("Loaded API key from ~/.bashrc")
        else:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not found. "
                "Please set it in ~/.bashrc or export it in terminal"
            )
    
    client = genai.Client(api_key=api_key)
    
    # load dataset
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a subset
    sample_limit = args.get("sample_limit", None)
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)
    
    # prepare data
    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    
    # build prompt
    candidate_labels = class_names
    classes_text = ", ".join(candidate_labels)
    prompt_template = args.get("prompt_template", 
        "Classify this image into one of the following categories: {classes}. "
        "Respond with ONLY the category name, nothing else.")
    
    fmt_map = {"classes": classes_text}
    user_prompt = prompt_template.format_map(SafeFormatDict(fmt_map))
    
    print("=" * 80)
    print("GEMINI BATCH API MODE")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_type}")
    print(f"Images to process: {len(df)}")
    print(f"Cost savings: 50% off standard pricing")
    print("=" * 80)
    print()
    
    # Step 1: Upload images to Files API
    print("Step 1/4: Uploading images to Google Files API...")
    paths = df["image_path"].tolist()
    uploaded_files = []
    
    for idx, image_path in enumerate(tqdm(paths, desc="Uploading")):
        try:
            # Determine MIME type
            ext = os.path.splitext(image_path)[1].lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(ext, 'image/jpeg')
            
            # Upload file - file parameter expects the filename as a string
            uploaded_file = client.files.upload(
                file=image_path,
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=f"{dataset}_{idx}"
                )
            )
            
            uploaded_files.append({
                'file_uri': uploaded_file.uri,
                'mime_type': mime_type,
                'image_path': image_path,
                'index': idx
            })
            
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to upload {image_path}: {e}")
            uploaded_files.append({
                'file_uri': None,
                'mime_type': None,
                'image_path': image_path,
                'index': idx
            })
    
    print(f"✓ Uploaded {sum(1 for f in uploaded_files if f['file_uri'] is not None)}/{len(paths)} images")
    print()
    
    # Step 2: Create batch requests JSONL
    print("Step 2/4: Creating batch job...")
    
    batch_requests = []
    for file_info in uploaded_files:
        if file_info['file_uri'] is None:
            continue
        
        request = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [
                        {'fileData': {'fileUri': file_info['file_uri'], 'mimeType': file_info['mime_type']}},
                        {'text': user_prompt}
                    ]
                }
            ]
        }
        batch_requests.append({
            'request': request,
            'custom_id': str(file_info['index'])
        })
    
    # Write batch requests to JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        batch_file_path = f.name
        for req in batch_requests:
            json.dump(req, f)
            f.write('\n')
    
    print(f"✓ Created batch file with {len(batch_requests)} requests")
    print()
    
    # Step 3: Submit batch job
    print("Step 3/4: Submitting batch job to Gemini...")
    
    try:
        # Upload batch file - file parameter expects the filename as a string
        batch_upload = client.files.upload(
            file=batch_file_path,
            config=types.UploadFileConfig(
                mime_type='jsonl',
                display_name=f"{dataset}_batch_{int(time.time())}"
            )
        )
        
        # Create batch job using the uploaded file's name
        batch_job = client.batches.create(
            model=model_type,
            src=batch_upload.name,
            config={
                'display_name': f"{dataset}_{model_type}_{int(time.time())}"
            }
        )
        
        batch_job_name = batch_job.name
        print(f"✓ Batch job submitted: {batch_job_name}")
        print()
        print("=" * 80)
        print("BATCH JOB COMMANDS")
        print("=" * 80)
        print(f"Monitor: client.batches.get(name='{batch_job_name}')")
        print(f"Cancel:  client.batches.cancel(name='{batch_job_name}')")
        print(f"Delete:  client.batches.delete(name='{batch_job_name}')")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"❌ Error submitting batch job: {e}")
        # Clean up
        os.unlink(batch_file_path)
        raise
    
    # Step 4: Poll for completion
    print("Step 4/4: Waiting for batch job to complete...")
    print("This may take several minutes to hours depending on queue size.")
    print()
    
    poll_interval = args.get("batch_poll_interval", 60)  # seconds
    max_wait_time = args.get("batch_max_wait", 3600 * 24)  # 24 hours default
    
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > max_wait_time:
            print(f"⚠️  Timeout: Batch job did not complete within {max_wait_time/3600:.1f} hours")
            print(f"   Job name: {batch_job_name}")
            print(f"   Check status later with: client.batches.get(name='{batch_job_name}')")
            break
        
        # Get job status
        batch_job = client.batches.get(name=batch_job_name)
        status = batch_job.state
        
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
            last_status = status
        
        if status == 'SUCCEEDED':
            print(f"✓ Batch job completed in {elapsed/60:.1f} minutes")
            print()
            break
        elif status in ['FAILED', 'CANCELLED']:
            print(f"❌ Batch job {status.lower()}")
            if hasattr(batch_job, 'error'):
                print(f"   Error: {batch_job.error}")
            raise RuntimeError(f"Batch job {status.lower()}")
        
        # Wait before next poll
        time.sleep(poll_interval)
    
    # Download and process results
    print("Processing results...")
    
    # Download output file
    output_uri = batch_job.output_uri
    output_file = client.files.get(name=output_uri.split('/')[-1])
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file_path = f.name
    
    client.files.download(name=output_file.name, path=output_file_path)
    
    # Parse results
    results_by_index = {}
    total_input_tokens = 0
    total_output_tokens = 0
    
    with open(output_file_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            idx = int(result['custom_id'])
            
            # Extract response
            if 'response' in result and 'candidates' in result['response']:
                candidates = result['response']['candidates']
                if len(candidates) > 0:
                    prediction = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    
                    # Extract token usage
                    usage = result['response'].get('usageMetadata', {})
                    input_tokens = usage.get('promptTokenCount', 0)
                    output_tokens = usage.get('candidatesTokenCount', 0)
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    results_by_index[idx] = {
                        'prediction': prediction,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
    
    # Process predictions
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    
    for idx in range(len(df)):
        if idx in results_by_index:
            result = results_by_index[idx]
            prediction = result['prediction']
            
            # Fuzzy matching
            predicted_class, match_score, matched_label = fuzzy_match_label(
                prediction, candidate_labels, threshold=0.6
            )
            
            if predicted_class is None:
                match_score = 0.0
            
            # Create one-hot probabilities
            probs = [0.0] * len(candidate_labels)
            if predicted_class is not None:
                probs[predicted_class] = 1.0
            
            preds_ids.append(predicted_class)
            probs_rows.append(probs)
            generated_texts.append(prediction)
            match_scores.append(match_score)
        else:
            # Missing result
            preds_ids.append(None)
            probs_rows.append([0.0] * len(candidate_labels))
            generated_texts.append("")
            match_scores.append(0.0)
    
    # Calculate costs (with 50% discount)
    total_price = (total_input_tokens * INPUT_TOKEN_PRICE + total_output_tokens * OUTPUT_TOKEN_PRICE) / 1e6
    standard_price = total_price * 2  # What it would cost without batch discount
    savings = standard_price - total_price
    
    print("\nToken Usage Summary (Batch API):")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Batch cost: ${total_price:.6f}")
    print(f"  Standard cost: ${standard_price:.6f}")
    print(f"  Savings (50%): ${savings:.6f}")
    
    # Save token usage
    token_usage = {
        "model": model_type,
        "dataset": dataset,
        "batch_mode": True,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_token_price_per_million": INPUT_TOKEN_PRICE,
        "output_token_price_per_million": OUTPUT_TOKEN_PRICE,
        "batch_cost_usd": round(total_price, 6),
        "standard_cost_usd": round(standard_price, 6),
        "savings_usd": round(savings, 6),
        "num_images": len(df),
        "batch_job_name": batch_job_name
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    token_usage_path = os.path.join(output_dir, "token_usage.json")
    with open(token_usage_path, 'w') as f:
        json.dump(token_usage, f, indent=2)
    print(f"Token usage saved to: {token_usage_path}")
    
    # Clean up temp files
    os.unlink(batch_file_path)
    os.unlink(output_file_path)
    
    # Save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores
    )


def test_claude(args: dict, model_type: str, dataset: str, output_dir: str):
    """
    Test using Anthropic Claude API for image classification.
    
    Supports Claude models with vision capabilities:
    - Claude Sonnet 4.5: Best balance of intelligence, speed, and cost
    - Claude Haiku 4.5: Fastest model with near-frontier intelligence
    - Claude Opus 4.5: Maximum intelligence with practical performance
    
    Pricing (as of Dec 2024):
    - Claude Sonnet 4.5: $3/1M input tokens, $15/1M output tokens
    - Claude Haiku 4.5: $1/1M input tokens, $5/1M output tokens
    - Claude Opus 4.5: $5/1M input tokens, $25/1M output tokens
    
    Note: Images are tokenized as (width * height) / 750 tokens.
          Images larger than 1568px will be resized automatically.
    """
    
    # Pricing for different Claude models
    pricing_map = {
        'claude-sonnet-4-5': {'input': 3.0, 'output': 15.0},
        'claude-haiku-4-5': {'input': 1.0, 'output': 5.0},
        'claude-opus-4-5': {'input': 5.0, 'output': 25.0},
    }
    
    # Default to Sonnet pricing
    pricing = pricing_map.get(model_type, {'input': 1.0, 'output': 5.0})
    INPUT_TOKEN_PRICE = pricing['input']
    OUTPUT_TOKEN_PRICE = pricing['output']
    
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        # Try loading from bashrc
        import subprocess
        result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $ANTHROPIC_API_KEY'], 
                            capture_output=True, text=True)
        api_key = result.stdout.strip()
        if not api_key:
            result = subprocess.run(['bash', '-c', 'source ~/.bashrc && echo $CLAUDE_API_KEY'], 
                                capture_output=True, text=True)
            api_key = result.stdout.strip()
        
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            print("Loaded API key from ~/.bashrc")
        else:
            raise ValueError(
                "ANTHROPIC_API_KEY or CLAUDE_API_KEY not found. "
                "Please set it in ~/.bashrc or export it in terminal"
            )
    
    client = Anthropic(api_key=api_key)
    
    # load dataset
    dataset_path = load_agml_dataset(dataset)
    df = agml_to_df(os.path.join(dataset_path, "val"))
    
    # if sample limit is set, take a subset
    sample_limit = args.get("sample_limit", None)
    if sample_limit and 0 < sample_limit < 1:
        df = df.sample(frac=sample_limit, random_state=42).reset_index(drop=True)
    
    # prepare data
    class_names = sorted(df["label"].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(class_names)}
    y_true = df["label"].map(class_to_id).to_numpy()
    
    # build prompt
    candidate_labels = class_names
    classes_text = ", ".join(candidate_labels)
    prompt_template = args.get("prompt_template", 
        "Classify this image into one of the following categories: {classes}. "
        "Respond with ONLY the category name, nothing else.")
    
    fmt_map = {"classes": classes_text}
    user_prompt = prompt_template.format_map(SafeFormatDict(fmt_map))
    
    # Get parallel processing settings with rate limiting
    max_workers = args.get("max_workers", 1)
    
    # Claude rate limit: 50 requests/minute
    # Calculate safe rate based on number of workers and total images
    RATE_LIMIT_RPM = 50  # requests per minute
    num_images = len(df)
    
    # Auto-adjust workers to stay under rate limit
    # Leave some headroom (use 80% of limit to be safe)
    safe_rate = int(RATE_LIMIT_RPM * 0.8)  # 40 requests/minute
    
    if max_workers > safe_rate:
        print(f"⚠️  Reducing max_workers from {max_workers} to {safe_rate} to comply with Claude rate limit (50 req/min)")
        max_workers = safe_rate
    
    # Calculate delay between batches to stay under rate limit
    # Process in batches of max_workers, with delays between batches
    requests_per_batch = max_workers
    delay_between_batches = 60.0 / safe_rate * requests_per_batch  # seconds
    
    use_parallel = max_workers > 1
    
    print(f"Rate limiting: {safe_rate} requests/minute (Claude limit: {RATE_LIMIT_RPM}/min)")
    print(f"Processing strategy: {max_workers} workers, {delay_between_batches:.1f}s delay between batches")
    print(f"Estimated time: {(num_images / safe_rate):.1f} minutes for {num_images} images")
    print()
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    generated_texts = []
    match_scores = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Thread-safe lock for updating token counts
    token_lock = Lock()
    
    def process_single_image(image_path):
        """Process a single image and return prediction results."""
        # encode image (converts TIFF to PNG if needed)
        base64_image, mime_type = encode_image_base64(image_path)
        
        # Call Claude API
        response = client.messages.create(
            model=model_type,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ],
                }
            ],
        )
        
        # Parse response
        prediction = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # Fuzzy matching to find the predicted class
        predicted_class, match_score, matched_label = fuzzy_match_label(
            prediction, candidate_labels, threshold=0.6
        )
        
        # If no match found, keep as None
        if predicted_class is None:
            match_score = 0.0
        
        # create one-hot encoded probabilities
        probs = [0.0] * len(candidate_labels)
        if predicted_class is not None:
            probs[predicted_class] = 1.0
        
        return predicted_class, probs, input_tokens, output_tokens, prediction, match_score
    
    # Process images with rate limiting
    if use_parallel:
        import time
        print(f"Processing images with {max_workers} parallel workers (rate-limited)...")
        
        # Process in batches to control rate
        results = [None] * len(paths)
        
        for batch_start in range(0, len(paths), requests_per_batch):
            batch_end = min(batch_start + requests_per_batch, len(paths))
            batch_paths = paths[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch
                future_to_idx = {executor.submit(process_single_image, path): idx 
                               for path, idx in zip(batch_paths, batch_indices)}
                
                # Collect batch results
                for future in tqdm(as_completed(future_to_idx), 
                                 total=len(batch_paths), 
                                 desc=f"Batch {batch_start//requests_per_batch + 1}/{(len(paths) + requests_per_batch - 1)//requests_per_batch}",
                                 leave=False):
                    idx = future_to_idx[future]
                    try:
                        predicted_class, probs, input_tokens, output_tokens, prediction, match_score = future.result()
                        results[idx] = (predicted_class, probs, prediction, match_score)
                        
                        # Thread-safe token counting
                        with token_lock:
                            total_input_tokens += input_tokens
                            total_output_tokens += output_tokens
                            
                        # Print warning for non-matches
                        if predicted_class is None:
                            print(f"\nWARNING: No match found for: '{prediction}'")
                            
                    except Exception as e:
                        print(f"\nWarning: Error processing image {idx}: {e}")
                        results[idx] = (None, [0.0] * len(candidate_labels), "", 0.0)
            
            # Delay between batches (except after last batch)
            if batch_end < len(paths):
                print(f"  Waiting {delay_between_batches:.1f}s before next batch (rate limiting)...")
                time.sleep(delay_between_batches)
        
        # Extract predictions in correct order
        for predicted_class, probs, prediction, match_score in results:
            preds_ids.append(predicted_class)
            probs_rows.append(probs)
            generated_texts.append(prediction)
            match_scores.append(match_score)
    
    else:
        # Sequential processing with rate limiting
        import time
        print("Processing images sequentially (rate-limited)...")
        
        # For sequential, just add delay between requests
        delay_per_request = 60.0 / safe_rate  # seconds per request
        
        for idx, image_path in enumerate(tqdm(paths, desc="Testing")):
            try:
                predicted_class, probs, input_tokens, output_tokens, prediction, match_score = process_single_image(image_path)
                
                preds_ids.append(predicted_class)
                probs_rows.append(probs)
                generated_texts.append(prediction)
                match_scores.append(match_score)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Print warning for non-matches
                if predicted_class is None:
                    print(f"\nWARNING: No match found for: '{prediction}'")
                
                # Rate limit delay (except for last image)
                if idx < len(paths) - 1:
                    time.sleep(delay_per_request)
                
            except Exception as e:
                print(f"\nWarning: Error processing {image_path}: {e}")
                preds_ids.append(None)
                probs_rows.append([0.0] * len(candidate_labels))
                generated_texts.append("")
                match_scores.append(0.0)
    
    # calculate costs
    total_price = (total_input_tokens * INPUT_TOKEN_PRICE + total_output_tokens * OUTPUT_TOKEN_PRICE) / 1e6
    print("\nToken Usage Summary:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total cost: ${total_price:.6f}")
    print(f"  Model: {model_type}")
    
    # save token usage to JSON file
    token_usage = {
        "model": model_type,
        "dataset": dataset,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_token_price_per_million": INPUT_TOKEN_PRICE,
        "output_token_price_per_million": OUTPUT_TOKEN_PRICE,
        "total_cost_usd": round(total_price, 6),
        "num_images": len(df)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    token_usage_path = os.path.join(output_dir, "token_usage.json")
    with open(token_usage_path, 'w') as f:
        json.dump(token_usage, f, indent=2)
    print(f"Token usage saved to: {token_usage_path}")
    
    # save metrics
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir,
        generated_texts=generated_texts,
        match_scores=match_scores
    )
