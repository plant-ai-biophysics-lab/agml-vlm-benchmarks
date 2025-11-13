import os
import base64
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tasks.classification import load_agml_dataset, agml_to_df
from utils.utils import save_classification_results

def encode_image_base64(image_path: str) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_openai(args: dict, model_type: str, dataset: str, output_dir: str):
    
    INPUT_TOKEN_PRICE = 1.25
    OUTPUT_TOKEN_PRICE = 10.0

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
    user_prompt = prompt_template.format(classes=classes_text)
    
    # Get parallel processing settings
    max_workers = args.get("max_workers", 1)  # Default to sequential (1 worker)
    use_parallel = max_workers > 1
    
    # run predictions
    paths = df["image_path"].tolist()
    preds_ids = []
    probs_rows = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Thread-safe lock for updating token counts
    token_lock = Lock()
    
    def process_single_image(image_path):
        """Process a single image and return prediction results."""
        # encode image
        base64_image = encode_image_base64(image_path)
        
        # call API
        response = client.responses.create(
            model=model_type,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt
                        },
                        {
                            "type": "input_image",
                           "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ]
                }
            ]
        )
        
        # parse response
        prediction = response.output_text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # match to candidate labels (case-insensitive)
        pred_id = -1
        for i, label in enumerate(candidate_labels):
            if label.lower() in prediction.lower():
                pred_id = i
                break
        
        # if no match found, try exact match
        if pred_id == -1 and prediction in candidate_labels:
            pred_id = candidate_labels.index(prediction)
        
        # if still no match, default to first class
        if pred_id == -1:
            pred_id = 0
        
        # create uniform probability distribution
        probs = [0.0] * len(candidate_labels)
        probs[pred_id] = 1.0
        
        return pred_id, probs, input_tokens, output_tokens, prediction
    
    # Process images
    if use_parallel:
        print(f"Processing images with {max_workers} parallel workers...")
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_single_image, path): idx 
                           for idx, path in enumerate(paths)}
            
            # Collect results as they complete
            results = [None] * len(paths)  # Preserve order
            
            for future in tqdm(as_completed(future_to_idx), total=len(paths), desc="Testing"):
                idx = future_to_idx[future]
                try:
                    pred_id, probs, input_tokens, output_tokens, prediction = future.result()
                    results[idx] = (pred_id, probs)
                    
                    # Thread-safe token counting
                    with token_lock:
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        
                except Exception as e:
                    print(f"\nWarning: Error processing image {idx}: {e}")
                    # Default to first class on error
                    results[idx] = (0, [1.0] + [0.0] * (len(candidate_labels) - 1))
            
            # Extract predictions in correct order
            for pred_id, probs in results:
                preds_ids.append(pred_id)
                probs_rows.append(probs)
    
    else:
        # Sequential processing (original implementation)
        print("Processing images sequentially...")
        
        for image_path in tqdm(paths, desc="Testing"):
            try:
                pred_id, probs, input_tokens, output_tokens, prediction = process_single_image(image_path)
                
                preds_ids.append(pred_id)
                probs_rows.append(probs)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
            except Exception as e:
                print(f"\nWarning: Error processing {image_path}: {e}")
                # Default to first class on error
                preds_ids.append(0)
                probs_rows.append([1.0] + [0.0] * (len(candidate_labels) - 1))
            
    
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
    save_classification_results(
        candidate_labels,
        preds_ids,
        probs_rows,
        df,
        y_true,
        output_dir
    )