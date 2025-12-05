"""
LLM Judge for semantic label matching in agricultural classification.

Confidence Scores:
- 0: Very unsure about the judgment
- 1: Uncertain / could possibly be the same
- 2: Very confident about the judgment
"""

import os
import json
import pandas as pd

from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

if TYPE_CHECKING:
    import pandas as pd

@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    
    confidence: int  # 0, 1, or 2
    reasoning: str
    is_match: bool  # True if confidence >= threshold
    raw_response: str

class LLMJudge:
    """
    LLM-based judge for semantic label matching.
    """
    
    JUDGE_PROMPT_TEMPLATE = """You are an expert agricultural scientist evaluating plant disease, pest, and weed classifications.

    Your task is to determine if two labels refer to the SAME plant condition, disease, pest, or species, even if they use different terminology.

    Ground Truth Label: "{ground_truth}"
    Predicted Label: "{predicted}"

    Consider:
    - Are these referring to the same disease/pest/species?
    - Could these be different names for the same condition?
    - Are they synonyms or related terms in agriculture/botany?
    - Could differences be due to regional naming conventions?

    Respond ONLY with valid JSON in this exact format (no other text before or after):
    {{
        "is_match": true,
        "reasoning": "Brief explanation here",
        "confidence": 2
    }}

    Confidence levels:
    - 0: Very unsure about your judgment
    - 1: Somewhat confident / could possibly be the same
    - 2: Very confident about your judgment

    JSON response:"""

    def __init__(
        self,
        model_name: str = "gpt-oss-20b",
        api_provider: str = "openai",
        confidence_threshold: int = 1,
        max_workers: int = 10,
        context_info: Optional[str] = None,
        device: str = "auto",
        reasoning_level: str = "medium"
    ):
        """
        Initialize LLM Judge.
        
        Args:
            model_name: Name of the LLM model to use
            api_provider: API provider ("openai", "anthropic", or "hf" for Hugging Face)
            confidence_threshold: Minimum confidence (0-2) to consider a match
            max_workers: Number of parallel API calls (for API) or batch size (for local)
            context_info: Optional context about the classification task
            device: Device for local models ("auto", "cuda", "cpu")
            reasoning_level: Reasoning effort for gpt-oss models ("low", "medium", "high")
        """
        self.model_name = model_name
        self.api_provider = api_provider.lower()
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self.context_info = context_info
        self.device = device
        self.reasoning_level = reasoning_level.lower()
        
        # initialize API client or load local model
        if self.api_provider == "openai":
            self._init_openai()
        elif self.api_provider == "hf":
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try loading from bashrc
            import subprocess
            result = subprocess.run(
                ['bash', '-c', 'source ~/.bashrc && echo $OPENAI_API_KEY'],
                capture_output=True, text=True
            )
            api_key = result.stdout.strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
    
    def _init_huggingface(self):
        """Initialize Hugging Face local model (e.g., gpt-oss-120b)."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")
        
        print(f"Loading local model: {self.model_name}")
        print("This may take a few minutes on first load...")
        
        # Load model with pipeline
        self.client = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            # attn_implementation="kernels-community/vllm-flash-attn3"
        )
        
        print(f"Model loaded successfully on device: {self.device}")
    
    def _create_judge_prompt(self, ground_truth: str, predicted: str) -> str:
        """Create prompt for the LLM judge."""
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            ground_truth=ground_truth,
            predicted=predicted
        )
        
        if self.context_info:
            prompt = f"Context: {self.context_info}\n\n" + prompt
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API or local model and return response."""
        
        if self.api_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural scientist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # deterministic for consistency
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        
        elif self.api_provider == "hf":
            # for Hugging Face models (like gpt-oss-120b)
            # add reasoning level to system prompt for gpt-oss models
            system_content = "You are an expert agricultural scientist."
            if "gpt-oss" in self.model_name.lower():
                system_content += f" Reasoning: {self.reasoning_level}."
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            outputs = self.client(
                messages,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False, # deterministic
            )
            
            # extract the generated text from the last message
            return outputs[0]["generated_text"][-1]["content"].strip()
    
    def _parse_response(self, response) -> Dict:
        """Parse JSON response from LLM.
        
        Args:
            response: Can be a string or a structured output from pipeline
        """
        # If response is a list/dict structure from pipeline, extract the text
        if isinstance(response, list):
            # Handle [{'generated_text': [...]}] structure
            if len(response) > 0 and isinstance(response[0], dict) and 'generated_text' in response[0]:
                gen_text = response[0]['generated_text']
                if isinstance(gen_text, list) and len(gen_text) > 0:
                    # Extract last message content
                    last_msg = gen_text[-1]
                    if isinstance(last_msg, dict) and 'content' in last_msg:
                        response = last_msg['content']
                    else:
                        response = str(gen_text)
                else:
                    response = str(gen_text)
            else:
                response = str(response)
        
        # Ensure it's a string
        response = str(response).strip()
        
        # find JSON object in response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # JSON is malformed - try regex extraction as fallback
                import re
                print(f"JSON parse error: {e}")
                print(f"Attempting regex extraction from: {json_str[:200]}...")
                
                result = {"confidence": 0, "is_match": False, "reasoning": ""}
                
                # Extract confidence (as float or int)
                conf_match = re.search(r'"confidence":\s*([0-9.]+)', json_str)
                if conf_match:
                    try:
                        result["confidence"] = float(conf_match.group(1))
                    except ValueError:
                        pass
                
                # Extract is_match (bool)
                match_match = re.search(r'"(?:is_match|match)":\s*(true|false)', json_str, re.IGNORECASE)
                if match_match:
                    result["is_match"] = match_match.group(1).lower() == 'true'
                
                # Extract answer (yes/no) as alternative to is_match
                answer_match = re.search(r'"answer":\s*"(yes|no)"', json_str, re.IGNORECASE)
                if answer_match:
                    result["is_match"] = answer_match.group(1).lower() == 'yes'
                
                # Extract reasoning - this is trickier due to unescaped quotes
                # Get everything between "reasoning":" and the next " that's followed by , or }
                reason_match = re.search(r'"reasoning":\s*"([^"]*(?:"[^"]*)*?)"[\s,}]', json_str)
                if not reason_match:
                    # Fallback: get until next field or end
                    reason_match = re.search(r'"reasoning":\s*"(.+?)(?:",\s*"|"\s*})', json_str, re.DOTALL)
                if reason_match:
                    result["reasoning"] = reason_match.group(1).strip()
                else:
                    result["reasoning"] = "Could not extract reasoning from malformed JSON"
                
                print(f"Regex extraction result: confidence={result['confidence']}, is_match={result['is_match']}")
                return result
        
        # fallback if no JSON found
        print(f"No JSON found in response: {response[:200]}...")
        return {
            "confidence": 0,
            "reasoning": "No valid JSON in LLM response",
            "is_match": False
        }
    
    def judge_single(
        self,
        ground_truth: str,
        predicted: str
    ) -> JudgeResult:
        """
        Judge a single prediction against ground truth.
        
        Args:
            ground_truth: The true label
            predicted: The predicted label
            
        Returns:
            JudgeResult with confidence score and reasoning
        """
        # create prompt
        prompt = self._create_judge_prompt(ground_truth, predicted)
        
        # call LLM
        raw_response = self._call_llm(prompt)
        
        # parse response
        parsed = self._parse_response(raw_response)
        
        confidence = parsed.get("confidence", 0)
        reasoning = parsed.get("reasoning", "No reasoning provided")
        llm_says_match = parsed.get("is_match", False)
        
        # determine if it's a match: LLM must say it matches AND meet confidence threshold
        is_match = llm_says_match and (confidence >= self.confidence_threshold)
        
        # extract just the JSON portion for cleaner storage
        # (the full response may have reasoning tokens before the JSON)
        clean_response = raw_response
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_response = raw_response[start_idx:end_idx + 1]
        
        return JudgeResult(
            confidence=confidence,
            reasoning=reasoning,
            is_match=is_match,
            raw_response=clean_response
        )
    
    def judge_batch(
        self,
        ground_truths: List[str],
        predictions: List[str],
        show_progress: bool = True,
        batch_size: int = 4
    ) -> List[JudgeResult]:
        """
        Judge multiple predictions with batching support for local models.
        
        Args:
            ground_truths: List of true labels
            predictions: List of predicted labels
            show_progress: Whether to show progress bar
            batch_size: Batch size for local model inference (ignored for API)
            
        Returns:
            List of JudgeResults
        """
        if len(ground_truths) != len(predictions):
            raise ValueError("ground_truths and predictions must have same length")
        
        # for local models, use batched inference
        if self.api_provider == "hf":
            results = []
            n_samples = len(ground_truths)
            
            # create all prompts upfront
            prompts = [self._create_judge_prompt(gt, pred) 
                      for gt, pred in zip(ground_truths, predictions)]
            
            # process in batches
            iterator = range(0, n_samples, batch_size)
            if show_progress:
                iterator = tqdm(list(iterator), desc="LLM Judge (batched)")
            
            for i in iterator:
                batch_prompts = prompts[i:i + batch_size]
                
                try:
                    # prepare batch messages
                    system_content = "You are an expert agricultural scientist."
                    if "gpt-oss" in self.model_name.lower():
                        system_content += f" Reasoning: {self.reasoning_level}."
                    
                    batch_messages = [
                        [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": prompt}
                        ]
                        for prompt in batch_prompts
                    ]
                    
                    # batch inference
                    outputs = self.client(
                        batch_messages,
                        max_new_tokens=300,
                        temperature=0.0,
                        do_sample=False,
                        batch_size=len(batch_messages)
                    )
                    
                    # parse each output using the SAME logic as _call_llm
                    for idx, output in enumerate(outputs):
                        # Print debug info for the first batch only
                        if i == 0 and idx == 0:
                            print(f"[judge_batch debug] Example output type: {type(output)}")
                            print(f"[judge_batch debug] Example output preview: {repr(output)[:500]}")
                        
                        # Extract response using same method as _call_llm for consistency
                        try:
                            raw_response = output[0]["generated_text"][-1]["content"].strip()
                        except (KeyError, IndexError, TypeError) as e:
                            print(f"Warning: Could not extract content using standard method: {e}")
                            print(f"Output type: {type(output)}, preview: {repr(output)[:300]}")
                            # Fallback to empty response
                            raw_response = ""

                        parsed = self._parse_response(raw_response)
                        confidence = parsed.get("confidence", 0)
                        reasoning = parsed.get("reasoning", "No reasoning provided")
                        llm_says_match = parsed.get("is_match", False)
                        is_match = llm_says_match and (confidence >= self.confidence_threshold)
                        # extract clean JSON
                        clean_response = raw_response
                        start_idx = raw_response.find('{')
                        end_idx = raw_response.rfind('}')
                        if start_idx != -1 and end_idx != -1:
                            clean_response = raw_response[start_idx:end_idx + 1]
                        results.append(JudgeResult(
                            confidence=confidence,
                            reasoning=reasoning,
                            is_match=is_match,
                            raw_response=clean_response
                        ))
                        
                except Exception as e:
                    print(f"Error judging batch: {e}")
                    # add error results for each item in batch
                    for _ in batch_prompts:
                        results.append(JudgeResult(
                            confidence=0,
                            reasoning=f"Error: {str(e)}",
                            is_match=False,
                            raw_response=""
                        ))
            
            return results
        
        # for API providers, use parallel processing
        results = [None] * len(ground_truths)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.judge_single, gt, pred): idx
                for idx, (gt, pred) in enumerate(zip(ground_truths, predictions))
            }
            
            # collect results with progress bar
            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=len(future_to_idx), desc="LLM Judge")
            
            for future in iterator:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error judging pair {idx}: {e}")

                    results[idx] = JudgeResult(
                        confidence=0,
                        reasoning=f"Error: {str(e)}",
                        is_match=False,
                        raw_response=""
                    )
        
        return results
    
    def evaluate_predictions(
        self,
        predictions_csv: str,
        output_dir: str = None,
        skip_completed: bool = True
    ) -> Dict:
        """
        Evaluate predictions from a CSV file and save results.
        
        Args:
            predictions_csv: Path to predictions.csv from model evaluation
            output_dir: Directory to save judge results (defaults to same as predictions)
            skip_completed: If True, skip evaluation if judge results already exist
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        # get output directory
        if output_dir is None:
            output_dir = os.path.dirname(predictions_csv)
        
        # check if already completed
        judge_metrics_json = os.path.join(output_dir, "judge_metrics.json")
        if skip_completed and os.path.exists(judge_metrics_json):
            print(f"Skipping - already evaluated: {output_dir}")
            with open(judge_metrics_json, 'r') as f:
                return json.load(f)
        
        # load predictions
        df = pd.read_csv(predictions_csv)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # determine which column to use for predictions
        # if pred_label is mostly empty, use generated_text instead
        if 'pred_label' in df.columns:
            pred_label_empty_ratio = df['pred_label'].isna().sum() / len(df)
            use_generated_text = pred_label_empty_ratio > 0.5
        else:
            use_generated_text = True
        
        if use_generated_text and 'generated_text' in df.columns:
            print("Using 'generated_text' column for predictions (pred_label is mostly empty)")
            prediction_col = 'generated_text'
            valid_mask = df['generated_text'].notna()
        else:
            print("Using 'pred_label' column for predictions")
            prediction_col = 'pred_label'
            valid_mask = df['pred_label'].notna()
        
        df_valid = df[valid_mask].copy()
        
        print(f"Evaluating {len(df_valid)} predictions with LLM judge...")
        if (~valid_mask).sum() > 0:
            print(f"Skipping {(~valid_mask).sum()} None predictions")
        
        # judge all predictions
        judge_results = self.judge_batch(
            ground_truths=df_valid['label'].tolist(),
            predictions=df_valid[prediction_col].tolist()
        )
        
        # add judge results to dataframe
        df_valid['judge_confidence'] = [r.confidence for r in judge_results]
        df_valid['judge_reasoning'] = [r.reasoning for r in judge_results]
        df_valid['judge_is_match'] = [r.is_match for r in judge_results]
        df_valid['judge_raw_response'] = [r.raw_response for r in judge_results]
        
        # add corrected prediction column (ground truth when judge says it matches)
        df_valid['judge_corrected_pred'] = df_valid.apply(
            lambda row: row['label'] if row['judge_is_match'] else row.get('pred_label', row.get('generated_text', '')),
            axis=1
        )
        
        # store which column was used for predictions (for reporting)
        df_valid['_prediction_source'] = prediction_col
        
        # calculate metrics on FULL dataset (including skipped samples)
        # this makes judge accuracy comparable to original zero-shot accuracy
        total_dataset = len(df)
        total_judged = len(df_valid)
        total_skipped = total_dataset - total_judged
        
        # count matches on judged samples
        exact_match = (df_valid['label'] == df_valid.get('pred_label', pd.Series([None]*len(df_valid)))).sum()
        judge_match = df_valid['judge_is_match'].sum()
        
        # confidence score distribution
        confidence_dist = df_valid['judge_confidence'].value_counts().to_dict()
        
        metrics = {
            'total_samples': total_dataset,
            'total_judged': total_judged,
            'total_skipped': total_skipped,
            'exact_match_count': int(exact_match),
            'exact_match_accuracy': float(exact_match / total_dataset),
            'judge_match_count': int(judge_match),
            'judge_match_accuracy': float(judge_match / total_dataset),
            'accuracy_gain': float((judge_match - exact_match) / total_dataset),
            'judge_match_accuracy_on_judged': float(judge_match / total_judged) if total_judged > 0 else 0.0,
            'confidence_threshold': self.confidence_threshold,
            'confidence_distribution': {int(k): int(v) for k, v in confidence_dist.items()},
            'model_name': self.model_name
        }
        
        # save results
        judge_predictions_csv = os.path.join(output_dir, "predictions_with_judge.csv")
        df_valid.to_csv(judge_predictions_csv, index=False)
        
        judge_metrics_json = os.path.join(output_dir, "judge_metrics.json")
        with open(judge_metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # create detailed report
        self._create_report(df_valid, metrics, output_dir)
        
        print("\nLLM Judge Evaluation Complete!")
        print(f"Total samples: {metrics['total_samples']} (judged: {metrics['total_judged']}, skipped: {metrics['total_skipped']})")
        print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})")
        print(f"Judge match accuracy: {metrics['judge_match_accuracy']:.4f} ({metrics['judge_match_count']}/{metrics['total_samples']})")
        print(f"Accuracy gain: {metrics['accuracy_gain']:.4f}")
        print(f"\nResults saved to: {output_dir}")
        
        return metrics
    
    def _create_report(self, df: 'pd.DataFrame', metrics: Dict, output_dir: str):
        """Create detailed evaluation report."""
        report_path = os.path.join(output_dir, "judge_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM JUDGE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model: {metrics['model_name']}\n")
            f.write(f"Confidence Threshold: {metrics['confidence_threshold']}\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n")
            f.write(f"Judged: {metrics['total_judged']}\n")
            f.write(f"Skipped (no prediction): {metrics['total_skipped']}\n\n")
            
            f.write("ACCURACY METRICS (on full dataset)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exact Match:  {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_count']}/{metrics['total_samples']})\n")
            f.write(f"Judge Match:  {metrics['judge_match_accuracy']:.4f} ({metrics['judge_match_count']}/{metrics['total_samples']})\n")
            f.write(f"Gain:         {metrics['accuracy_gain']:.4f}\n")
            if metrics['total_skipped'] > 0:
                f.write(f"\nJudge Match (on judged only): {metrics['judge_match_accuracy_on_judged']:.4f} ({metrics['judge_match_count']}/{metrics['total_judged']})\n")
            f.write("\n")
            
            f.write("CONFIDENCE DISTRIBUTION (of judged samples)\n")
            f.write("-" * 80 + "\n")
            for conf in [0, 1, 2]:
                count = metrics['confidence_distribution'].get(conf, 0)
                pct = count / metrics['total_judged'] * 100 if metrics['total_judged'] > 0 else 0
                f.write(f"Confidence {conf}: {count:4d} ({pct:5.1f}%)\n")
            f.write("\n")
            
            # cases where judge corrected exact match failures
            # for pred_label column, compare directly. For generated_text, all are "mismatches" that got corrected
            if 'pred_label' in df.columns:
                corrected = df[(df['label'] != df['pred_label']) & df['judge_is_match']]
            else:
                corrected = df[df['judge_is_match']]
            
            if len(corrected) > 0:
                f.write("EXAMPLES: Judge Corrected Mismatches\n")
                f.write("-" * 80 + "\n")
                
                # determine which column was used
                uses_generated_text = corrected['_prediction_source'].iloc[0] == 'generated_text' if '_prediction_source' in corrected.columns else False
                
                for _, row in corrected.head(10).iterrows():
                    f.write(f"Ground Truth:      {row['label']}\n")
                    
                    if uses_generated_text and 'generated_text' in row:
                        # show abbreviated generated text
                        gen_text = str(row['generated_text'])[:150]
                        if len(str(row['generated_text'])) > 150:
                            gen_text += "..."
                        f.write(f"Generated Text:    {gen_text}\n")
                    
                    # show original prediction
                    if 'pred_label' in row and pd.notna(row['pred_label']):
                        f.write(f"Original Pred:     {row['pred_label']}\n")
                    
                    f.write(f"Corrected Pred:    {row['judge_corrected_pred']}\n")
                    f.write(f"Confidence:        {row['judge_confidence']}\n")
                    f.write(f"Reasoning:         {row['judge_reasoning']}\n")
                    f.write("\n")
            
            # cases where judge was uncertain (confidence = 1)
            uncertain = df[df['judge_confidence'] == 1]
            if len(uncertain) > 0:
                f.write("\nEXAMPLES: Uncertain Matches (Confidence = 1)\n")
                f.write("-" * 80 + "\n")
                
                uses_generated_text = uncertain['_prediction_source'].iloc[0] == 'generated_text' if '_prediction_source' in uncertain.columns else False
                
                for _, row in uncertain.head(10).iterrows():
                    f.write(f"Ground Truth:      {row['label']}\n")
                    
                    if uses_generated_text and 'generated_text' in row:
                        gen_text = str(row['generated_text'])[:150]
                        if len(str(row['generated_text'])) > 150:
                            gen_text += "..."
                        f.write(f"Generated Text:    {gen_text}\n")
                    
                    if 'pred_label' in row and pd.notna(row['pred_label']):
                        f.write(f"Predicted:         {row['pred_label']}\n")
                    
                    f.write(f"Reasoning:         {row['judge_reasoning']}\n")
                    f.write("\n")
        
        print(f"Detailed report saved to: {report_path}")


def main():
    """Example usage and CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Judge for classification evaluation")
    parser.add_argument("predictions_csv", help="Path to predictions.csv file")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "hf"],
                       help="API provider: openai, anthropic, or hf (Hugging Face)")
    parser.add_argument("--threshold", type=int, default=1, choices=[0, 1, 2],
                       help="Confidence threshold for match (0-2)")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Number of parallel API calls")
    parser.add_argument("--device", default="auto",
                       help="Device for local models: auto, cuda, cpu")
    parser.add_argument("--reasoning-level", default="medium", choices=["low", "medium", "high"],
                       help="Reasoning effort for gpt-oss models: low (fast), medium (balanced), high (detailed)")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--no-skip-completed", action="store_true",
                       help="Re-evaluate even if results already exist (default: skip completed)")
    
    args = parser.parse_args()
    
    # Initialize judge
    judge = LLMJudge(
        model_name=args.model,
        api_provider=args.provider,
        confidence_threshold=args.threshold,
        max_workers=args.max_workers,
        device=args.device,
        reasoning_level=args.reasoning_level
    )
    
    # Evaluate predictions
    judge.evaluate_predictions(
        predictions_csv=args.predictions_csv,
        output_dir=args.output_dir,
        skip_completed=not args.no_skip_completed
    )


if __name__ == "__main__":
    main()
