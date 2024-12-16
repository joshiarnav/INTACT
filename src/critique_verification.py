import json
from datetime import datetime
import logging
import numpy as np
from tqdm import tqdm
from agentic import critique_step #pylint: disable=import-error
from environment import MODEL #pylint: disable=import-error
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from debug_output import save_debug_info #pylint: disable=import-error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_prm800k_data(file_path):
    """Load and parse the PRM800K dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Only include non-quality control and non-screening questions
                if not item.get('is_quality_control_question', False) and \
                   not item.get('is_initial_screening_question', False):
                    data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in dataset")
    return data

def validate_critique(critique_result):
    """Validate that the critique result is in the expected format."""
    if not isinstance(critique_result, str):
        logger.error(f"Unexpected critique result type: {type(critique_result)}")
        return False
    
    critique_lower = critique_result.strip().lower()
    if "correct" not in critique_lower and "incorrect" not in critique_lower:
        logger.error(f"Unexpected critique format: {critique_result}")
        return False
    
    return True

def evaluate_step(args):
    """Evaluate a single step using the critique model."""
    problem, step_data, previous_steps, output_dir = args
    try:
        rating = step_data['rating']
        # Skip steps with rating 0
        if rating == 0:
            return None
        step_text = step_data['text']
        previous_steps_text = previous_steps
        # Get the step text and ground truth rating
        # if step_data['chosen_completion'] is not None:
        #     completion = step_data['completions'][step_data['chosen_completion']]
        #     # Skip if step is flagged
        #     if completion.get('flagged'):
        #         logger.warning("Skipping flagged step")
        #         return None
        #     step_text = completion['text']
        #     rating = int(completion['rating'])
        # elif step_data['human_completion']:
        #     step_text = step_data['human_completion']
        #     rating = 1  # Human completions are considered correct by default
        # else:
        #     return None
            
        # Convert rating to boolean: 1 is correct (True), 0 or -1 is incorrect (False)
        ground_truth = rating == 1
        
        # Log the previous steps
        # logger.info(f"Previous steps: {previous_steps}")
            
        # Get model's judgment
        # Clean and convert previous steps list to list of strings
        # previous_steps_text = []
        # for step in previous_steps:
        #     if isinstance(step, dict):
        #         # Handle dictionary case (e.g., completion or human step)
        #         if 'text' in step:
        #             previous_steps_text.append(step['text'])
        #     elif isinstance(step, str):
        #         # Handle string case
        #         previous_steps_text.append(step)
        #     else:
        #         logger.warning(f"Unexpected step type: {type(step)}")
        #         previous_steps_text.append(str(step))
        
        # Debug: Print input to critique_step
        # logger.info(f"Evaluating step:")
        # logger.info(f"Problem: {problem}")
        # logger.info(f"Step text: {step_text}")
        # logger.info(f"Previous steps: {previous_steps_text}")
        # logger.info(f"Ground truth (rating={rating}): {ground_truth}")
        
        critique_result, total_tokens = critique_step(
            problem=problem,
            step=step_text,
            previous_steps=previous_steps_text,
            temperature=0.7
        )
        
        # Validate critique result
        if not validate_critique(critique_result):
            logger.error("Invalid critique result, skipping step")
            return None
        
        # Convert critique result to boolean
        raw_critique = critique_result.strip().lower()
        if "incorrect" in raw_critique:
            model_judgment = False
        elif "correct" in raw_critique:
            model_judgment = True
        else:
            logger.error(f"Unexpected critique result: {critique_result}")
            return None
        
        # Debug: Print output from critique_step
        # logger.info(f"Raw critique result: {critique_result}")
        # logger.info(f"Model judgment: {model_judgment}")
        # logger.info(f"Total tokens: {total_tokens}")
        
        # Save debug information
        debug_file = save_debug_info(
            problem=problem,
            step_text=step_text,
            previous_steps=previous_steps_text,
            model_judgment=model_judgment,
            ground_truth=ground_truth,
            rating=rating,
            output_dir=output_dir
        )
        # logger.info(f"Debug info saved to: {debug_file}")
        
        # Compare with ground truth
        is_correct = model_judgment == ground_truth

        output = {
            'model_judgment': model_judgment,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'step_text': step_text,
            'rating': rating,
            'raw_critique': critique_result,
            'tokens': total_tokens
        }

        # Write output to a file in the output directory from main
        # output_file = os.path.join(output_dir, f"step_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # with open(output_file, 'w') as f:
        #     json.dump(output, f, indent=2)

        
        # Ensure rating is preserved as -1, 0, or 1
        return output
    except Exception as e:
        logger.error(f"Error evaluating step: {e}")
        logger.error(f"Step data: {step_data}")
        return None

def calculate_metrics(results):
    """Calculate accuracy metrics from evaluation results."""
    # Filter out None results
    results = [r for r in results if r is not None]
    
    total = len(results)
    if total == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'total_samples': 0,
            'correct_predictions': 0,
            'metrics_by_rating': {-1: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
                                 0: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
                                 1: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0}}
        }
    
    correct = sum(1 for r in results if r['is_correct'])
    
    # Calculate metrics for each rating category
    metrics_by_rating = {-1: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
                         0: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
                         1: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0}}
    
    # First count totals for each rating
    for r in results:
        rating = r['rating']
        metrics_by_rating[rating]['total'] += 1
    
    # Then calculate TP, FP, FN for each rating
    for r in results:
        rating = r['rating']
        if r['model_judgment'] == r['ground_truth']:
            metrics_by_rating[rating]['tp'] += 1
        elif r['model_judgment']:
            metrics_by_rating[rating]['fp'] += 1
        else:
            metrics_by_rating[rating]['fn'] += 1
    
    # Overall metrics
    true_positives = sum(m['tp'] for m in metrics_by_rating.values())
    false_positives = sum(m['fp'] for m in metrics_by_rating.values())
    false_negatives = sum(m['fn'] for m in metrics_by_rating.values())
    
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_samples': total,
        'correct_predictions': correct,
        'metrics_by_rating': metrics_by_rating
    }

def verify_critique_model(dataset_path, max_workers=4, sample_size=None, output_dir="./results"):
    """
    Verify the critique model's accuracy against the PRM800K dataset.
    
    Args:
        dataset_path: Path to the PRM800K dataset
        max_workers: Number of parallel workers
        sample_size: Optional number of samples to evaluate (for testing)
        output_dir: Directory to save results
    """
    logger.info(f"Loading dataset from {dataset_path}")
    data = load_prm800k_data(dataset_path)
    
    if sample_size:
        data = data[:sample_size]
    
    logger.info(f"Evaluating {len(data)} samples using model: {MODEL}")
    
    # Prepare evaluation tasks
    eval_tasks = []
    for item in data:
        problem = item['question']['problem']
        steps = item['label']['steps']
        previous_steps = []
        
        for step_data in steps:
            # Skip if no valid completion was chosen and no human completion exists
            if step_data['chosen_completion'] is None and not step_data['human_completion']:
                continue

            # Add all completions to evaluation tasks but only use the chosen completion for previous steps
            for completion in step_data['completions']:
                # Skip steps with neutral rating
                if completion['rating'] and completion['rating'] != 0:
                    eval_tasks.append((
                        problem,
                        completion,
                        previous_steps.copy(),
                        output_dir
                    ))
                
            # Add the chosen step to previous steps
            if step_data['chosen_completion'] is not None:
                completion = step_data['completions'][step_data['chosen_completion']]
                previous_steps.append(completion['text'])
            elif step_data['human_completion']:
                previous_steps.append(step_data['human_completion']['text'])
    
    # Run evaluations in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(evaluate_step, task): task for task in eval_tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(eval_tasks), desc="Evaluating steps"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Calculate and report metrics
    metrics = calculate_metrics(results)
    
    # Add model information and timestamp to metrics
    metrics['model'] = MODEL
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['dataset'] = dataset_path
    metrics['sample_size'] = sample_size if sample_size else len(data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with model name and timestamp
    model_name = MODEL.replace('/', '-').replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'critique_results_{model_name}_{timestamp}.json')
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"Total Samples: {metrics['total_samples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Overall Precision: {metrics['precision']:.4f}")
    logger.info(f"Overall Recall: {metrics['recall']:.4f}")
    logger.info(f"Overall F1 Score: {metrics['f1']:.4f}")
    
    # Report metrics by rating category
    logger.info("\nMetrics by Rating Category:")
    for rating in sorted(metrics['metrics_by_rating'].keys()):
        m = metrics['metrics_by_rating'][rating]
        total = m['tp'] + m['fp'] + m['fn']
        if total > 0:
            accuracy = m['tp'] / total if total > 0 else 0
            logger.info(f"\nRating {rating}:")
            logger.info(f"  Samples: {m['total']}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  True Positives: {m['tp']}")
            logger.info(f"  False Positives: {m['fp']}")
            logger.info(f"  False Negatives: {m['fn']}")
    
    return metrics

if __name__ == "__main__":
    # Create a timestamped output directory
    output_dir = f"../critique_verification_results/critique_results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = verify_critique_model(
        dataset_path="../datasets/phase1_train.jsonl",
        max_workers=16,
        sample_size=50,
        output_dir=output_dir
    )
