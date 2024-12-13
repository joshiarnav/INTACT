import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_debug_info(problem, step_text, previous_steps, model_judgment, ground_truth, rating, output_dir="./debug_results"):
    """Save debug information for each step evaluation."""
    import os
    # Add datetime to output directory
    # output_dir = os.path.join(output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "problem": problem,
        "step": {
            "text": step_text,
            "previous_steps": previous_steps
        },
        "evaluation": {
            "model_judgment": model_judgment,
            "ground_truth": ground_truth,
            "rating": rating
        }
    }
    
    # Create a unique filename for each debug entry
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"debug_step_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    return filepath
