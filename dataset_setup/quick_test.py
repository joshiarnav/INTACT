import json
import numpy as np
from collections import defaultdict

def analyze_steps_statistics():
    step_counts = []
    
    with open('../datasets/phase1_train.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            steps = data['label']['steps']
            step_counts.append(len(steps))
    
    step_counts = np.array(step_counts)
    
    stats = {
        'min_steps': int(np.min(step_counts)),
        'max_steps': int(np.max(step_counts)),
        'mean_steps': float(np.mean(step_counts)),
        'median_steps': float(np.median(step_counts)),
        'std_steps': float(np.std(step_counts)),
        'total_problems': len(step_counts)
    }
    
    # Calculate distribution of step counts
    step_dist = defaultdict(int)
    for count in step_counts:
        step_dist[int(count)] += 1
    
    # Convert to regular dict and sort
    step_dist = dict(sorted(step_dist.items()))
    
    print("\nStep Statistics:")
    print(f"Minimum steps: {stats['min_steps']}")
    print(f"Maximum steps: {stats['max_steps']}")
    print(f"Mean steps: {stats['mean_steps']:.2f}")
    print(f"Median steps: {stats['median_steps']:.2f}")
    print(f"Standard deviation: {stats['std_steps']:.2f}")
    print(f"Total problems analyzed: {stats['total_problems']}")
    
    print("\nStep Distribution:")
    for steps, count in step_dist.items():
        print(f"{steps} steps: {count} problems ({count/stats['total_problems']*100:.1f}%)")

if __name__ == '__main__':
    analyze_steps_statistics()