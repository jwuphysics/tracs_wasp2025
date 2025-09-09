#!/usr/bin/env python3
"""
Step 3: Evaluate classification results against ground truth.

This script evaluates model performance:
- Compares predictions against ground truth labels
- Calculates accuracy, precision, recall, F1 scores
- Generates confusion matrices for each telescope and classification type
- Provides detailed error analysis and misclassification examples

Usage:
    python scripts/3-evaluate_results.py results/submission.csv ground_truth.json
    python scripts/3-evaluate_results.py results/submission.csv --ground-truth-csv train.csv
    python scripts/3-evaluate_results.py results/submission.csv --analyze-only
"""

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_id_field(id_value: str) -> Tuple[str, str]:
    """Parse combined Id field into bibcode and telescope."""
    parts = id_value.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return id_value, "UNKNOWN"


class ResultsEvaluator:
    """Evaluator for telescope classification performance."""
    
    def __init__(self, predictions_file: Path, ground_truth_file: Optional[Path] = None):
        self.predictions_file = predictions_file
        self.ground_truth_file = ground_truth_file
        
        # Load predictions
        self.predictions = self._load_predictions()
        
        # Load ground truth if provided
        self.ground_truth = {}
        if ground_truth_file:
            self.ground_truth = self._load_ground_truth()
        
        # Classification types
        self.classification_types = ['science', 'instrumentation', 'mention', 'not_telescope']
        self.telescopes = ['CHANDRA', 'HST', 'JWST', 'NONE']
    
    def _load_predictions(self) -> Dict:
        """Load predictions from CSV file."""
        if not self.predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_file}")
        
        df = pd.read_csv(self.predictions_file)
        
        preds = {}
        for _, row in df.iterrows():
            bibcode, telescope = parse_id_field(row['Id'])
            preds[row['Id']] = {
                'bibcode': bibcode,
                'telescope': telescope,
                'science': bool(row['science']),
                'instrumentation': bool(row['instrumentation']),
                'mention': bool(row['mention']),
                'not_telescope': bool(row['not_telescope'])
            }
        
        return preds
    
    def _load_ground_truth(self) -> Dict:
        """Load ground truth labels."""
        if not self.ground_truth_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_file}")
        
        gt = {}
        
        if self.ground_truth_file.suffix == '.json':
            # JSON format
            with open(self.ground_truth_file, 'r') as f:
                papers = json.load(f)
            
            for paper in papers:
                if isinstance(paper, dict):
                    bibcode = paper.get('bibcode', '')
                    telescope = paper.get('telescope', '')
                    id_key = f"{bibcode}_{telescope}"
                    gt[id_key] = {
                        'telescope': telescope,
                        'science': paper.get('science_label', paper.get('science', False)),
                        'instrumentation': paper.get('instrumentation_label', paper.get('instrumentation', False)),
                        'mention': paper.get('mention_label', paper.get('mention', False)),
                        'not_telescope': paper.get('not_telescope_label', paper.get('not_telescope', False))
                    }
        
        elif self.ground_truth_file.suffix == '.csv':
            # CSV format (like train.csv)
            df = pd.read_csv(self.ground_truth_file)
            
            # Check if this CSV actually has ground truth labels
            required_cols = ['science', 'instrumentation', 'mention', 'not_telescope']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"WARNING: Ground truth CSV is missing classification columns: {missing_cols}")
                print(f"This appears to be test data without ground truth labels.")
                print(f"Cannot perform evaluation - all ground truth values will be False.")
                print()
            
            for _, row in df.iterrows():
                id_key = row['Id']
                bibcode, telescope = parse_id_field(id_key)
                gt[id_key] = {
                    'telescope': telescope,
                    'science': bool(row.get('science', False)),
                    'instrumentation': bool(row.get('instrumentation', False)),
                    'mention': bool(row.get('mention', False)),
                    'not_telescope': bool(row.get('not_telescope', False))
                }

        return gt
    
    def _calculate_metrics(self, y_true: List[bool], y_pred: List[bool]) -> Dict:
        """Calculate classification metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = np.sum((y_true == True) & (y_pred == True))
        fp = np.sum((y_true == False) & (y_pred == True))
        fn = np.sum((y_true == True) & (y_pred == False))
        tn = np.sum((y_true == False) & (y_pred == False))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'support': int(tp + fn)  # Number of true positives in ground truth
        }
    
    def evaluate_performance(self) -> Dict:
        """Evaluate performance for each classification type per telescope."""
        if not self.ground_truth:
            print("No ground truth data available for evaluation")
            return {}
        
        results = {}
        
        # Find common papers between predictions and ground truth
        common_ids = set(self.predictions.keys()) & set(self.ground_truth.keys())
        
        if not common_ids:
            print("No common papers found between predictions and ground truth")
            return {}
        
        # Evaluate by telescope
        for telescope in self.telescopes:
            telescope_ids = [id_key for id_key in common_ids 
                           if self.predictions[id_key]['telescope'] == telescope]
            
            if not telescope_ids:
                continue
            
            results[telescope] = {}
            # Evaluate each classification type
            for class_type in self.classification_types:
                y_true = [self.ground_truth[id_key][class_type] for id_key in telescope_ids]
                y_pred = [self.predictions[id_key][class_type] for id_key in telescope_ids]
                
                metrics = self._calculate_metrics(y_true, y_pred)
                results[telescope][class_type] = metrics
        
        return results
    
    def analyze_predictions(self) -> Dict:
        """Analyze prediction distribution and patterns."""
        
        analysis = {
            'total_predictions': len(self.predictions),
            'telescope_distribution': defaultdict(int),
            'classification_distribution': defaultdict(int),
            'classification_by_telescope': defaultdict(lambda: defaultdict(int))
        }
        
        for pred in self.predictions.values():
            telescope = pred['telescope']
            analysis['telescope_distribution'][telescope] += 1
            
            for class_type in self.classification_types:
                if pred[class_type]:
                    analysis['classification_distribution'][class_type] += 1
                    analysis['classification_by_telescope'][telescope][class_type] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        analysis['telescope_distribution'] = dict(analysis['telescope_distribution'])
        analysis['classification_distribution'] = dict(analysis['classification_distribution'])
        analysis['classification_by_telescope'] = {
            telescope: dict(classifications)
            for telescope, classifications in analysis['classification_by_telescope'].items()
        }
        
        return analysis
    
    def find_misclassifications(self, limit: int = 10) -> Dict:
        """Find and analyze misclassification examples."""
        if not self.ground_truth:
            return {}
        
        misclassifications = defaultdict(list)
        common_ids = set(self.predictions.keys()) & set(self.ground_truth.keys())
        
        for id_key in common_ids:
            pred = self.predictions[id_key]
            truth = self.ground_truth[id_key]
            
            for class_type in self.classification_types:
                if pred[class_type] != truth[class_type]:
                    misclassifications[class_type].append({
                        'id': id_key,
                        'telescope': pred['telescope'],
                        'predicted': pred[class_type],
                        'ground_truth': truth[class_type]
                    })
        
        # Limit examples for each type
        for class_type in misclassifications:
            misclassifications[class_type] = misclassifications[class_type][:limit]
        
        return dict(misclassifications)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Evaluate classification results against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "predictions_file",
        type=Path,
        help="Path to predictions CSV file"
    )
    
    parser.add_argument(
        "ground_truth_file",
        type=Path,
        nargs='?',
        help="Path to ground truth file (JSON or CSV format)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./evaluation"),
        help="Directory for evaluation output files"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze predictions without ground truth comparison"
    )
    
    args = parser.parse_args()
    
    if not args.predictions_file.exists():
        print(f"Error: Predictions file not found: {args.predictions_file}")
        return 1
    
    if args.ground_truth_file and not args.ground_truth_file.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 3: RESULTS EVALUATION")
    print("=" * 60)
    print(f"Predictions: {args.predictions_file}")
    if args.ground_truth_file:
        print(f"Ground truth: {args.ground_truth_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Initialize evaluator
        evaluator = ResultsEvaluator(args.predictions_file, args.ground_truth_file)
        
        # Always analyze predictions
        print("PREDICTION ANALYSIS")
        print("-" * 30)
        analysis = evaluator.analyze_predictions()
        
        print(f"Total predictions: {analysis['total_predictions']:,}")
        print(f"\nTelescope distribution:")
        for telescope, count in analysis['telescope_distribution'].items():
            percentage = (count / analysis['total_predictions']) * 100
            print(f"  {telescope}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nClassification distribution:")
        for class_type, count in analysis['classification_distribution'].items():
            percentage = (count / analysis['total_predictions']) * 100
            print(f"  {class_type}: {count:,} ({percentage:.1f}%)")
        
        # Performance evaluation if ground truth available
        if not args.analyze_only and args.ground_truth_file:
            print(f"\nPERFORMANCE EVALUATION")
            print("-" * 30)
            performance = evaluator.evaluate_performance()
            
            if performance:
                # Overall summary
                all_f1_scores = []
                for telescope_results in performance.values():
                    for metrics in telescope_results.values():
                        if metrics['support'] > 0:  # Only include categories with actual positive examples
                            all_f1_scores.append(metrics['f1'])
                
                if all_f1_scores:
                    macro_f1 = np.mean(all_f1_scores)
                    print(f"\nMacro-F1 score: {macro_f1:.3f}")
                    print(f"(Averaged across {len(all_f1_scores)} telescope-classification combinations with support > 0)")
                else:
                    print(f"\nMacro-F1 score: Cannot calculate (no positive examples in ground truth)")
                    print(f"All ground truth labels appear to be False - check your ground truth data")
                
                # Find misclassifications
                print(f"\nMISCLASSIFICATION EXAMPLES")
                print("-" * 30)
                misclassifications = evaluator.find_misclassifications(limit=5)
                
                for class_type, examples in misclassifications.items():
                    if examples:
                        print(f"  {class_type}: {len(examples)} examples")
                        for i, example in enumerate(examples[:3], 1):
                            print(f"    {i}. {example['id']} ({example['telescope']}): "
                                  f"predicted={example['predicted']}, actual={example['ground_truth']}")
        
        # Save detailed results
        output_file = args.output_dir / "evaluation_results.json"
        results = {
            'analysis': analysis,
            'performance': performance,  # Use the already computed performance
            'misclassifications': evaluator.find_misclassifications() if args.ground_truth_file else {}
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Suggest next steps
        print(f"\nNext steps:")
        print(f"   • Generate detailed analysis: python scripts/4-analyze_results.py {args.predictions_file}")
        
        # Performance evaluation summary (already done above)
        if performance:
            avg_f1_scores = [metrics['f1'] for telescope_results in performance.values() 
                           for metrics in telescope_results.values() if metrics['support'] > 0]
            if avg_f1_scores:  # Only calculate if we have valid F1 scores
                avg_f1 = np.mean(avg_f1_scores)
                if avg_f1 < 0.8:
                    print(f"   • Consider improving model: F1 score ({avg_f1:.3f}) could be higher")
                else:
                    print(f"   • Model performance looks good: F1 score = {avg_f1:.3f}")
            else:
                print(f"   • No positive examples in ground truth - cannot calculate meaningful F1 score")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    print(f"\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())