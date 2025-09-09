#!/usr/bin/env python3
"""
Step 4: Analyze and visualize classification results.

This script provides detailed analysis:
- Distribution analysis across telescopes and classifications
- Cross-tabulation of telescope vs classification types
- Statistical summaries and patterns
- Export results for further analysis or visualization

Usage:
    python scripts/4-analyze_results.py results/submission.csv
    python scripts/4-analyze_results.py results/submission.csv --detailed
    python scripts/4-analyze_results.py results/submission.csv --export-formats json csv
"""

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import sys

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_id_field(id_value: str) -> Tuple[str, str]:
    """Parse combined Id field into bibcode and telescope."""
    parts = id_value.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return id_value, "UNKNOWN"


class ResultsAnalyzer:
    """Analyzer for telescope classification results."""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.df = self._load_results()
        self.classification_types = ['science', 'instrumentation', 'mention', 'not_telescope']
        self.telescopes = ['CHANDRA', 'HST', 'JWST', 'NONE']
    
    def _load_results(self) -> pd.DataFrame:
        """Load results from CSV file."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        df = pd.read_csv(self.results_file)
        
        # Add parsed bibcode column
        df[['bibcode', 'telescope_parsed']] = df['Id'].apply(
            lambda x: pd.Series(parse_id_field(x))
        )
        
        return df
    
    def basic_statistics(self) -> Dict:
        """Calculate basic statistics."""
        
        stats = {
            'total_papers': len(self.df),
            'unique_bibcodes': self.df['bibcode'].nunique(),
            'telescope_distribution': {},
            'classification_distribution': {},
            'classification_percentages': {}
        }
        
        # Telescope distribution
        telescope_counts = self.df['telescope'].value_counts()
        stats['telescope_distribution'] = telescope_counts.to_dict()
        
        # Classification distribution
        for class_type in self.classification_types:
            count = self.df[class_type].sum()
            stats['classification_distribution'][class_type] = int(count)
            stats['classification_percentages'][class_type] = float(count / len(self.df) * 100)
        
        return stats
    
    def cross_tabulation_analysis(self) -> Dict:
        """Analyze relationships between telescopes and classifications."""
        
        analysis = {}
        
        # Create cross-tabulation for each classification type
        for class_type in self.classification_types:
            crosstab = pd.crosstab(self.df['telescope'], self.df[class_type])
            analysis[class_type] = {
                'crosstab': crosstab.to_dict(),
                'percentages_by_telescope': {},
                'percentages_by_classification': {}
            }
            
            # Calculate percentages by telescope
            for telescope in crosstab.index:
                total_telescope = self.df[self.df['telescope'] == telescope].shape[0]
                if total_telescope > 0:
                    true_count = crosstab.loc[telescope, True] if True in crosstab.columns else 0
                    analysis[class_type]['percentages_by_telescope'][telescope] = float(
                        true_count / total_telescope * 100
                    )
        
        return analysis
    
    def multi_label_analysis(self) -> Dict:
        """Analyze multi-label classification patterns."""
        
        # Create binary combinations
        label_combinations = []
        for _, row in self.df.iterrows():
            combo = tuple(row[class_type] for class_type in self.classification_types)
            label_combinations.append(combo)
        
        combo_counts = Counter(label_combinations)
        
        # Convert to readable format
        readable_combos = {}
        for combo, count in combo_counts.most_common():
            combo_names = [self.classification_types[i] for i, val in enumerate(combo) if val]
            if not combo_names:
                combo_names = ['none']
            combo_key = '+'.join(combo_names)
            readable_combos[combo_key] = {
                'count': count,
                'percentage': float(count / len(self.df) * 100)
            }
        
        # Analyze by telescope
        telescope_combos = {}
        for telescope in self.telescopes:
            telescope_df = self.df[self.df['telescope'] == telescope]
            if len(telescope_df) == 0:
                continue
            
            telescope_label_combinations = []
            for _, row in telescope_df.iterrows():
                combo = tuple(row[class_type] for class_type in self.classification_types)
                telescope_label_combinations.append(combo)
            
            telescope_combo_counts = Counter(telescope_label_combinations)
            telescope_readable_combos = {}
            
            for combo, count in telescope_combo_counts.most_common(5):  # Top 5 for each telescope
                combo_names = [self.classification_types[i] for i, val in enumerate(combo) if val]
                if not combo_names:
                    combo_names = ['none']
                combo_key = '+'.join(combo_names)
                telescope_readable_combos[combo_key] = {
                    'count': count,
                    'percentage': float(count / len(telescope_df) * 100)
                }
            
            telescope_combos[telescope] = telescope_readable_combos
        
        return {
            'overall_combinations': readable_combos,
            'by_telescope': telescope_combos,
            'total_unique_combinations': len(combo_counts)
        }
    
    def paper_level_analysis(self) -> Dict:
        """Analyze patterns at the paper level (multiple telescopes per paper)."""
        
        # Group by bibcode to find papers with multiple telescope entries
        bibcode_groups = self.df.groupby('bibcode')
        
        multi_telescope_papers = []
        single_telescope_papers = []
        
        for bibcode, group in bibcode_groups:
            if len(group) > 1:
                telescopes = list(group['telescope'].unique())
                classifications = {}
                for class_type in self.classification_types:
                    classifications[class_type] = list(group[class_type])
                
                multi_telescope_papers.append({
                    'bibcode': bibcode,
                    'telescopes': telescopes,
                    'count': len(group),
                    'classifications': classifications
                })
            else:
                single_telescope_papers.append(bibcode)
        
        # Analyze classification consistency for multi-telescope papers
        consistency_analysis = {}
        if multi_telescope_papers:
            for class_type in self.classification_types:
                consistent_count = 0
                total_multi_papers = len(multi_telescope_papers)
                
                for paper in multi_telescope_papers:
                    classifications = paper['classifications'][class_type]
                    if len(set(classifications)) == 1:  # All same value
                        consistent_count += 1
                
                consistency_analysis[class_type] = {
                    'consistent_papers': consistent_count,
                    'total_multi_telescope_papers': total_multi_papers,
                    'consistency_rate': float(consistent_count / total_multi_papers * 100) if total_multi_papers > 0 else 0
                }
        
        return {
            'total_unique_papers': len(bibcode_groups),
            'single_telescope_papers': len(single_telescope_papers),
            'multi_telescope_papers': len(multi_telescope_papers),
            'multi_telescope_examples': multi_telescope_papers[:10],  # Show first 10 examples
            'classification_consistency': consistency_analysis
        }
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        stats = self.basic_statistics()
        multi_label = self.multi_label_analysis()
        paper_analysis = self.paper_level_analysis()
        
        report = []
        report.append("TELESCOPE CLASSIFICATION RESULTS ANALYSIS")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS")
        report.append("-" * 30)
        report.append(f"Total classifications: {stats['total_papers']:,}")
        report.append(f"Unique papers (bibcodes): {stats['unique_bibcodes']:,}")
        report.append("")
        
        report.append("Telescope Distribution:")
        for telescope, count in stats['telescope_distribution'].items():
            percentage = count / stats['total_papers'] * 100
            report.append(f"  {telescope}: {count:,} ({percentage:.1f}%)")
        report.append("")
        
        report.append("Classification Distribution:")
        for class_type, count in stats['classification_distribution'].items():
            percentage = stats['classification_percentages'][class_type]
            report.append(f"  {class_type}: {count:,} ({percentage:.1f}%)")
        report.append("")
        
        # Multi-label patterns
        report.append("CLASSIFICATION PATTERNS")
        report.append("-" * 30)
        report.append(f"Total unique label combinations: {multi_label['total_unique_combinations']}")
        report.append("")
        report.append("Most common classification combinations:")
        
        for combo, data in list(multi_label['overall_combinations'].items())[:10]:
            report.append(f"  {combo}: {data['count']:,} ({data['percentage']:.1f}%)")
        report.append("")
        
        # Paper-level analysis
        report.append("PAPER-LEVEL ANALYSIS")
        report.append("-" * 30)
        report.append(f"Papers appearing with single telescope: {paper_analysis['single_telescope_papers']:,}")
        report.append(f"Papers appearing with multiple telescopes: {paper_analysis['multi_telescope_papers']:,}")
        
        if paper_analysis['classification_consistency']:
            report.append("")
            report.append("Classification consistency for multi-telescope papers:")
            for class_type, consistency in paper_analysis['classification_consistency'].items():
                rate = consistency['consistency_rate']
                report.append(f"  {class_type}: {rate:.1f}% consistent")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Analyze and visualize classification results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to results CSV file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./analysis"),
        help="Directory for analysis output files"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed analysis including cross-tabulations"
    )
    
    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=["json", "csv", "txt"],
        default=["json"],
        help="Export formats for analysis results"
    )
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 4: RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = ResultsAnalyzer(args.results_file)
        
        # Generate analyses
        stats = analyzer.basic_statistics()
        multi_label = analyzer.multi_label_analysis()
        paper_analysis = analyzer.paper_level_analysis()
        
        analysis_results = {
            'basic_statistics': stats,
            'multi_label_analysis': multi_label,
            'paper_level_analysis': paper_analysis
        }
        
        if args.detailed:
            analysis_results['cross_tabulation'] = analyzer.cross_tabulation_analysis()
        
        # Generate summary report
        summary_report = analyzer.generate_summary_report()
        print("\n" + summary_report)
        
        # Export results in requested formats
        base_filename = args.results_file.stem + "_analysis"
        
        if "json" in args.export_formats:
            json_file = args.output_dir / f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"\nJSON analysis saved to: {json_file}")
        
        if "txt" in args.export_formats:
            txt_file = args.output_dir / f"{base_filename}.txt"
            with open(txt_file, 'w') as f:
                f.write(summary_report)
            print(f"Text report saved to: {txt_file}")
        
        if "csv" in args.export_formats:
            # Export key statistics as CSV
            csv_file = args.output_dir / f"{base_filename}.csv"
            
            # Create a summary DataFrame
            summary_data = []
            
            # Add telescope distribution
            for telescope, count in stats['telescope_distribution'].items():
                summary_data.append({
                    'metric_type': 'telescope_distribution',
                    'category': telescope,
                    'count': count,
                    'percentage': count / stats['total_papers'] * 100
                })
            
            # Add classification distribution
            for class_type, count in stats['classification_distribution'].items():
                summary_data.append({
                    'metric_type': 'classification_distribution',
                    'category': class_type,
                    'count': count,
                    'percentage': stats['classification_percentages'][class_type]
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")
        
        # Key insights
        print(f"\nKEY INSIGHTS")
        print("-" * 30)
        
        # Try to calculate Macro-F1 if ground truth is available
        macro_f1 = None
        potential_gt_files = [
            args.results_file.parent.parent / "ground_truth.json",
            args.results_file.parent / "ground_truth.json", 
            Path("ground_truth.json"),
            Path("data/ground_truth.json"),
            Path("data/test_subset.csv")  # For test cases
        ]
        
        for gt_file in potential_gt_files:
            if gt_file.exists():
                try:
                    print(f"• Found ground truth file: {gt_file}")
                    print(f"• Run 'python scripts/3-evaluate_results.py {args.results_file} {gt_file}' for detailed Macro-F1 evaluation")
                    break
                except Exception as e:
                    continue
        else:
            print(f"• No ground truth file found - cannot calculate Macro-F1 score")
            print(f"• For evaluation, run: python scripts/3-evaluate_results.py {args.results_file} ground_truth.json")
        
        # Most common telescope
        most_common_telescope = max(stats['telescope_distribution'].items(), key=lambda x: x[1])
        print(f"• Most common telescope: {most_common_telescope[0]} ({most_common_telescope[1]:,} papers)")
        
        # Most common classification
        most_common_class = max(stats['classification_distribution'].items(), key=lambda x: x[1])
        print(f"• Most common classification: {most_common_class[0]} ({most_common_class[1]:,} papers)")
        
        # Multi-telescope papers
        if paper_analysis['multi_telescope_papers'] > 0:
            multi_rate = paper_analysis['multi_telescope_papers'] / paper_analysis['total_unique_papers'] * 100
            print(f"• Papers with multiple telescopes: {multi_rate:.1f}%")
        
        # Classification combination diversity
        combo_count = multi_label['total_unique_combinations']
        max_possible = 2 ** len(analyzer.classification_types)
        diversity_rate = combo_count / max_possible * 100
        print(f"• Classification diversity: {combo_count}/{max_possible} combinations used ({diversity_rate:.1f}%)")
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())