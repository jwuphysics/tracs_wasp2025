#!/usr/bin/env python3
"""
Step 1: Process raw CSV data for telescope classification pipeline.

This script handles the initial data processing:
- Parses combined Id field (bibcode_telescope format)
- Creates test subsets for development/validation
- Converts CSV to JSON format if needed
- Creates ground truth JSON from training CSV files
- Shows data statistics and distribution

Usage:
    python scripts/1-process_data.py data/test.csv
    python scripts/1-process_data.py data/test.csv --create-subset 100
    python scripts/1-process_data.py data/test.csv --convert-to-json
    python scripts/1-process_data.py data/train.csv --create-ground-truth
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_id_field(id_value: str) -> Tuple[str, str]:
    """
    Parse the combined Id field to extract bibcode and telescope.
    
    Args:
        id_value: Combined ID like "2012A&A...537A..18M_CHANDRA"
        
    Returns:
        Tuple of (bibcode, telescope)
    """
    parts = id_value.rsplit('_', 1)
    if len(parts) == 2:
        bibcode, telescope = parts
        return bibcode, telescope
    else:
        # Fallback if no underscore found
        return id_value, "UNKNOWN"


def analyze_csv_data(csv_file: Path) -> Dict:
    """
    Analyze CSV data and return statistics.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing CSV data from {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Parse telescope distribution
    telescopes = df['Id'].str.split('_', expand=True)[1].value_counts()
    
    # Check for duplicate bibcodes (same paper, multiple telescopes)
    bibcodes = df['Id'].str.split('_', expand=True)[0]
    duplicate_bibcodes = bibcodes[bibcodes.duplicated()].value_counts()
    
    # Check for missing data
    missing_data = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_data[col] = missing_count
    
    analysis = {
        "total_rows": len(df),
        "telescope_distribution": telescopes.to_dict(),
        "duplicate_bibcodes_count": len(duplicate_bibcodes),
        "papers_with_multiple_telescopes": duplicate_bibcodes.to_dict(),
        "missing_data": missing_data
    }
    
    return analysis


def create_test_subset(csv_file: Path, n_papers: int = 100, output_file: Path = None):
    """
    Create a balanced test subset from the CSV file.

    Args:
        csv_file: Input CSV file
        n_papers: Number of papers to include in subset
        output_file: Output file (defaults to {input_stem}_subset.csv)
    """
    if output_file is None:
        # Use input filename stem to determine output name (e.g., train.csv -> train_subset.csv)
        output_file = csv_file.parent / f"{csv_file.stem}_subset.csv"

    print(f"Creating subset with {n_papers} papers...")
    
    df = pd.read_csv(csv_file)
    
    # Sample papers trying to get representation from each telescope
    telescopes = df['Id'].str.split('_', expand=True)[1].value_counts()
    print(f"Available telescopes: {telescopes.to_dict()}")
    
    # Sample proportionally
    subset_rows = []
    papers_per_telescope = max(1, n_papers // len(telescopes))
    
    for telescope in telescopes.index:
        telescope_rows = df[df['Id'].str.endswith(f'_{telescope}')]
        sample_size = min(papers_per_telescope, len(telescope_rows))
        sampled = telescope_rows.sample(n=sample_size, random_state=42)
        subset_rows.append(sampled)
        print(f"  {telescope}: sampled {sample_size} from {len(telescope_rows)} available")
    
    # Combine all samples
    subset_df = pd.concat(subset_rows, ignore_index=True)
    
    # If we need more papers, sample randomly from remaining
    if len(subset_df) < n_papers:
        remaining_needed = n_papers - len(subset_df)
        remaining_df = df[~df.index.isin(subset_df.index)]
        if len(remaining_df) > 0:
            additional = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=42)
            subset_df = pd.concat([subset_df, additional], ignore_index=True)
    
    # Save subset
    subset_df.to_csv(output_file, index=False)

    print(f"Created subset with {len(subset_df)} papers saved to {output_file}")
    
    # Print telescope distribution in subset
    subset_telescopes = subset_df['Id'].str.split('_', expand=True)[1].value_counts()
    print("Subset telescope distribution:")
    for telescope, count in subset_telescopes.items():
        print(f"  {telescope}: {count}")
    
    return output_file


def create_ground_truth_json(csv_file: Path, output_file: Path = None) -> Path:
    """
    Create ground truth JSON file from CSV with classification labels.
    
    Args:
        csv_file: Path to CSV file with ground truth labels
        output_file: Optional output path (defaults to ground_truth.json)
        
    Returns:
        Path to created JSON file
    """
    if output_file is None:
        output_file = csv_file.parent / "ground_truth.json"
    
    df = pd.read_csv(csv_file)
    
    # Check if this CSV has the required ground truth columns
    required_cols = ['science', 'instrumentation', 'mention', 'not_telescope']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: CSV missing ground truth columns: {missing_cols}")
        print(f"Cannot create ground truth JSON from this file.")
        return None
    
    print(f"Creating ground truth JSON from {csv_file}")
    print(f"Output: {output_file}")
    
    ground_truth_data = []
    
    for _, row in df.iterrows():
        bibcode, telescope = parse_id_field(row['Id'])
        
        paper = {
            'bibcode': bibcode,
            'telescope': telescope,
            'science': bool(row['science']),
            'instrumentation': bool(row['instrumentation']),
            'mention': bool(row['mention']),
            'not_telescope': bool(row['not_telescope'])
        }
        ground_truth_data.append(paper)
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    print(f"Created ground truth JSON with {len(ground_truth_data):,} entries")
    
    # Show distribution of labels
    label_counts = {col: df[col].sum() for col in required_cols}
    print(f"\nGround truth label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    return output_file


def convert_csv_to_json(csv_file: Path, output_file: Path = None):
    """
    Convert CSV to JSON format for alternative processing.
    
    Args:
        csv_file: Input CSV file
        output_file: Output JSON file (defaults to csv_file with .json extension)
    """
    if output_file is None:
        output_file = csv_file.with_suffix('.json')
    
    print(f"Converting CSV to JSON format...")
    
    df = pd.read_csv(csv_file)
    papers = []
    
    for _, row in df.iterrows():
        # Parse the Id field
        bibcode, telescope = parse_id_field(row['Id'])
        
        # Helper function to handle NaN values
        def safe_get_string(value, default=''):
            if pd.isna(value):
                return default
            return str(value) if value is not None else default
        
        def safe_get_int(value, default=None):
            if pd.isna(value):
                return default
            try:
                return int(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Create paper dictionary
        paper = {
            "id": row['Id'],  # Keep original combined ID
            "bibcode": bibcode,
            "telescope": telescope,
            "author": safe_get_string(row.get('author')),
            "year": safe_get_int(row.get('year')),
            "title": safe_get_string(row.get('title')),
            "abstract": safe_get_string(row.get('abstract')),
            "body": safe_get_string(row.get('body')),
            "acknowledgments": safe_get_string(row.get('acknowledgments')),
            "grants": safe_get_string(row.get('grants'))
        }
        
        papers.append(paper)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(papers)} papers to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Process CSV data for telescope classification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (defaults based on operation)"
    )
    
    parser.add_argument(
        "--create-subset",
        type=int,
        metavar="N",
        help="Create a test subset with N papers"
    )
    
    parser.add_argument(
        "--convert-to-json",
        action="store_true",
        help="Convert CSV to JSON format"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze data, don't create output files"
    )
    
    parser.add_argument(
        "--create-ground-truth",
        action="store_true",
        help="Create ground truth JSON file from CSV with classification labels"
    )
    
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return 1
    
    # Always perform analysis
    print("=" * 60)
    print("STEP 1: DATA PROCESSING AND ANALYSIS")
    print("=" * 60)
    
    analysis = analyze_csv_data(args.csv_file)
    
    print(f"\nData Summary:")
    print(f"  Total rows: {analysis['total_rows']:,}")
    print(f"  Papers with multiple telescopes: {analysis['duplicate_bibcodes_count']}")
    
    print(f"\nTelescope Distribution:")
    for telescope, count in sorted(analysis['telescope_distribution'].items()):
        percentage = (count / analysis['total_rows']) * 100
        print(f"  {telescope}: {count:,} ({percentage:.1f}%)")
    
    if analysis['papers_with_multiple_telescopes']:
        print(f"\nPapers appearing with multiple telescopes:")
        for bibcode, count in list(analysis['papers_with_multiple_telescopes'].items())[:5]:
            print(f"  {bibcode}: {count + 1} telescopes")
        if len(analysis['papers_with_multiple_telescopes']) > 5:
            print(f"  ... and {len(analysis['papers_with_multiple_telescopes']) - 5} more")
    
    if analysis['missing_data']:
        print(f"\nMissing data:")
        for col, count in analysis['missing_data'].items():
            percentage = (count / analysis['total_rows']) * 100
            print(f"  {col}: {count:,} ({percentage:.1f}%)")
    
    if args.analyze_only:
        print(f"\nAnalysis complete. Use --create-subset, --convert-to-json, or --create-ground-truth for next steps.")
        return 0
    
    # Perform requested operations
    if args.create_subset:
        create_test_subset(args.csv_file, args.create_subset, args.output)
        print(f"\nNext step: Run classification on the test subset")
        print(f"   python scripts/2-classify_papers.py data/test_subset.csv")
    
    if args.convert_to_json:
        convert_csv_to_json(args.csv_file, args.output)
        print(f"\nNext step: Use JSON processing mode")
        print(f"   python -m automated_mission_classifier --data-file {args.output or args.csv_file.with_suffix('.json')}")
    
    if args.create_ground_truth:
        ground_truth_file = create_ground_truth_json(args.csv_file, args.output)
        if ground_truth_file:
            print(f"\nNext step: Use ground truth for evaluation")
            print(f"   python scripts/3-evaluate_results.py results/submission.csv {ground_truth_file}")
    
    if not args.create_subset and not args.convert_to_json and not args.create_ground_truth:
        print(f"\nSuggested next steps:")
        print(f"   • Create test subset: python scripts/1-process_data.py {args.csv_file} --create-subset 100")
        print(f"   • Convert to JSON: python scripts/1-process_data.py {args.csv_file} --convert-to-json")
        print(f"   • Create ground truth: python scripts/1-process_data.py {args.csv_file} --create-ground-truth")
        print(f"   • Run full classification: python scripts/2-classify_papers.py {args.csv_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())