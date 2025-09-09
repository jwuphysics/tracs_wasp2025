#!/usr/bin/env python3
"""Process CSV competition data for telescope classification."""

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


def load_csv_data(csv_file: Path) -> List[Dict]:
    """
    Load CSV data and convert to expected format.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        List of paper dictionaries
    """
    print(f"Loading CSV data from {csv_file}")
    
    df = pd.read_csv(csv_file)
    papers = []
    
    for _, row in df.iterrows():
        # Parse the Id field
        bibcode, telescope = parse_id_field(row['Id'])
        
        # Create paper dictionary
        paper = {
            "id": row['Id'],  # Keep original combined ID
            "bibcode": bibcode,
            "telescope": telescope,
            "author": row.get('author', ''),
            "year": row.get('year', ''),
            "title": row.get('title', ''),
            "abstract": row.get('abstract', ''),
            "body": row.get('body', ''),
            "acknowledgments": row.get('acknowledgments', ''),
            "grants": row.get('grants', '')
        }
        
        papers.append(paper)
    
    print(f"Loaded {len(papers)} paper records")
    return papers


def convert_csv_to_json(csv_file: Path, output_file: Path = None):
    """
    Convert CSV to JSON format for processing.
    
    Args:
        csv_file: Input CSV file
        output_file: Output JSON file (defaults to csv_file with .json extension)
    """
    if output_file is None:
        output_file = csv_file.with_suffix('.json')
    
    papers = load_csv_data(csv_file)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(papers)} papers to {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    telescopes = {}
    for paper in papers:
        telescope = paper['telescope']
        telescopes[telescope] = telescopes.get(telescope, 0) + 1
    
    for telescope, count in sorted(telescopes.items()):
        print(f"  {telescope}: {count}")
    
    return output_file


def create_test_subset(csv_file: Path, n_papers: int = 100, output_file: Path = None):
    """
    Create a small test subset from the CSV file.
    
    Args:
        csv_file: Input CSV file
        n_papers: Number of papers to include in subset
        output_file: Output file (defaults to test_subset.csv)
    """
    if output_file is None:
        output_file = csv_file.parent / "test_subset.csv"
    
    df = pd.read_csv(csv_file)
    
    # Sample papers trying to get representation from each telescope
    telescopes = df['Id'].str.split('_', expand=True)[1].value_counts()
    print(f"Available telescopes: {telescopes.to_dict()}")
    
    # Sample proportionally
    subset_rows = []
    papers_per_telescope = n_papers // len(telescopes)
    
    for telescope in telescopes.index:
        telescope_rows = df[df['Id'].str.endswith(f'_{telescope}')]
        sample_size = min(papers_per_telescope, len(telescope_rows))
        sampled = telescope_rows.sample(n=sample_size, random_state=42)
        subset_rows.append(sampled)
    
    # Combine all samples
    subset_df = pd.concat(subset_rows, ignore_index=True)
    
    # If we need more papers, sample randomly from remaining
    if len(subset_df) < n_papers:
        remaining_needed = n_papers - len(subset_df)
        remaining_df = df[~df.index.isin(subset_df.index)]
        additional = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=42)
        subset_df = pd.concat([subset_df, additional], ignore_index=True)
    
    # Save subset
    subset_df.to_csv(output_file, index=False)
    
    print(f"Created test subset with {len(subset_df)} papers saved to {output_file}")
    
    # Print telescope distribution in subset
    subset_telescopes = subset_df['Id'].str.split('_', expand=True)[1].value_counts()
    print("Subset telescope distribution:")
    for telescope, count in subset_telescopes.items():
        print(f"  {telescope}: {count}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Process CSV data for telescope classification",
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
        help="Output file (defaults to input with .json extension)"
    )
    
    parser.add_argument(
        "--convert-to-json",
        action="store_true",
        help="Convert CSV to JSON format"
    )
    
    parser.add_argument(
        "--create-subset",
        type=int,
        metavar="N",
        help="Create a test subset with N papers"
    )
    
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    if args.create_subset:
        create_test_subset(args.csv_file, args.create_subset, args.output)
    elif args.convert_to_json:
        convert_csv_to_json(args.csv_file, args.output)
    else:
        # Default: just show info about the CSV file
        papers = load_csv_data(args.csv_file)
        print(f"CSV file contains {len(papers)} papers")


if __name__ == "__main__":
    main()