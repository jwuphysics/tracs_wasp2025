"""Command-line interface for Multi-Telescope Classifier."""

import argparse
import logging
import sys
from pathlib import Path

from .multi_classifier import MultiTelescopeClassifier
from .csv_classifier import CSVTelescopeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-telescope classification system for WASP-2025 competition. Identifies telescopes in papers and classifies relationships (science/instrumentation/mention/not_telescope).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input file (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-file",
        type=Path,
        help="Path to JSON data file containing paper records"
    )
    input_group.add_argument(
        "--csv-file",
        type=Path,
        help="Path to CSV data file with combined Id field (bibcode_telescope)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./output"),
        help="Directory for output files (CSV and reports)"
    )
    parser.add_argument(
        "--prompts-dir", "-p",
        type=Path,
        default=Path("./prompts"), 
        help="Directory containing prompt template files"
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-5-mini", 
        help="GPT model for telescope classification"
    )
    parser.add_argument(
        "--reranker-model",
        default="gpt-4.1-nano",
        help="GPT model for snippet reranking"
    )
    parser.add_argument(
        "--top-k-snippets",
        type=int,
        default=5, 
        help="Number of top reranked snippets to send to the LLM"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=3, 
        help="Number of sentences before and after a keyword to include in snippets"
    )
    parser.add_argument(
        "--reranker-threshold",
        type=float,
        default=0.001,
        help="Minimum reranker score for snippets to proceed with LLM analysis"
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        help="Limit processing to the first N papers (useful for testing, JSON mode)"
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        help="Limit processing to the first N rows (useful for testing, CSV mode)"
    )
    parser.add_argument(
        "--openai-key", 
        help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--output-filename",
        default="submission.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.data_file and not args.data_file.exists():
        parser.error(f"JSON data file not found: {args.data_file}")
    
    if args.csv_file and not args.csv_file.exists():
        parser.error(f"CSV data file not found: {args.csv_file}")
        
    if args.limit_papers is not None and args.limit_papers < 1:
        parser.error("--limit-papers must be a positive integer")
    
    if args.limit_rows is not None and args.limit_rows < 1:
        parser.error("--limit-rows must be a positive integer")
    
    # Determine processing mode
    csv_mode = args.csv_file is not None

    # Create directories
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not args.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {args.prompts_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    # Initialize and run classifier
    try:
        if csv_mode:
            logger.info("Initializing CSV telescope classifier...")
            
            classifier = CSVTelescopeClassifier(
                csv_file=args.csv_file,
                output_dir=args.output_dir,
                prompts_dir=args.prompts_dir,
                openai_key=args.openai_key,
                gpt_model=args.gpt_model,
                reranker_model=args.reranker_model,
                top_k_snippets=args.top_k_snippets,
                context_sentences=args.context_sentences,
                reranker_threshold=args.reranker_threshold,
                limit_rows=args.limit_rows,
            )

            logger.info("Processing CSV rows...")
            results = classifier.process_rows()
            processing_unit = "rows"
        else:
            logger.info("Initializing multi-telescope classifier...")
            
            classifier = MultiTelescopeClassifier(
                data_file=args.data_file,
                output_dir=args.output_dir,
                prompts_dir=args.prompts_dir,
                openai_key=args.openai_key,
                gpt_model=args.gpt_model,
                reranker_model=args.reranker_model,
                top_k_snippets=args.top_k_snippets,
                context_sentences=args.context_sentences,
                reranker_threshold=args.reranker_threshold,
                limit_papers=args.limit_papers,
            )

            logger.info("Processing papers...")
            results = classifier.process_papers()
            processing_unit = "papers"
        
        if not results:
            logger.warning(f"No {processing_unit} were successfully processed")
            sys.exit(1)
        
        # Save competition CSV
        csv_path = classifier.save_competition_csv(results, args.output_filename)
        logger.info(f"Competition CSV saved to: {csv_path}")
        
        # Generate and save report
        report = classifier.generate_report(results)
        logger.info(f"Classification report generated")
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"- Total {processing_unit} processed: {len(results)}")
        print(f"- Output CSV: {csv_path}")
        
        if 'telescope_distribution' in report:
            print(f"\nTelescope Distribution:")
            for telescope, count in report['telescope_distribution'].items():
                percentage = report['telescope_percentages'][telescope]
                print(f"  {telescope}: {count} {processing_unit} ({percentage:.1f}%)")
        
        if 'classification_distribution' in report:
            print(f"\nClassification Distribution:")
            for category, count in report['classification_distribution'].items():
                print(f"  {category}: {count} {processing_unit}")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    logger.info("Multi-telescope classification completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()