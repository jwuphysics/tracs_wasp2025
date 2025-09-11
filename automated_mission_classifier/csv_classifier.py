"""CSV-specific classifier for single telescope per row processing."""

import pandas as pd
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .clients.openai import OpenAIClient
from .clients.gpt_reranker import GPTReranker
from .processing.text_extractor import TextExtractor
from .analysis import TelescopeAnalyzer
from .models import CSVInputRow, SingleTelescopeResult, CompetitionOutput
from .utils.prompts import load_prompts
from .utils.cache import load_cache, save_cache

logger = logging.getLogger(__name__)


def parse_id_field(id_value: str) -> tuple[str, str]:
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


class CSVTelescopeClassifier:
    """Classifier for CSV format with single telescope per row."""
    
    def __init__(self,
                 csv_file: Path,
                 output_dir: Path,
                 prompts_dir: Path = Path("./prompts"),
                 openai_key: Optional[str] = None,
                 gpt_model: str = 'gpt-5-mini',
                 reranker_model: str = 'gpt-4.1-nano',
                 top_k_snippets: int = 5,
                 context_sentences: int = 3,
                 reranker_threshold: float = 0.001,
                 limit_rows: Optional[int] = None):
        """
        Initialize the CSV telescope classifier.
        
        Args:
            csv_file: Path to CSV data file
            output_dir: Directory for output files
            prompts_dir: Directory containing prompt templates
            openai_key: OpenAI API key
            gpt_model: GPT model for classification
            reranker_model: GPT model for reranking
            top_k_snippets: Number of snippets to send to LLM
            context_sentences: Context window for snippet extraction
            reranker_threshold: Minimum score for snippet filtering
            limit_rows: Limit number of rows to process (for testing)
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.prompts_dir = prompts_dir
        self.limit_rows = limit_rows
        
        # Validate data file
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        # Setup directories
        self.results_dir = output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        if not openai_key:
            import os
            openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OpenAI API key required")
            
        self.openai_client = OpenAIClient(openai_key, gpt_model)
        self.gpt_reranker = GPTReranker(self.openai_client, reranker_model)
        self.text_extractor = TextExtractor(context_sentences)
        
        # Load prompts
        self.prompts = load_prompts(self.prompts_dir)
        
        # Initialize telescope analyzer (no detector needed for CSV mode)
        self.telescope_analyzer = TelescopeAnalyzer(
            self.openai_client,
            self.text_extractor,
            self.gpt_reranker,
            self.prompts,
            top_k_snippets,
            reranker_threshold
        )
        
        # Setup cache file
        self.cache_file = self.results_dir / "csv_classification_cache.json"
        
        logger.info(f"CSVTelescopeClassifier initialized with data: {self.csv_file}")
    
    def load_csv_data(self) -> List[CSVInputRow]:
        """Load and parse CSV data."""
        logger.info(f"Loading CSV data from {self.csv_file}")
        
        df = pd.read_csv(self.csv_file)
        
        # Apply limit if specified
        if self.limit_rows:
            df = df.head(self.limit_rows)
            logger.info(f"Limited to first {len(df)} rows")
        
        rows = []
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
            
            # Create CSVInputRow object
            csv_row = CSVInputRow(
                id=row['Id'],
                bibcode=bibcode,
                telescope=telescope,
                author=safe_get_string(row.get('author')),
                year=safe_get_int(row.get('year')),
                title=safe_get_string(row.get('title')),
                abstract=safe_get_string(row.get('abstract')),
                body=safe_get_string(row.get('body')),
                acknowledgments=safe_get_string(row.get('acknowledgments')),
                grants=safe_get_string(row.get('grants'))
            )
            
            rows.append(csv_row)
        
        logger.info(f"Loaded {len(rows)} CSV rows")
        return rows
    
    def process_rows(self) -> List[CompetitionOutput]:
        """
        Process all CSV rows and return competition results.
        
        Returns:
            List of CompetitionOutput objects ready for CSV export
        """
        start_time = time.time()
        logger.info("Starting CSV telescope classification pipeline...")
        
        # Load CSV data
        rows = self.load_csv_data()
        if not rows:
            logger.warning("No rows to process")
            return []
        
        # Process each row
        competition_results = []
        cache = load_cache(self.cache_file)
        
        for i, row in enumerate(rows):
            logger.info(f"Processing row {i+1}/{len(rows)}: {row.id}")
            
            try:
                # Check cache first
                if row.id in cache:
                    logger.debug(f"Using cached result for {row.id}")
                    result = SingleTelescopeResult(**cache[row.id])
                else:
                    result = self._process_single_row(row)
                    if result:
                        # Cache the result
                        cache[row.id] = result.model_dump()
                        save_cache(self.cache_file, cache)
                
                if result:
                    competition_output = self._create_competition_output(result)
                    competition_results.append(competition_output)
                else:
                    logger.warning(f"Failed to process row: {row.id}")
                    
            except Exception as e:
                logger.error(f"Error processing row {row.id}: {e}")
                continue
        
        end_time = time.time()
        logger.info(f"Completed processing {len(competition_results)} rows in {end_time - start_time:.2f} seconds")
        
        return competition_results
    
    def _process_single_row(self, row: CSVInputRow) -> Optional[SingleTelescopeResult]:
        """Process a single CSV row through the classification pipeline."""
        
        # Handle _NONE IDs without LLM processing
        if row.telescope == "NONE":
            logger.info(f"Skipping LLM processing for _NONE entry: {row.id}")
            from .models import TelescopeClassificationModel
            
            classification = TelescopeClassificationModel(
                telescope="NONE",
                science=False,
                instrumentation=False,
                mention=False,
                not_telescope=False,
                quotes=[],
                reasoning="Automatically classified as NONE - no telescope relationship"
            )
            
            result = SingleTelescopeResult(
                id=row.id,
                bibcode=row.bibcode,
                telescope=row.telescope,
                classification=classification
            )
            
            return result
        
        # Combine all text sources
        text_sources = []
        for field in ['title', 'abstract', 'body', 'acknowledgments', 'grants']:
            content = getattr(row, field, '')
            if content and isinstance(content, str) and content.strip():
                text_sources.append(content.strip())
        
        if not text_sources:
            logger.warning(f"No text content for {row.id}")
            return None
        
        combined_text = "\n\n".join(text_sources)
        
        # Classify relationship for the specified telescope
        classification = self.telescope_analyzer.analyze_telescope_relationship(
            combined_text, row.telescope, row.bibcode
        )
        
        if not classification:
            logger.warning(f"Failed to classify {row.id} for telescope {row.telescope}")
            return None
        
        # Create result object
        result = SingleTelescopeResult(
            id=row.id,
            bibcode=row.bibcode,
            telescope=row.telescope,
            classification=classification
        )
        
        return result
    
    def _create_competition_output(self, result: SingleTelescopeResult) -> CompetitionOutput:
        """Convert SingleTelescopeResult to CompetitionOutput format."""
        # Handle _NONE telescope entries with all False labels
        if result.telescope == "NONE":
            return CompetitionOutput(
                Id=result.id,  # Keep the combined ID format
                telescope=result.telescope,
                science=False,
                instrumentation=False,
                mention=False,
                not_telescope=False
            )
        
        classification = result.classification
        
        # Extract values handling both dict and model formats
        if isinstance(classification, dict):
            science = classification.get('science', False)
            instrumentation = classification.get('instrumentation', False)
            mention = classification.get('mention', False)
            not_telescope = classification.get('not_telescope', True)
        else:
            science = getattr(classification, 'science', False)
            instrumentation = getattr(classification, 'instrumentation', False)
            mention = getattr(classification, 'mention', False)
            not_telescope = getattr(classification, 'not_telescope', True)
        
        return CompetitionOutput(
            Id=result.id,  # Keep the combined ID format
            telescope=result.telescope,
            science=science,
            instrumentation=instrumentation,
            mention=mention,
            not_telescope=not_telescope
        )
    
    def save_competition_csv(self, results: List[CompetitionOutput], filename: str = "submission.csv") -> Path:
        """Save results to competition CSV format."""
        output_path = self.results_dir / filename
        
        # Convert to DataFrame
        data = [result.model_dump() for result in results]
        df = pd.DataFrame(data)
        
        # Ensure proper column order
        columns = ['Id', 'telescope', 'science', 'instrumentation', 'mention', 'not_telescope']
        df = df[columns]
        
        # Save CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved competition CSV with {len(results)} entries to {output_path}")
        
        return output_path
    
    def generate_report(self, results: List[CompetitionOutput]) -> Dict:
        """Generate summary report of classification results."""
        if not results:
            return {"error": "No results to analyze"}
        
        total_rows = len(results)
        
        # Telescope distribution
        telescope_counts = {}
        for result in results:
            telescope = result.telescope
            telescope_counts[telescope] = telescope_counts.get(telescope, 0) + 1
        
        # Classification distribution
        classification_counts = {
            'science': sum(1 for r in results if r.science),
            'instrumentation': sum(1 for r in results if r.instrumentation),
            'mention': sum(1 for r in results if r.mention),
            'not_telescope': sum(1 for r in results if r.not_telescope)
        }
        
        report = {
            "summary": {
                "total_rows": total_rows,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "telescope_distribution": telescope_counts,
            "classification_distribution": classification_counts,
            "telescope_percentages": {
                telescope: (count / total_rows) * 100 
                for telescope, count in telescope_counts.items()
            }
        }
        
        # Save report
        report_path = self.results_dir / "csv_classification_report.json"
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated classification report: {report_path}")
        return report