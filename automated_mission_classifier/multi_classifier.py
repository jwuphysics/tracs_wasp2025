"""Multi-telescope classifier for competition submission."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .clients.openai import OpenAIClient
from .clients.gpt_reranker import GPTReranker
from .processing.text_extractor import TextExtractor
from .detection import TelescopeDetector
from .analysis import TelescopeAnalyzer
from .models import MultiTelescopeResult, CompetitionOutput
from .utils.prompts import load_prompts
from .utils.cache import load_cache, save_cache

logger = logging.getLogger(__name__)


class MultiTelescopeClassifier:
    """Main classifier for multi-telescope competition analysis."""
    
    SUPPORTED_TELESCOPES = ["CHANDRA", "HST", "JWST"]
    
    def __init__(self,
                 data_file: Path,
                 output_dir: Path,
                 prompts_dir: Path = Path("./prompts"),
                 openai_key: Optional[str] = None,
                 gpt_model: str = 'gpt-5-mini',
                 reranker_model: str = 'gpt-4.1-nano',
                 top_k_snippets: int = 5,
                 context_sentences: int = 3,
                 reranker_threshold: float = 0.001,
                 limit_papers: Optional[int] = None):
        """
        Initialize the multi-telescope classifier.
        
        Args:
            data_file: Path to JSON data file with papers
            output_dir: Directory for output files
            prompts_dir: Directory containing prompt templates
            openai_key: OpenAI API key
            gpt_model: GPT model for classification
            reranker_model: GPT model for reranking
            top_k_snippets: Number of snippets to send to LLM
            context_sentences: Context window for snippet extraction
            reranker_threshold: Minimum score for snippet filtering
            limit_papers: Limit number of papers to process (for testing)
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.prompts_dir = prompts_dir
        self.limit_papers = limit_papers
        
        # Validate data file
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
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
        
        # Initialize components
        self.telescope_detector = TelescopeDetector(self.openai_client, self.prompts)
        self.telescope_analyzer = TelescopeAnalyzer(
            self.openai_client,
            self.text_extractor,
            self.gpt_reranker,
            self.prompts,
            top_k_snippets,
            reranker_threshold
        )
        
        # Setup cache files
        self.cache_files = {
            'detection': self.results_dir / "telescope_detection.json",
            'classification': self.results_dir / "telescope_classification.json",
            'results': self.results_dir / "multi_telescope_results.json"
        }
        
        logger.info(f"MultiTelescopeClassifier initialized with data: {self.data_file}")
    
    def process_papers(self) -> List[CompetitionOutput]:
        """
        Process all papers and return competition results.
        
        Returns:
            List of CompetitionOutput objects ready for CSV export
        """
        start_time = time.time()
        logger.info("Starting multi-telescope classification pipeline...")
        
        # Load papers
        papers = self._load_papers()
        if not papers:
            logger.warning("No papers to process")
            return []
        
        # Apply limit if specified
        if self.limit_papers:
            papers = papers[:self.limit_papers]
            logger.info(f"Processing limited set: {len(papers)} papers")
        
        # Process each paper
        competition_results = []
        
        for i, paper in enumerate(papers):
            bibcode = paper.get('bibcode', f'unknown_{i}')
            logger.info(f"Processing paper {i+1}/{len(papers)}: {bibcode}")
            
            try:
                result = self._process_single_paper(paper)
                if result:
                    competition_output = self._create_competition_output(result)
                    competition_results.append(competition_output)
                else:
                    logger.warning(f"Failed to process paper: {bibcode}")
                    
            except Exception as e:
                logger.error(f"Error processing paper {bibcode}: {e}")
                continue
        
        end_time = time.time()
        logger.info(f"Completed processing {len(competition_results)} papers in {end_time - start_time:.2f} seconds")
        
        return competition_results
    
    def _load_papers(self) -> List[Dict]:
        """Load papers from the JSON data file."""
        logger.info(f"Loading papers from {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Data file must contain a JSON array of papers")
            
            logger.info(f"Loaded {len(data)} papers")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load papers: {e}")
            raise
    
    def _process_single_paper(self, paper: Dict) -> Optional[MultiTelescopeResult]:
        """Process a single paper through the full pipeline."""
        bibcode = paper.get('bibcode', 'unknown')
        
        # Check cache first
        results_cache = load_cache(self.cache_files['results'])
        if bibcode in results_cache:
            logger.debug(f"Using cached result for {bibcode}")
            return MultiTelescopeResult(**results_cache[bibcode])
        
        # Combine all text sources
        text_sources = []
        for field in ['title', 'abstract', 'body']:
            content = paper.get(field, '')
            if content and isinstance(content, str) and content.strip():
                text_sources.append(content.strip())
        
        if not text_sources:
            logger.warning(f"No text content for {bibcode}")
            return None
        
        combined_text = "\n\n".join(text_sources)
        
        # Step 1: Detect telescopes
        detection_result = self.telescope_detector.detect_telescopes(combined_text, bibcode)
        detected_telescopes = detection_result.get("detected_telescopes", [])
        primary_telescope = detection_result.get("primary_telescope", "NONE")
        
        # Step 2: Classify relationship for each detected telescope
        classifications = []
        
        if detected_telescopes:
            # Analyze each detected telescope
            for telescope in detected_telescopes:
                if telescope in self.SUPPORTED_TELESCOPES:
                    classification = self.telescope_analyzer.analyze_telescope_relationship(
                        combined_text, telescope, bibcode
                    )
                    classifications.append(classification)
        
        # If no supported telescopes detected, analyze as NONE
        if not classifications:
            none_classification = self.telescope_analyzer.analyze_telescope_relationship(
                combined_text, "NONE", bibcode
            )
            classifications.append(none_classification)
            primary_telescope = "NONE"
        
        # Create result object
        result = MultiTelescopeResult(
            bibcode=bibcode,
            detected_telescopes=detected_telescopes,
            classifications=classifications,
            primary_telescope=primary_telescope
        )
        
        # Cache the result
        results_cache[bibcode] = result.model_dump()
        save_cache(self.cache_files['results'], results_cache)
        
        return result
    
    def _create_competition_output(self, result: MultiTelescopeResult) -> CompetitionOutput:
        """Convert MultiTelescopeResult to CompetitionOutput format."""
        primary_telescope = result.primary_telescope
        
        # Find classification for primary telescope
        primary_classification = None
        for classification in result.classifications:
            # Handle both dict and model formats
            if isinstance(classification, dict):
                telescope = classification.get('telescope')
            else:
                telescope = getattr(classification, 'telescope', None)
                
            if telescope == primary_telescope:
                primary_classification = classification
                break
        
        # Default values if no classification found
        if not primary_classification:
            logger.warning(f"No classification found for primary telescope {primary_telescope} in {result.bibcode}")
            primary_classification = {
                'science': False,
                'instrumentation': False,
                'mention': False,
                'not_telescope': True
            }
        
        # Extract values handling both dict and model formats
        if isinstance(primary_classification, dict):
            science = primary_classification.get('science', False)
            instrumentation = primary_classification.get('instrumentation', False)
            mention = primary_classification.get('mention', False)
            not_telescope = primary_classification.get('not_telescope', True)
        else:
            science = getattr(primary_classification, 'science', False)
            instrumentation = getattr(primary_classification, 'instrumentation', False)
            mention = getattr(primary_classification, 'mention', False)
            not_telescope = getattr(primary_classification, 'not_telescope', True)
        
        return CompetitionOutput(
            Id=result.bibcode,
            telescope=primary_telescope,
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
        
        total_papers = len(results)
        
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
                "total_papers": total_papers,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "telescope_distribution": telescope_counts,
            "classification_distribution": classification_counts,
            "telescope_percentages": {
                telescope: (count / total_papers) * 100 
                for telescope, count in telescope_counts.items()
            }
        }
        
        # Save report
        report_path = self.results_dir / "classification_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated classification report: {report_path}")
        return report