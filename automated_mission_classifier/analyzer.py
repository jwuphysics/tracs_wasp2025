"""Main analyzer class that orchestrates the analysis pipeline."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .clients.openai import OpenAIClient
from .clients.cohere import CohereClient
from .clients.gpt_reranker import GPTReranker
from .processing.text_extractor import TextExtractor
from .analysis.science import ScienceAnalyzer
from .reporting import ReportGenerator
from .utils.cache import load_cache, save_cache
from .utils.prompts import load_prompts

logger = logging.getLogger(__name__)


class AutomatedMissionClassifier:
    """Main analyzer class for automated mission classification."""
    
    def __init__(self,
                 output_dir: Path,
                 data_file: Path,
                 mission: str,
                 bibcode: Optional[str] = None,
                 batch_mode: Optional[str] = None,
                 prompts_dir: Path = Path("./prompts"),
                 science_threshold: float = 0.5,
                 reranker_threshold: float = 0.0,
                 openai_key: Optional[str] = None,
                 cohere_key: Optional[str] = None,
                 gpt_model: str = 'gpt-5-mini',
                 cohere_reranker_model: str = 'rerank-v3.5', 
                 top_k_snippets: int = 15,
                 context_sentences: int = 3,
                 reprocess: bool = False,
                 use_gpt_reranker: bool = True,
                 limit_papers: Optional[int] = None):
        """Initialize the automated mission classifier."""
        
        # Validate that exactly one mode is provided
        modes_provided = sum([bool(bibcode), bool(batch_mode)])
        if modes_provided != 1:
            raise ValueError("Exactly one of 'bibcode' or 'batch_mode' must be provided.")

        self.bibcode = bibcode
        self.batch_mode = batch_mode  # Can be file path or list of bibcodes
        self.run_mode = "batch" if batch_mode else "single"
        self.data_file = data_file
        self.mission = mission.upper()  # Store mission name
        
        self.reprocess = reprocess
        self.science_threshold = science_threshold
        self.reranker_threshold = reranker_threshold
        self.gpt_model = gpt_model
        self.cohere_reranker_model = cohere_reranker_model 
        self.top_k_snippets = top_k_snippets
        self.context_sentences = context_sentences
        self.use_gpt_reranker = use_gpt_reranker
        self.limit_papers = limit_papers

        # Setup API keys
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.cohere_key = cohere_key or os.getenv('COHERE_API_KEY')
        
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY must be provided (as argument or environment variable)")
            
        # Validate data file exists
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Initialize clients
        self.openai_client = OpenAIClient(self.openai_key, self.gpt_model)
        self.cohere_client = CohereClient(self.cohere_key, self.cohere_reranker_model)
        self.gpt_reranker = GPTReranker(self.openai_client, 'gpt-5-nano') if self.use_gpt_reranker else None
        
        # Create directories
        self.output_dir = output_dir 
        self.results_dir = output_dir / "results"
        self.prompts_dir = prompts_dir
        self._setup_directories()
        
        # Initialize components
        self.text_extractor = TextExtractor(self.context_sentences)
        
        # Load prompts
        self.prompts = load_prompts(self.prompts_dir)
        
        # Initialize analyzers - only science analyzer needed
        self.science_analyzer = ScienceAnalyzer(
            self.openai_client, self.cohere_client, self.text_extractor,
            self.prompts, self.mission, self.top_k_snippets, self.reranker_threshold,
            self.gpt_reranker
        )
        
        # Setup cache files
        if self.batch_mode:
            cache_prefix = f"{self.mission.lower()}_batch"
        else:
            cache_prefix = f"{self.mission.lower()}_{self.bibcode.replace('/', '_')}" if self.bibcode else "single_run"
            
        self.cache_files = {
            'science': self.results_dir / f"{cache_prefix}_science.json",
            'skipped': self.results_dir / f"{cache_prefix}_skipped.json",
            'snippets': self.results_dir / f"{cache_prefix}_snippets.json",
            'papers': self.results_dir / f"{cache_prefix}_papers.json",
            'downloaded': self.results_dir / f"{cache_prefix}_downloaded.json"
        }
        
        # Initialize report generator
        model_config = {
            "gpt_model": self.gpt_model,
            "reranker_type": "GPT-5-nano" if self.use_gpt_reranker else "Cohere",
            "cohere_reranker_model": self.cohere_reranker_model if self.cohere_client.client else "N/A (Cohere unavailable)",
            "top_k_snippets": self.top_k_snippets,
            "context_sentences": self.context_sentences,
            "prompts_directory": str(self.prompts_dir.resolve()),
            "mission": self.mission
        }
        self.report_generator = ReportGenerator(
            self.results_dir, self.science_threshold, model_config
        )

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_paper_data(self) -> List[Dict]:
        """Load papers from the JSON data file."""
        logger.info(f"Loading paper data from {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Data file must contain a JSON array of papers")
            
        logger.info(f"Loaded {len(data)} papers from data file")
        return data
        
    def _get_papers_to_process(self) -> List[Dict]:
        """Get the list of papers to process based on mode and filters."""
        all_papers = self._load_paper_data()
        
        if self.run_mode == "single":
            # Find single paper by bibcode
            for paper in all_papers:
                if paper.get('bibcode') == self.bibcode:
                    return [paper]
            raise ValueError(f"Paper with bibcode '{self.bibcode}' not found in data file")
        else:
            # Batch mode - process based on batch_mode parameter
            if isinstance(self.batch_mode, str) and Path(self.batch_mode).exists():
                # Read bibcodes from file
                with open(self.batch_mode, 'r') as f:
                    target_bibcodes = [line.strip() for line in f if line.strip()]
            elif isinstance(self.batch_mode, (list, tuple)):
                # List of bibcodes provided
                target_bibcodes = list(self.batch_mode)
            else:
                # Process all papers
                target_bibcodes = None
                
            if target_bibcodes:
                # Filter papers by bibcodes
                papers = [p for p in all_papers if p.get('bibcode') in target_bibcodes]
                logger.info(f"Found {len(papers)} papers matching {len(target_bibcodes)} target bibcodes")
            else:
                papers = all_papers
                
            # Apply limit if specified
            if self.limit_papers is not None:
                original_count = len(papers)
                papers = papers[:self.limit_papers]
                logger.info(f"Limiting processing to first {len(papers)} papers out of {original_count} total papers")
                
            return papers

    def _has_body_text(self, paper: Dict) -> bool:
        """Check if paper has any text content available."""
        for field in ['title', 'abstract', 'body']:
            content = paper.get(field, '')
            if isinstance(content, str) and len(content.strip()) > 0:
                return True
        return False

    def _is_skipped(self, paper: Dict) -> bool:
        """Check if paper was previously skipped."""
        if self.run_mode == "single":
            return False

        skipped = load_cache(self.cache_files['skipped'])
        return not self.reprocess and paper.get('bibcode') in skipped

    def _mark_as_skipped(self, paper: Dict, reason: str, save_to_cache: bool = True):
        """Mark a paper as skipped with the given reason."""
        bibcode = paper.get('bibcode', 'unknown')
        logger.warning(f"Skipping paper {bibcode}: {reason}")

        if save_to_cache and self.run_mode == "batch":
            skipped = load_cache(self.cache_files['skipped'])
            skipped[bibcode] = {
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(self.cache_files['skipped'], skipped)

    def _needs_analysis(self, paper: Dict, cache_key: str) -> bool:
        """Generic check if paper needs analysis based on cache and reprocess flag."""
        if self.run_mode == 'single':
            return True
        analyzed = load_cache(self.cache_files[cache_key])
        return self.reprocess or paper.get('bibcode') not in analyzed

    def _process_paper(self, paper: Dict[str, str]) -> bool:
        """Download and convert a paper. Returns True if successful."""
        arxiv_id = paper['arxiv_id']
        
        # Download
        if not self.downloader.download_paper(arxiv_id, self.reprocess):
            self._mark_as_skipped(paper, "Download failed", save_to_cache=True)
            return False
            
        # Convert
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        if not self.converter.convert_to_text(arxiv_id, pdf_path, self.reprocess):
            self._mark_as_skipped(paper, "PDF conversion failed", save_to_cache=True)
            return False
            
        # Update download cache in both modes
        downloaded = load_cache(self.cache_files['downloaded'])
        downloaded[arxiv_id] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
        save_cache(self.cache_files['downloaded'], downloaded)
            
        return True

    def run_batch(self):
        """Main execution pipeline for batch mode."""
        if self.run_mode != "batch":
            logger.error("run_batch called in non-batch mode.")
            return

        start_time = time.time()
        # Generate batch identifier from batch mode file if provided, otherwise use data file
        if isinstance(self.batch_mode, str) and Path(self.batch_mode).exists():
            batch_identifier = f"{self.mission.lower()}_{Path(self.batch_mode).stem}"
        else:
            batch_identifier = f"{self.mission.lower()}_{self.data_file.stem}"
        logger.info(f"Starting {self.mission} mission classification batch analysis for {batch_identifier}...")

        try:
            # Get papers to process from data file
            papers = self._get_papers_to_process()
            if not papers:
                logger.warning("No papers found to process. Exiting.")
                self.report_generator.generate_report(batch_identifier, self.cache_files, self.limit_papers)
                return
            
            # Store paper metadata in cache
            papers_cache = load_cache(self.cache_files['papers'])
            for paper in papers:
                bibcode = paper.get('bibcode', 'unknown')
                if bibcode not in papers_cache:
                    papers_cache[bibcode] = {
                        'bibcode': bibcode,
                        'title': paper.get('title', '')
                    }
            save_cache(self.cache_files['papers'], papers_cache)

            # Process each paper
            for i, paper in enumerate(papers):
                bibcode = paper.get('bibcode', 'unknown')
                logger.info(f"Processing paper {i+1}/{len(papers)}: {bibcode}")

                if self._is_skipped(paper):
                    logger.info(f"Paper {bibcode} was previously skipped. Skipping.")
                    continue

                # Check if paper has any text content
                if not self._has_body_text(paper):
                    self._mark_as_skipped(paper, "No text content available")
                    continue

                # Analyze for science content
                science_result = None                
                if self._needs_analysis(paper, 'science'):
                    science_result = self._analyze_paper(paper)
                    if science_result and "error" not in science_result:
                        science_cache = load_cache(self.cache_files['science'])
                        science_cache[bibcode] = science_result
                        save_cache(self.cache_files['science'], science_cache)
                else:
                    science_cache = load_cache(self.cache_files['science'])
                    science_result = science_cache.get(bibcode)
                    if science_result and isinstance(science_result, dict):
                        logger.info(f"Using cached science analysis for {bibcode}")
                    else:
                        logger.warning(f"Cache logic error: Science cache missing or invalid for {bibcode}. Re-analyzing.")
                        science_result = self._analyze_paper(paper)
                        if science_result and "error" not in science_result:
                            science_cache[bibcode] = science_result
                            save_cache(self.cache_files['science'], science_cache)

                # Check science score
                current_science_score = -1.0
                if science_result and isinstance(science_result, dict) and "error" not in science_result:
                    current_science_score = science_result.get("science", science_result.get("jwstscience", -1.0))
                else:
                    logger.warning(f"Science analysis failed for {bibcode}.")
                    continue

                # Log result for mission classification
                if current_science_score >= self.science_threshold:
                    logger.info(f"Paper {bibcode} classified as science ({current_science_score:.2f} >= {self.science_threshold})")
                else:
                    logger.info(f"Paper {bibcode} classified as non-science ({current_science_score:.2f} < {self.science_threshold})")

            # Generate final summary report
            self.report_generator.generate_report(batch_identifier, self.cache_files, self.limit_papers)
            
            # Generate CSV report
            self.report_generator.generate_csv_report(batch_identifier, self.cache_files, self.limit_papers)
            
            # Generate competition CSV format
            self.report_generator.generate_competition_csv(batch_identifier, self.cache_files, self.limit_papers)

            end_time = time.time()
            logger.info(f"Analysis complete for {self.mission} mission classification in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logger.exception(f"An unhandled error occurred during the run: {e}")
            try:
                logger.info("Attempting to generate partial report after error...")
                self.report_generator.generate_report(batch_identifier, self.cache_files)
            except Exception as report_err:
                logger.error(f"Failed to generate partial report: {report_err}")
            raise

    def _analyze_paper(self, paper: Dict) -> Dict:
        """Analyze a single paper using combined text sources for telescope classification."""
        bibcode = paper.get('bibcode', 'unknown')
        
        # Combine all available text sources
        text_sources = []
        for field in ['title', 'abstract', 'body']:
            content = paper.get(field, '')
            if content and isinstance(content, str) and content.strip():
                text_sources.append(content.strip())
        
        if not text_sources:
            logger.warning(f"No text content available for {bibcode}")
            return {"telescope": "NONE", "science": False, "instrumentation": False, 
                   "mention": False, "not_telescope": True, "reason": "Analysis failed: No text content", 
                   "quotes": [], "error": "missing_text"}
        
        combined_text = "\n\n".join(text_sources)
        
        # Step 1: Identify primary telescope if not specified
        detected_telescope = paper.get('telescope', 'UNKNOWN')
        if detected_telescope == 'UNKNOWN' or not detected_telescope:
            detected_telescope = self._identify_telescope(combined_text)
        
        # Step 2: Extract relevant snippets for the detected telescope
        if detected_telescope == "NONE":
            # For NONE papers, no telescope-specific analysis is needed
            # Return a default NONE classification
            logger.info(f"Paper {bibcode} classified as NONE (no telescope detected)")
            return {"telescope": "NONE", "science": False, "instrumentation": False,
                   "mention": False, "not_telescope": True, "quotes": [], 
                   "reason": "No telescope keywords found in text - classified as NONE"}

        telescope_keywords = self.science_analyzer.TELESCOPE_KEYWORDS.get(detected_telescope, [])
        if not telescope_keywords:
            # If telescope not supported, try generic keywords from supported telescopes
            telescope_keywords = []
            for tel, keywords in self.science_analyzer.TELESCOPE_KEYWORDS.items():
                if tel in ['CHANDRA', 'HST', 'JWST']:
                    telescope_keywords.extend(keywords)
        
        keywords_lower = sorted([k.lower() for k in telescope_keywords], key=len, reverse=True)
        all_snippets = self.science_analyzer.text_extractor.extract_relevant_snippets(
            combined_text, keywords_lower
        )

        if not all_snippets:
            logger.info(f"No relevant keywords found for telescope analysis in {bibcode}.")
            return {"telescope": detected_telescope, "science": False, "instrumentation": False,
                   "mention": False, "not_telescope": True, "quotes": [],
                   "reason": f"No relevant keywords for {detected_telescope} found in text."}

        # Skip reranking for short texts (e.g., abstract-only conference papers)
        # If combined text is short and we have few snippets, just use them all
        skip_reranking = len(combined_text) < 3000 and len(all_snippets) <= self.science_analyzer.top_k_snippets

        if skip_reranking:
            logger.info(f"Skipping reranking for {bibcode}: short text ({len(combined_text)} chars) with {len(all_snippets)} snippets")
            filtered_snippets = [{"snippet": s, "score": None} for s in all_snippets]
        else:
            # Rerank snippets
            rerank_query = self.science_analyzer.prompts.get('rerank_science_query')
            if not rerank_query:
                logger.error("Rerank science query prompt not found or empty.")
                return {"science": -1.0, "reason": "Analysis failed: Missing rerank science query prompt",
                       "quotes": [], "error": "prompt_missing"}

            rerank_query = rerank_query.format(telescope=detected_telescope)

            # Use GPT reranker if available, otherwise fall back to Cohere
            if self.science_analyzer.gpt_reranker:
                reranked_data = self.science_analyzer.gpt_reranker.rerank_snippets(
                    rerank_query, all_snippets, self.science_analyzer.top_k_snippets
                )
            else:
                reranked_data = self.science_analyzer.cohere_client.rerank_snippets(
                    rerank_query, all_snippets, self.science_analyzer.top_k_snippets
                )

            if not reranked_data:
                logger.warning(f"Reranking produced no snippets for {bibcode}. Skipping LLM analysis.")
                return {"science": 0.0, "quotes": [],
                       "reason": "Keyword snippets found but none survived reranking/filtering."}

            # Check reranker threshold for top score
            top_score = reranked_data[0].get('score')
            if top_score is not None and top_score < self.science_analyzer.reranker_threshold:
                logger.info(f"Skipping LLM science analysis for {bibcode}: Top reranker score ({top_score:g}) below threshold ({self.science_analyzer.reranker_threshold}).")
                return {
                    "science": 0.0,
                    "quotes": [],
                    "reason": f"Skipped LLM analysis: Top reranker score ({top_score:g}) was below the threshold ({self.science_analyzer.reranker_threshold}).",
                }

            # Filter snippets above threshold and respect top_k limit
            filtered_snippets = []
            for item in reranked_data:
                score = item.get('score')
                if score is not None and score >= self.science_analyzer.reranker_threshold:
                    filtered_snippets.append(item)
                if len(filtered_snippets) >= self.science_analyzer.top_k_snippets:
                    break

            if not filtered_snippets:
                logger.info(f"No snippets above threshold ({self.science_analyzer.reranker_threshold}) for {bibcode}.")
                return {
                    "science": 0.0,
                    "quotes": [],
                    "reason": f"No snippets scored above the threshold ({self.science_analyzer.reranker_threshold}).",
                }

            logger.info(f"Using {len(filtered_snippets)} snippets above threshold for {bibcode} (filtered from {len(reranked_data)})")

        # Prepare LLM input with scores
        formatted_snippets = []
        for i, item in enumerate(filtered_snippets):
            score = item.get('score')
            snippet = item.get('snippet', '')

            if score is not None:
                formatted_snippets.append(f"Excerpt {i+1} (relevance: {score:.3f}):\n{snippet}")
            else:
                formatted_snippets.append(f"Excerpt {i+1}:\n{snippet}")

        snippets_text = "\n---\n".join(formatted_snippets)
        max_chars = 50000
        if len(snippets_text) > max_chars:
            logger.warning(f"Total snippet text for {bibcode} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.science_analyzer.prompts.get('science_system')
        user_prompt_template = self.science_analyzer.prompts.get('science_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"Science prompts not loaded correctly for {bibcode}. Check prompts directory.")
            return {"science": -1.0, "reason": "Analysis failed: Prompts missing", 
                   "quotes": [], "error": "prompt_missing"}
        
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text, telescope=detected_telescope)
        except KeyError as e:
            logger.error(f"Failed to format science user prompt - missing placeholder {e}")
            return {"telescope": detected_telescope, "science": False, "instrumentation": False,
                   "mention": False, "not_telescope": True, "reason": "Analysis failed: Prompt formatting error", 
                   "quotes": [], "error": "prompt_format_error"}
        
        # Call LLM using telescope classification
        llm_result = self.science_analyzer.openai_client.call_telescope_classification(
            system_prompt, user_prompt, detected_telescope
        )

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            return {"telescope": detected_telescope, "science": False, "instrumentation": False,
                   "mention": False, "not_telescope": True, "reason": error_reason, "quotes": [], "error": error_type}

        # Ensure telescope is set
        if "telescope" not in llm_result:
            llm_result["telescope"] = detected_telescope
        
        return llm_result

    def _identify_telescope(self, combined_text: str) -> str:
        """Identify which telescope a paper primarily discusses."""
        if not combined_text:
            return "NONE"
            
        # Use LLM to identify telescope if available
        telescope_id_system = self.science_analyzer.prompts.get('telescope_identification_system')
        telescope_id_user = self.science_analyzer.prompts.get('telescope_identification_user')
        
        if telescope_id_system and telescope_id_user:
            try:
                # Limit text length for telescope identification
                text_for_id = combined_text[:10000] if len(combined_text) > 10000 else combined_text
                user_prompt = telescope_id_user.format(snippets_text=text_for_id)
                
                result = self.science_analyzer.openai_client.identify_telescope(
                    telescope_id_system, user_prompt
                )
                
                if result and "telescope" in result:
                    identified = result["telescope"].upper()
                    if identified in ["CHANDRA", "HST", "JWST", "NONE"]:
                        return identified
                        
            except Exception as e:
                logger.warning(f"Telescope identification failed: {e}")
        
        # Fallback: simple keyword matching for actual telescopes only
        text_lower = combined_text.lower()
        
        # Check for specific telescopes in order of specificity
        for telescope, keywords in self.science_analyzer.TELESCOPE_KEYWORDS.items():
            if telescope in ["CHANDRA", "HST", "JWST"]:
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        return telescope
        
        # Return "NONE" only when no actual telescope keywords are found
        return "NONE"

    def process_single_paper(self, bibcode: str):
        """Processes a single paper by bibcode and prints results to stdout."""
        start_time = time.time()
        logger.info(f"Starting SINGLE {self.mission} mission classification analysis for bibcode: {bibcode}")

        final_output = {
            "bibcode": bibcode,
            "mission": self.mission,
            "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "Started",
            "science_analysis": None,
            "error_info": None
        }

        try:
            # Get papers to process (should return single paper)
            papers = self._get_papers_to_process()
            if not papers:
                logger.error(f"Paper with bibcode '{bibcode}' not found in data file.")
                final_output["status"] = "Error: Paper Not Found"
                final_output["error_info"] = f"Bibcode '{bibcode}' not found in data file."
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return
                
            paper = papers[0]  # Should be exactly one paper

            # Analyze science content
            logger.info(f"Analyzing science content for single paper {bibcode}")
            science_result = self._analyze_paper(paper)
            final_output["science_analysis"] = science_result 

            current_science_score = -1.0
            if science_result and isinstance(science_result, dict) and "error" not in science_result:
                current_science_score = science_result.get("science", science_result.get("jwstscience", -1.0))
                final_output["status"] = "Complete" 
                final_output["classification"] = "science" if current_science_score >= self.science_threshold else "non-science"
                final_output["confidence_score"] = current_science_score
            else:
                logger.error(f"Science analysis failed for {bibcode}. See results.")
                final_output["status"] = "Error: Science Analysis Failed"
                final_output["error_info"] = science_result.get("reason", "Science analysis failed") if science_result else "Science analysis failed"
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return

        except Exception as e:
            logger.exception(f"Unhandled error during single paper processing for {bibcode}: {e}")
            final_output["status"] = "Error: Unhandled Exception"
            final_output["error_info"] = f"Unexpected error: {str(e)}"

        try:
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        except Exception as json_e:
            logger.error(f"Failed to serialize final result to JSON for {bibcode}: {json_e}")
            print(f'{{"error": "json_serialization_failed", "bibcode": "{bibcode}", "message": "{str(json_e)}"}}')