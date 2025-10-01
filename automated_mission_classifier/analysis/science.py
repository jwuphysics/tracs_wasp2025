"""Mission science content analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..clients.openai import OpenAIClient
from ..clients.cohere import CohereClient
from ..clients.gpt_reranker import GPTReranker
from ..processing.text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class ScienceAnalyzer:
    """Analyzes papers for mission-specific science content."""
    
    # Telescope-specific keyword mappings for TRACS dataset
    TELESCOPE_KEYWORDS = {
        "CHANDRA": [
            "chandra", "cxc", "cxo", "axaf", "chandra x-ray observatory", 
            "advanced x-ray astrophysics facility", "chandra xray observatory",
            "acis", "hrc", "hetg", "letg", "hrma", "pcad", "ephin",
            "chandra data", "chandra observation", "chandra observations",
            "chandra image", "chandra images", "chandra spectrum", "chandra spectra"
        ],
        "HST": [
            "hst", "hubble", "hubble space telescope",
            "wfc3", "acs", "stis", "cos", "nicmos", 
            "wfpc2", "foc", "fos", "ghrs", "hsp", "wfpc",
            "hubble data", "hubble observations", "hubble images",
            "hubble photometry", "hst data", "hst observations"
        ],
        "JWST": [
            "jwst", "james webb space telescope", "webb", "ngst",
            "next generation space telescope", "james webb",
            "nircam", "nirspec", "miri", "niriss", "fgs",
            "jwst data", "webb data", "jwst observations", "webb observations"
        ],
        # Legacy support
        "TESS": [
            "tess", "transiting exoplanet survey satellite",
            "tess data", "tess observations", "tess photometry",
            "tess light curve", "tic", "toi"
        ],
        "GALEX": [
            "galex", "galaxy evolution explorer",
        ],
        "PANSTARRS": [
            "pan-starrs", "panstarrs", "panoramic survey telescope",
            "ps1", "pan-starrs1"
        ],
        "FUSE": [
            "fuse", "far ultraviolet spectroscopic explorer"
        ],
        "EUVE": [
            "euve", "extreme ultraviolet explorer"
        ],
        "Kepler": [
            "kepler", "kepler space telescope",
        ],
        "K2": [
            "kepler", "k2", 
        ],
        "Roman": [
            "roman", "roman space telescope", "nancy grace roman space telescope",
            "wfirst", "wide-field infrared survey telescope", "WFI"
        ],
        "IUE": [
            "iue", "international ultraviolet explorer"
        ]
    }
    
    # Legacy alias for backward compatibility
    MISSION_KEYWORDS = TELESCOPE_KEYWORDS
    
    def __init__(self, 
                 openai_client: OpenAIClient,
                 cohere_client: CohereClient,
                 text_extractor: TextExtractor,
                 prompts: Dict[str, str],
                 mission: str = "JWST",
                 top_k_snippets: int = 15,
                 reranker_threshold: float = 0.0,
                 gpt_reranker: Optional[GPTReranker] = None):
        self.openai_client = openai_client
        self.cohere_client = cohere_client
        self.gpt_reranker = gpt_reranker
        self.text_extractor = text_extractor
        self.prompts = prompts
        self.mission = mission.upper()
        self.top_k_snippets = top_k_snippets
        self.reranker_threshold = reranker_threshold
        
        # Set telescope-specific keywords
        if self.mission == "MULTI":
            # For multi-telescope mode, combine all keywords
            self.science_keywords = []
            for tel in ["CHANDRA", "HST", "JWST"]:
                self.science_keywords.extend(self.TELESCOPE_KEYWORDS[tel])
        elif self.mission not in self.TELESCOPE_KEYWORDS:
            raise ValueError(f"Unsupported telescope: {self.mission}. Supported telescopes: {list(self.TELESCOPE_KEYWORDS.keys())}")
        else:
            self.science_keywords = self.TELESCOPE_KEYWORDS[self.mission]
            
        self.science_keywords_lower = sorted(
            [k.lower() for k in self.science_keywords], 
            key=len, reverse=True
        )
        
    def analyze(self, paper_id: str, text_path: Path) -> Dict:
        """Analyze paper for mission science content using extracted snippets."""
        logger.info(f"Analyzing {self.mission} science content for {paper_id}")
        
        if not text_path.exists():
            logger.warning(f"Text file not found for {paper_id}, cannot analyze science.")
            return {"science": -1.0, "reason": "Analysis failed: Text file missing", 
                   "quotes": [], "error": "missing_text_file"}

        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {text_path}: {e}")
            return {"science": -1.0, "reason": "Analysis failed: Cannot read text file", 
                   "quotes": [], "error": "read_error"}

        # Extract snippets
        all_snippets = self.text_extractor.extract_relevant_snippets(
            paper_text, self.science_keywords_lower
        )

        if not all_snippets:
            logger.info(f"No relevant keywords found for science analysis in {paper_id}.")
            return {"science": 0.0, "quotes": [], 
                   "reason": f"No relevant keywords for {self.mission} found in text."}

        # Rerank snippets
        rerank_query = self.prompts.get('rerank_science_query')
        if not rerank_query:
            logger.error("Rerank science query prompt ('rerank_science_query.txt') not found or empty.")
            return {"science": -1.0, "reason": "Analysis failed: Missing rerank science query prompt", 
                   "quotes": [], "error": "prompt_missing"}

        rerank_query = rerank_query.format(telescope=self.mission)

        # Use GPT reranker if available, otherwise fall back to Cohere
        if self.gpt_reranker:
            reranked_data = self.gpt_reranker.rerank_snippets(
                rerank_query, all_snippets, self.top_k_snippets
            )
        else:
            reranked_data = self.cohere_client.rerank_snippets(
                rerank_query, all_snippets, self.top_k_snippets
            )

        if not reranked_data:
            logger.warning(f"Reranking produced no snippets for {paper_id}. Skipping LLM analysis.")
            return {"science": 0.0, "quotes": [], 
                   "reason": "Keyword snippets found but none survived reranking/filtering."}
                   
        # Check reranker threshold for top score
        top_score = reranked_data[0].get('score')
        if top_score is not None and top_score < self.reranker_threshold:
            logger.info(f"Skipping LLM science analysis for {paper_id}: Top reranker score ({top_score:g}) below threshold ({self.reranker_threshold}).")
            return {
                "science": 0.0, 
                "quotes": [],
                "reason": f"Skipped LLM analysis: Top reranker score ({top_score:g}) was below the threshold ({self.reranker_threshold}).",
            }

        # Filter snippets above threshold and respect top_k limit
        filtered_snippets = []
        for item in reranked_data:
            score = item.get('score')
            if score is not None and score >= self.reranker_threshold:
                filtered_snippets.append(item)
            if len(filtered_snippets) >= self.top_k_snippets:
                break
        
        if not filtered_snippets:
            logger.info(f"No snippets above threshold ({self.reranker_threshold}) for {paper_id}.")
            return {
                "science": 0.0,
                "quotes": [],
                "reason": f"No snippets scored above the threshold ({self.reranker_threshold}).",
            }

        logger.info(f"Using {len(filtered_snippets)} snippets above threshold for {paper_id} (filtered from {len(reranked_data)})")

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
            logger.warning(f"Total snippet text for {paper_id} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.prompts.get('science_system')
        user_prompt_template = self.prompts.get('science_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"Science prompts not loaded correctly for {paper_id}. Check prompts directory.")
            return {"science": -1.0, "reason": "Analysis failed: Prompts missing", 
                   "quotes": [], "error": "prompt_missing"}
        
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text, telescope=self.mission)
        except KeyError as e:
            logger.error(f"Failed to format science user prompt - missing placeholder {e}")
            return {"science": -1.0, "reason": "Analysis failed: Prompt formatting error", 
                   "quotes": [], "error": "prompt_format_error"}

        # Call LLM with telescope classification
        llm_result = self.openai_client.call_telescope_classification(
            system_prompt, user_prompt, self.mission
        )

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            # Return in legacy format for backward compatibility
            return {"science": -1.0, "reason": error_reason, "quotes": [], "error": error_type}

        # Convert to legacy format for backward compatibility
        legacy_result = {
            "science": 1.0 if llm_result.get("science", False) else 0.0,
            "reason": llm_result.get("reason", ""),
            "quotes": llm_result.get("quotes", [])
        }
        
        # Add new telescope classification fields
        legacy_result.update({
            "telescope": llm_result.get("telescope", self.mission),
            "instrumentation": llm_result.get("instrumentation", False),
            "mention": llm_result.get("mention", False), 
            "not_telescope": llm_result.get("not_telescope", False)
        })
        
        return legacy_result