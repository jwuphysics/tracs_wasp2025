"""Telescope detection module for identifying telescopes mentioned in papers."""

import logging
from typing import List, Dict, Set
from ..clients.openai import OpenAIClient
from ..models import TelescopeDetectionModel

logger = logging.getLogger(__name__)


class TelescopeDetector:
    """Detects which telescopes are mentioned in a paper."""


    # Meant to be HIGH RECALL but LOW PRECISION -- lots of false positives expected
    TELESCOPE_KEYWORDS = {
        "CHANDRA": [
            "chandra", "cxc", "cxo", "axaf",
            "acis", "hrc", "hetg", "letg", "hrma",
            "pcad", "ephin",
            "ciao", "csc"
        ],
        "HST": [
            "hst", "hubble",
            "wfc3", "acs", "stis", "cos", "nicmos",           # note: "cos" will match on "cosmic" or "cosine"
            "wfpc2", "foc", "fos", "ghrs", "hsp", "wfpc",     # similarly "foc" will match on "focus" or
            "costar", "fgs"
        ],
        "JWST": [
            "jwst", "webb", "ngst",
            "nircam", "nirspec", "miri", "niriss", "fgs"
        ]
    }
    
    def __init__(self, openai_client: OpenAIClient, prompts: Dict[str, str]):
        self.openai_client = openai_client
        self.prompts = prompts
        
    def detect_telescopes(self, paper_text: str, bibcode: str = "unknown") -> Dict:
        """
        Detect which telescopes are mentioned in the paper.
        
        Returns:
            Dict with detected telescopes and reasoning
        """
        if not paper_text or not paper_text.strip():
            logger.warning(f"No text provided for telescope detection in {bibcode}")
            return {
                "detected_telescopes": [],
                "primary_telescope": "NONE",
                "reasoning": "No text content available for analysis",
                "confidence": "low"
            }
        
        # Step 1: Keyword-based pre-screening
        keyword_telescopes = self._keyword_screening(paper_text)
        
        # Step 2: LLM-based verification and refinement
        llm_result = self._llm_detection(paper_text, keyword_telescopes)
        
        if llm_result:
            detected = llm_result.get("telescopes", [])
            reasoning = llm_result.get("reasoning", "LLM-based detection")
            
            # Determine primary telescope
            primary = self._determine_primary_telescope(detected, paper_text)
            
            return {
                "detected_telescopes": detected,
                "primary_telescope": primary,
                "reasoning": reasoning,
                "confidence": "high" if detected else "medium"
            }
        else:
            # Fallback to keyword-only results
            primary = self._determine_primary_telescope(keyword_telescopes, paper_text)
            
            return {
                "detected_telescopes": keyword_telescopes,
                "primary_telescope": primary,
                "reasoning": f"Keyword-based detection found: {', '.join(keyword_telescopes) if keyword_telescopes else 'no telescopes'}",
                "confidence": "medium" if keyword_telescopes else "low"
            }
    
    def _keyword_screening(self, text: str) -> List[str]:
        """Screen text for telescope keywords."""
        text_lower = text.lower()
        detected = []
        
        for telescope, keywords in self.TELESCOPE_KEYWORDS.items():
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                detected.append(telescope)
                logger.debug(f"Found {telescope} keywords: {found_keywords}...")

        return detected
    
    def _llm_detection(self, paper_text: str, keyword_telescopes: List[str]) -> Dict:
        """Use LLM to verify and refine telescope detection."""
        system_prompt = self.prompts.get('telescope_detection_system')
        user_prompt_template = self.prompts.get('telescope_detection_user')
        
        if not system_prompt or not user_prompt_template:
            logger.warning("Telescope detection prompts not found, using keyword-only detection")
            return None
        
        # Limit text length for LLM processing
        max_chars = 8000
        if len(paper_text) > max_chars:
            # Take first part + last part to capture both intro and conclusions
            mid_point = max_chars // 2
            text_for_llm = paper_text[:mid_point] + "\n\n[...text truncated...]\n\n" + paper_text[-mid_point:]
        else:
            text_for_llm = paper_text
        
        # Include keyword detection context
        keyword_context = f"Initial keyword screening detected: {', '.join(keyword_telescopes) if keyword_telescopes else 'no telescopes'}"
        
        try:
            user_prompt = user_prompt_template.format(
                paper_text=text_for_llm,
                keyword_context=keyword_context
            )
            
            result = self.openai_client.call_parse(
                system_prompt, user_prompt, TelescopeDetectionModel
            )
            
            if result and "error" not in result:
                return result
            else:
                logger.warning(f"LLM telescope detection failed: {result.get('message', 'Unknown error') if result else 'No response'}")
                return None
                
        except Exception as e:
            logger.error(f"LLM telescope detection error: {e}")
            return None
    
    def _determine_primary_telescope(self, detected_telescopes: List[str], paper_text: str) -> str:
        """Determine the primary telescope from detected telescopes."""
        if not detected_telescopes:
            return "NONE"
        
        if len(detected_telescopes) == 1:
            return detected_telescopes[0]
        
        # For multiple telescopes, use simple heuristic based on keyword frequency
        telescope_scores = {}
        text_lower = paper_text.lower()
        
        for telescope in detected_telescopes:
            score = 0
            keywords = self.TELESCOPE_KEYWORDS.get(telescope, [])
            
            for keyword in keywords:
                # Count occurrences, with higher weight for more specific keywords
                count = text_lower.count(keyword.lower())
                weight = len(keyword.split())  # Multi-word keywords get higher weight
                score += count * weight
            
            telescope_scores[telescope] = score
        
        # Return telescope with highest score
        primary = max(telescope_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Multiple telescopes detected {detected_telescopes}, selected primary: {primary}")
        
        return primary