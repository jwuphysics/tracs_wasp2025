"""OpenAI API client wrapper."""

import json
import logging
from typing import Optional, Dict, Any, Type, List

from openai import OpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""
    
    def __init__(self, api_key: str, model: str = 'gpt-5-mini'):
        self.client = OpenAI(api_key=api_key, max_retries=2)
        self.model = model
        
    def call_parse(self, system_prompt: str, user_prompt: str, 
                   response_model: Type[BaseModel]) -> Optional[Dict[str, Any]]:
        """Helper function to call OpenAI parse API with error handling."""
        try:
            if not system_prompt or not user_prompt:
                logger.error("System or User prompt is empty. Cannot call OpenAI.")
                return {"error": "empty_prompt", "message": "System or User prompt was empty."}

            # Only apply reasoning_effort for gpt-o1-mini and gpt-o1-preview models
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": response_model,
                "timeout": 60
            }
            
            # Add reasoning_effort only for gpt-5-mini
            if self.model == "gpt-5-mini":
                kwargs["reasoning_effort"] = "minimal"

            result = self.client.beta.chat.completions.parse(**kwargs)

            parsed_object = result.choices[0].message.parsed
            if parsed_object:
                if isinstance(parsed_object, response_model):
                    return parsed_object.model_dump()
                else:
                    logger.error(f"OpenAI response parsed, but yielded unexpected type: {type(parsed_object)}. Expected {response_model}.")
                    logger.debug(f"Raw parsed object: {parsed_object}")
                    return {"error": "parse_type_mismatch", "message": f"Parsed object type mismatch: got {type(parsed_object)}"}
            else:
                logger.error("OpenAI response parsed, but no Pydantic object found in expected location.")
                logger.debug(f"Full OpenAI response object: {result}")
                try:
                    raw_content = result.choices[0].message.content
                    if isinstance(raw_content, str):
                        parsed_json = json.loads(raw_content)
                        validated_model = response_model.model_validate(parsed_json)
                        logger.warning("Used manual JSON parsing/validation fallback.")
                        return validated_model.model_dump()
                    else:
                        logger.error("Raw message content is not a string for manual parsing.")
                        return None
                except (json.JSONDecodeError, ValidationError, Exception) as manual_parse_err:
                    logger.error(f"Manual parsing/validation of OpenAI response failed: {manual_parse_err}")
                    return None

        except BadRequestError as e:
            logger.warning(f"OpenAI API BadRequestError: {e}")
            if "context length" in str(e).lower():
                logger.warning(f"Snippets might still exceed token limit: {e}")
                return {"error": "token_limit", "message": str(e)}
            else:
                return {"error": "bad_request", "message": str(e)}

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {"error": "api_error", "message": str(e)}

        return None
    
    def call_separated_analysis(self, system_prompt: str, user_prompt: str, mission: str) -> Optional[Dict[str, Any]]:
        """Legacy method for backward compatibility - calls telescope classification."""
        return self.call_telescope_classification(system_prompt, user_prompt, mission)
        
    def call_telescope_classification(self, system_prompt: str, user_prompt: str, telescope: str) -> Optional[Dict[str, Any]]:
        """Telescope classification: reasoning first, then scoring based on reasoning.""" 
        from ..models import TelescopeClassificationReasoningModel, TelescopeClassificationScoringModel
        
        try:
            # Step 1: Get reasoning and quotes only
            kwargs_reasoning = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": TelescopeClassificationReasoningModel,
                "timeout": 60
            }
            
            # Add reasoning_effort only for gpt-5-mini
            if self.model == "gpt-5-mini":
                kwargs_reasoning["reasoning_effort"] = "minimal"
                
            reasoning_result = self.client.beta.chat.completions.parse(**kwargs_reasoning)
            
            reasoning_data = reasoning_result.choices[0].message.parsed
            if not reasoning_data:
                logger.error("Failed to get reasoning from first step")
                return None
                
            reasoning_dict = reasoning_data.model_dump()
            
            # Step 2: Score based on the reasoning
            if telescope == "NONE":
                scoring_prompt = f"""Based on this analysis of a paper where no supported space telescopes (CHANDRA, HST, JWST) were detected:

REASONING: {reasoning_dict['reason']}

Based on this reasoning, determine the appropriate classification for this NONE paper:
- science: False (no supported telescope data used)
- instrumentation: False (no supported telescope technical aspects)  
- mention: True if paper discusses astronomy/astrophysics topics but uses ground-based observations, theory, or other telescopes
- not_telescope: True if the reasoning indicates this is about a completely different field, false references, or grant-only mentions

For NONE papers: Either mention=True (legitimate astronomy without supported telescopes) OR not_telescope=True (false positive/unrelated), but not both."""
            else:
                scoring_prompt = f"""Based on this analysis of {telescope} telescope classification evidence:

REASONING: {reasoning_dict['reason']}

Based on this reasoning, determine the boolean classifications for this paper:
- science: Does it use {telescope} data for NEW scientific results?
- instrumentation: Does it describe {telescope} technical aspects/instruments?  
- mention: Does it reference {telescope} without new results/contributions?
- not_telescope: Are the references actually about something else?

Remember: A paper can have multiple True values, but if science or instrumentation is True, mention should be False."""
            
            kwargs_scoring = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": scoring_prompt}
                ],
                "response_format": TelescopeClassificationScoringModel,
                "timeout": 60
            }
            
            # Add reasoning_effort only for gpt-5-mini
            if self.model == "gpt-5-mini":
                kwargs_scoring["reasoning_effort"] = "minimal"
                
            scoring_result = self.client.beta.chat.completions.parse(**kwargs_scoring)
            
            scoring_data = scoring_result.choices[0].message.parsed
            if not scoring_data:
                logger.error("Failed to get classification scores from second step")
                return None
                
            # Combine results
            result = {
                "telescope": telescope,
                "quotes": reasoning_dict["quotes"],
                "reason": reasoning_dict["reason"],
                **scoring_data.model_dump()
            }
            
            # Add legacy science score for backward compatibility
            result["science_score"] = 1.0 if result["science"] else 0.0
            
            # Validation: Check for consistency between reasoning and classifications
            validation_warnings = self._validate_classification_consistency(result, telescope)
            if validation_warnings:
                logger.warning(f"Classification consistency warnings for {telescope}: {'; '.join(validation_warnings)}")
                # Add validation info to the result for debugging
                result["validation_warnings"] = validation_warnings
            
            return result
                
        except Exception as e:
            logger.error(f"Telescope classification failed: {e}")
            return None
            
    def identify_telescope(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Identify which telescope a paper primarily discusses."""
        from ..models import TelescopeIdentificationModel
        
        try:
            kwargs_identification = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": TelescopeIdentificationModel,
                "timeout": 60
            }
            
            # Add reasoning_effort only for gpt-5-mini
            if self.model == "gpt-5-mini":
                kwargs_identification["reasoning_effort"] = "minimal"
                
            result = self.client.beta.chat.completions.parse(**kwargs_identification)
            
            parsed_data = result.choices[0].message.parsed
            if not parsed_data:
                logger.error("Failed to get telescope identification")
                return None
                
            return parsed_data.model_dump()
                
        except Exception as e:
            logger.error(f"Telescope identification failed: {e}")
            return None
    
    def _validate_classification_consistency(self, result: Dict, telescope: str) -> List[str]:
        """
        Validate consistency between reasoning text and boolean classifications.
        Returns list of warnings if inconsistencies detected.
        """
        warnings = []
        reason = result.get("reason", "").lower()
        quotes = [q.lower() for q in result.get("quotes", [])]
        all_text = reason + " " + " ".join(quotes)
        
        science = result.get("science", False)
        instrumentation = result.get("instrumentation", False)
        mention = result.get("mention", False)
        not_telescope = result.get("not_telescope", False)
        
        # Check for telescope name mentions in reasoning vs not_telescope classification
        if telescope != "NONE":
            telescope_lower = telescope.lower()
            if not_telescope and any(telescope_lower in text for text in [reason] + quotes):
                if any(phrase in all_text for phrase in ["observation", "data", "analysis", "image", "spectrum", "photometry"]):
                    warnings.append(f"Classified as not_telescope but reasoning discusses {telescope} observations/data")
        
        # Check for science indicators in reasoning vs science classification
        science_indicators = ["our ", "we observed", "we analyzed", "we found", "new results", "our data", "our observations"]
        if not science and any(indicator in all_text for indicator in science_indicators):
            warnings.append("Science indicators in reasoning but science=False")
        
        # Check for instrumentation indicators vs instrumentation classification  
        instrument_indicators = ["calibration", "pipeline", "instrument", "detector", "technical", "hardware", "software"]
        if not instrumentation and any(indicator in all_text for indicator in instrument_indicators):
            warnings.append("Instrumentation indicators in reasoning but instrumentation=False")
        
        # Check logic rules: if science or instrumentation is True, mention should be False
        if (science or instrumentation) and mention:
            warnings.append("Logic violation: science/instrumentation=True but mention=True (should be False)")
            
        # For NONE papers, check if reasoning mentions actual telescopes
        if telescope == "NONE" and not_telescope:
            real_telescopes = ["chandra", "hubble", "hst", "jwst", "webb", "spitzer", "kepler"]
            mentioned_telescopes = [tel for tel in real_telescopes if tel in all_text]
            if mentioned_telescopes:
                warnings.append(f"NONE paper classified not_telescope but reasoning mentions: {', '.join(mentioned_telescopes)}")
        
        return warnings
