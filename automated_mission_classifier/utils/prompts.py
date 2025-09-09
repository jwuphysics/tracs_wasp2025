"""Prompt management utilities."""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


PROMPT_FILES = {
    'science_system': 'science_system.txt',
    'science_user': 'science_user.txt', 
    'rerank_science_query': 'rerank_science_query.txt',
    'telescope_identification_system': 'telescope_identification_system.txt',
    'telescope_identification_user': 'telescope_identification_user.txt',
    'telescope_detection_system': 'telescope_detection_system.txt',
    'telescope_detection_user': 'telescope_detection_user.txt',
    'telescope_classification_system': 'telescope_classification_system.txt',
    'telescope_classification_user': 'telescope_classification_user.txt',
    'rerank_classification_query': 'rerank_classification_query.txt',
    'primary_telescope_identification_system': 'primary_telescope_identification_system.txt',
    'primary_telescope_identification_user': 'primary_telescope_identification_user.txt',
    'primary_telescope_rerank_query': 'primary_telescope_rerank_query.txt',
}


def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    """Loads system and user prompt templates from files."""
    loaded_prompts = {}
    logger.info(f"Loading prompts from directory: {prompts_dir}")
    
    if not prompts_dir.is_dir():
        logger.warning(f"Prompts directory not found: {prompts_dir}. Creating it.")
        try:
            prompts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create prompts directory {prompts_dir}: {e}")
            raise 

    for key, filename in PROMPT_FILES.items():
        filepath = prompts_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_prompts[key] = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {filepath}. Please create it.")
            raise FileNotFoundError(f"Essential prompt file missing: {filepath}") 
        except Exception as e:
            logger.error(f"Failed to load prompt file {filepath}: {e}")
            raise 

    return loaded_prompts