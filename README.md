# TRACS Telescope Classification System

**Automated telescope bibliography curation using Large Language Models**

This system classifies astronomical papers by telescope relevance and usage type for bibliography curation. Originally developed for the WASP2025 shared task using the TRACS (Telescope Bibliography Classification) dataset, it identifies how papers relate to major space telescopes (CHANDRA, HST, JWST) and ground truth classification.

## What This System Does

The system analyzes astronomical papers and determines:
- **Which telescope** the paper primarily discusses (CHANDRA, HST, JWST, or none)
- **How the paper uses the telescope** across four categories:
  - **Science**: Uses telescope data to obtain new scientific results
  - **Instrumentation**: Describes technical aspects, calibration, or instruments
  - **Mention**: References telescope without new results or contributions  
  - **Not Telescope**: Contains references confused with telescope but about something else

This classification supports librarians and data curators in building accurate telescope bibliographies for impact assessment and data discovery.

## Quick Start

**Process TRACS competition dataset:**
```bash
# Download TRACS test data and process for competition submission
python tracs_data_loader.py
python process_all_tracs_test.py
```

**Analyze individual papers:**
```bash
# Classify a specific paper
python -m automated_mission_classifier --mission MULTI --bibcode 2012A&A...537A..18M --data-file data/tracs_test_combined.json
```

## Installation

**Prerequisites**: Python 3.10+, OpenAI API key

```bash
# Clone and setup
git clone https://github.com/jwuphysics/tracs_wasp2025
cd tracs_wasp2025
uv venv && source .venv/bin/activate
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY=your_openai_key_here
```


## How It Works

The system processes astronomical papers through several stages:

1. **Text Analysis**: Extracts relevant content from paper title, abstract, body text, acknowledgments, and grants
2. **Telescope Identification**: Uses telescope-specific keywords and LLM analysis to identify which telescope (if any) the paper discusses
3. **Content Reranking**: Uses GPT-4.1-nano to identify and rank the most relevant text passages for classification
4. **Multi-Label Classification**: Applies specialized LLM prompts to classify papers into the four telescope usage categories
5. **Output Generation**: Produces structured results in both detailed JSON and competition CSV formats

## Results and Output

**Competition CSV Format** (`*_competition.csv`):
```csv
Id,telescope,science,instrumentation,mention,not_telescope
2012A&A...537A..18M,CHANDRA,False,False,True,False
1998SPIE.3356.1078P,CHANDRA,False,True,False,False
```

**Detailed Analysis** (`*_report.json`):
- Paper-by-paper classification reasoning and supporting quotes
- Processing statistics and performance metrics
- Model configuration and parameters used

## Supported Telescopes

- **CHANDRA**: X-ray Observatory (including ACIS, HRC, HETG, LETG instruments)
- **HST**: Hubble Space Telescope (including WFC3, ACS, STIS, COS, NICMOS, WFPC2)
- **JWST**: James Webb Space Telescope (including NIRCam, NIRSpec, MIRI, NIRISS, FGS)
- **NONE**: Papers not primarily about any specific telescope

## Citation and Acknowledgments

This work was developed for the WASP2025 shared task on telescope bibliography classification. The TRACS dataset is available on Hugging Face at `adsabs/TRACS`. Credit to Felix Grezes. TRACS @ WASP 2025. https://kaggle.com/competitions/tracs-wasp-2025, 2025. Kaggle.

For technical details about the system architecture and implementation, see `CLAUDE.md`.

