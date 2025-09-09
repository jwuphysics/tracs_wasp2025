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

The system provides a numbered pipeline for processing CSV data with combined Id fields (bibcode_telescope format):

### Mini Pipeline

```bash
# Step 1: Process and explore data
python scripts/1-process_data.py data/train.csv --create-subset 100

# Step 2: Run telescope classification  
python scripts/2-classify_papers.py data/train_subset.csv

# Step 3: Evaluate results (if ground truth available)
python scripts/3-evaluate_results.py results/submission.csv ground_truth.json

# Step 4: Generate detailed analysis
python scripts/4-analyze_results.py results/submission.csv --detailed
```

### Full Kaggle run

For full dataset processing:

```bash
python scripts/1-process_data.py data/test.csv
python scripts/2-classify_papers.py data/test.csv --output-dir results/production
python scripts/4-analyze_results.py results/production/results/submission.csv
```

### Data Format

The inputs are expected to be CSV files with:
- ID format: `bibcode_telescope` (e.g., `2012A&A...537A..18M_CHANDRA`)
- Text fields: title, abstract, body, acknowledgments, grants

The output Competition CSV with boolean classification columns

### Configuration Options

Common parameters across all scripts:
- `--gpt-model`: GPT model for classification (default: gpt-5-mini)
- `--reranker-model`: GPT model for snippet reranking (default: gpt-4.1-nano)  
- `--limit-rows`: Limit number of rows processed
- `--output-dir`: Directory for output files
- `--verbose`: Enable detailed logging

## Installation

Requires Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/jwuphysics/tracs_wasp2025
cd tracs_wasp2025
uv venv && source .venv/bin/activate
uv sync
export OPENAI_API_KEY=your_openai_key_here
```

## System Overview

The system classifies astronomical papers by telescope relevance using a multi-stage LLM pipeline:

1. **Text Processing**: Combines title, abstract, body, acknowledgments, and grants
2. **Telescope Detection**: Identifies relevant telescopes using keywords and LLM analysis  
3. **Content Reranking**: Extracts and ranks most relevant text passages
4. **Classification**: Applies LLM prompts to classify telescope usage type

## Input and Output Formats

### Input CSV Format
The system expects CSV files with combined Id fields:

```csv
Id,bibcode,author,year,title,abstract,body,acknowledgments,grants
2012A&A...537A..18M_CHANDRA,2012A&A...537A..18M,"Author, A.",2012,Paper Title,Abstract text,Body text,Ack text,Grant info
```

### Competition Output Format
Boolean classifications for submission:

```csv
Id,telescope,science,instrumentation,mention,not_telescope
2012A&A...537A..18M_CHANDRA,CHANDRA,False,False,True,False
1998SPIE.3356.1078P_CHANDRA,CHANDRA,False,True,False,False
2022ApJ...935..177S_HST,HST,True,False,False,False
```

### Detailed Analysis Output
The system also generates comprehensive JSON reports with:
- Classification reasoning and supporting quotes
- Processing statistics and error handling
- Telescope detection confidence scores

## Direct Classifier Configuration

For running the automated_mission_classifier directly:

```bash
python -m automated_mission_classifier \
    --csv-file data/test.csv \             # CSV input file
    --output-dir results \                 # Output directory
    --limit-rows 100 \                     # Limit processing (optional)
    --gpt-model gpt-5-mini \              # Classification model
    --reranker-model gpt-4.1-nano \       # Reranking model
    --top-k-snippets 5 \                  # Max snippets to LLM
    --reranker-threshold 0.001 \          # Min reranker confidence
    --verbose                              # Detailed logging
```

Alternative JSON input mode:
```bash
python -m automated_mission_classifier \
    --data-file papers.json \             # JSON input file
    --output-dir results
```

## Supported Telescopes

- **CHANDRA**: X-ray Observatory (including ACIS, HRC, HETG, LETG instruments)
- **HST**: Hubble Space Telescope (including WFC3, ACS, STIS, COS, NICMOS, WFPC2)
- **JWST**: James Webb Space Telescope (including NIRCam, NIRSpec, MIRI, NIRISS, FGS)
- **NONE**: Papers not primarily about any specific telescope

## Citation and Acknowledgments

This work was developed for the WASP2025 shared task on telescope bibliography classification. The TRACS dataset is available on Hugging Face at `adsabs/TRACS`. Credit to Felix Grezes. TRACS @ WASP 2025. https://kaggle.com/competitions/tracs-wasp-2025, 2025. Kaggle.