# Solving the TRACS @ WASP 2025 Shared Task

This repo gives a solution using LLMs for automated telescope bibliography classification. See [the Kaggle competition](https://www.kaggle.com/competitions/tracs-wasp-2025) for more details.

The system ingests a big CSV file of bibliographic data (including paper full text) and determiens:
- Which telescope the paper primarily discusses (CHANDRA, HST, JWST, or none)
- How the paper uses the telescope across four categories:
  - **Science**: Uses telescope data to obtain new scientific results
  - **Instrumentation**: Describes technical aspects, calibration, or instruments
  - **Mention**: References telescope without new results or contributions  
  - **Not Telescope**: Might contain substrings of a telescope keyword but it's about something else

## Quick Start

Run the full pipeline for testing outputs, e.g., evaluate on a "training" dataset

```bash
# Step 1a: Process and explore data
python scripts/1-process_data.py data/train.csv --create-subset 100

# Step 1b: Make ground truth dataset
python scripts/1-process_data.py data/train.csv --create-ground-truth

# Step 2: Run telescope classification  
python scripts/2-classify_papers.py data/train_subset.csv

# Step 3: Evaluate results (if ground truth available; *you should skip for the test sets)
python scripts/3-evaluate_results.py output/results/submission.csv data/ground_truth.json

# Step 4: Summarize results (some of this is Claude slop, sorry)
python scripts/4-analyze_results.py output/results/submission.csv --detailed
```

**If you want to run on the full Kaggle dataset, just simply run:**


```bash
python scripts/1-process_data.py data/test.csv
python scripts/2-classify_papers.py data/test.csv --output-dir results/production
python scripts/4-analyze_results.py results/production/results/submission.csv
```

This takes < 24 hrs to process ~9000 papers (see details about the dataset on [Kaggle](https://www.kaggle.com/competitions/tracs-wasp-2025/data) and [Huggingface](https://ui.adsabs.harvard.edu/WIESP/2025/shared_task#dataset-description)). With `gpt-5-mini` and the custom reranker (both enabled by default), this costs about $12. 

## Details about the codebase

### Data Format

The inputs are expected to be CSV files with:
- ID format: `bibcode_telescope` (e.g., `2012A&A...537A..18M_CHANDRA`)
- Text fields: title, abstract, body, acknowledgments, grants

The output Competition CSV with boolean classification columns

### LLM system Options

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

- **Text Processing**: Combines title, abstract, body, acknowledgments, and grants
- **Content Reranking**: Extracts and ranks most relevant text passages
- **Classification**: Applies LLM prompts to classify telescope usage type

### Data

#### Input CSV Format
The system expects CSV files with combined Id fields:

```csv
Id,bibcode,author,year,title,abstract,body,acknowledgments,grants
2012A&A...537A..18M_CHANDRA,2012A&A...537A..18M,"Author, A.",2012,Paper Title,Abstract text,Body text,Ack text,Grant info
```

#### Kaggle submission format
Boolean classifications for submission:

```csv
Id,telescope,science,instrumentation,mention,not_telescope
2012A&A...537A..18M_CHANDRA,CHANDRA,False,False,True,False
1998SPIE.3356.1078P_CHANDRA,CHANDRA,False,True,False,False
2022ApJ...935..177S_HST,HST,True,False,False,False
```

### More detailed analyses
The system also generates some JSON reports, which let you *look at your data* (thanks Hamel). E.g. you can check out
- the LLM provided reasoning and supporting quotes (beware hallucinations)
- processing statistics and error rates
- telescope detection confidence scores

### Run the `amc` (automated mission classifier) directly

This repo is adapted from the [`automated-mission-classifier`](https://github.com/jwuphysics/automated-mission-classifier) package, or `amc` for short. You can also run the `amc` directly like so

```bash
python -m automated_mission_classifier \
    --csv-file data/test.csv \             
    --output-dir results \                
    --limit-rows 100 \                   
    --gpt-model gpt-5-mini \            
    --reranker-model gpt-4.1-nano \    
    --top-k-snippets 5 \              
    --reranker-threshold 0.001 \     
    --verbose                       
```

And if your inputs are JSON rather than a CSV, then try:
```bash
python -m automated_mission_classifier \
    --data-file papers.json \            
    --output-dir results
```

## Telescopes/missions currently processed

- **CHANDRA**: X-ray Observatory (including ACIS, HRC, HETG, LETG instruments)
- **HST**: Hubble Space Telescope (including WFC3, ACS, STIS, COS, NICMOS, WFPC2)
- **JWST**: James Webb Space Telescope (including NIRCam, NIRSpec, MIRI, NIRISS, FGS)
- **NONE**: Papers not primarily about any specific telescope


I'm actually not sure what to do about the `*_NONE` paper IDs, so for now I'm just automatically labelling them all categories as "False" (following the Huggingface training dataset).

By the way, it's very easy to add additional telescopes or keywords to `automated_mission_classifier/detection/telescope_detector.py` (add entries to the `TELESCOPE_KEYWORDS` dict).

## Citation and Acknowledgments

This work was developed by John Wu, the Applied AI Scientist in the STScI Data Science Mission Office, for the WASP2025 shared task on telescope bibliography classification. The TRACS dataset is available on Hugging Face at `adsabs/TRACS`. Credit to Felix Grezes. TRACS @ WASP 2025. https://kaggle.com/competitions/tracs-wasp-2025, 2025. Kaggle.
