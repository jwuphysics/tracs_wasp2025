# Telescope Classification Pipeline

This directory contains a step-by-step pipeline for processing and classifying telescope-related papers. Each script builds on the previous one to create a complete workflow from raw CSV data to final analysis.

## Pipeline Overview

```
CSV Data → Process → Classify → Evaluate → Analyze
    ↓         ↓         ↓         ↓         ↓
  Step 1   Step 2    Step 3    Step 4    Reports
```

## Scripts

### `1-process_data.py` - Data Processing and Analysis
**Purpose**: Initial data exploration, preprocessing, and subset creation

**Key Features**:
- Parse combined Id fields (`bibcode_telescope` format)
- Generate data statistics and distribution analysis
- Create balanced test subsets for development
- Convert CSV to JSON format if needed

**Usage**:
```bash
# Basic analysis
python scripts/1-process_data.py data/test.csv

# Create test subset
python scripts/1-process_data.py data/test.csv --create-subset 100

# Convert to JSON
python scripts/1-process_data.py data/test.csv --convert-to-json
```

**Outputs**:
- Data distribution statistics
- Test subset CSV files
- JSON format conversion (optional)

---

### `2-classify_papers.py` - Telescope Classification
**Purpose**: Run the core classification algorithm on CSV data

**Key Features**:
- Process papers with single-telescope-per-row format
- Use LLM-based classification (science/instrumentation/mention/not_telescope)
- Generate competition-ready CSV output
- Provide detailed progress tracking and performance metrics

**Usage**:
```bash
# Process test subset
python scripts/2-classify_papers.py data/test_subset.csv

# Process with limits
python scripts/2-classify_papers.py data/test.csv --limit-rows 1000

# Custom output location
python scripts/2-classify_papers.py data/test.csv --output-dir results/full_run
```

**Outputs**:
- `submission.csv` - Competition format results
- Classification cache files
- Performance reports

---

### `3-evaluate_results.py` - Results Evaluation
**Purpose**: Compare predictions against ground truth and calculate performance metrics

**Key Features**:
- Calculate accuracy, precision, recall, F1 scores
- Generate confusion matrices per telescope/classification
- Find and analyze misclassification examples
- Work with both JSON and CSV ground truth formats

**Usage**:
```bash
# With ground truth
python scripts/3-evaluate_results.py results/submission.csv ground_truth.json

# Analysis only (no ground truth)
python scripts/3-evaluate_results.py results/submission.csv --analyze-only

# Using CSV ground truth
python scripts/3-evaluate_results.py results/submission.csv train.csv
```

**Outputs**:
- Performance metrics by telescope and classification type
- Misclassification examples
- Detailed evaluation JSON report

---

### `4-analyze_results.py` - Results Analysis and Visualization
**Purpose**: Generate comprehensive analysis and insights from classification results

**Key Features**:
- Distribution analysis across telescopes and classifications
- Multi-label classification pattern analysis
- Paper-level analysis (single vs multi-telescope papers)
- Cross-tabulation analysis
- Export in multiple formats (JSON, CSV, TXT)

**Usage**:
```bash
# Basic analysis
python scripts/4-analyze_results.py results/submission.csv

# Detailed analysis with cross-tabs
python scripts/4-analyze_results.py results/submission.csv --detailed

# Multiple export formats
python scripts/4-analyze_results.py results/submission.csv --export-formats json csv txt
```

**Outputs**:
- Comprehensive analysis report
- Statistical summaries
- Pattern identification
- Export files in requested formats

## Complete Workflow Example

Here's how to run the complete pipeline:

```bash
# Step 1: Explore the data and create test subset
python scripts/1-process_data.py data/test.csv --create-subset 100

# Step 2: Run classification on test subset
python scripts/2-classify_papers.py data/test_subset.csv --output-dir results/test

# Step 3: Evaluate results (if ground truth available)
python scripts/3-evaluate_results.py results/test/results/submission.csv ground_truth.json

# Step 4: Generate detailed analysis
python scripts/4-analyze_results.py results/test/results/submission.csv --detailed

# For production: run on full dataset
python scripts/2-classify_papers.py data/test.csv --output-dir results/full
python scripts/4-analyze_results.py results/full/results/submission.csv --detailed
```

## Data Flow

```
Input CSV (test.csv)
├── Combined Id field: "2012A&A...537A..18M_CHANDRA"
├── Text fields: title, abstract, body, acknowledgments, grants
└── Each row = one paper-telescope pair to classify

↓ Step 1: Processing
├── Parse Id → bibcode + telescope  
├── Handle missing values (NaN → empty strings)
└── Generate statistics and subsets

↓ Step 2: Classification  
├── Extract relevant text snippets
├── Rerank snippets using GPT-4.1-nano
├── Classify using GPT-5-mini with structured output
└── Cache results for efficiency

↓ Step 3: Evaluation (optional)
├── Compare predictions vs ground truth
├── Calculate performance metrics
└── Identify misclassifications

↓ Step 4: Analysis
├── Distribution analysis
├── Pattern identification  
├── Multi-label analysis
└── Export reports
```

## Configuration Options

All scripts support common configuration options:

- **Models**: `--gpt-model`, `--reranker-model`
- **Filtering**: `--limit-rows`, `--reranker-threshold`
- **Output**: `--output-dir`, `--verbose`
- **Processing**: `--top-k-snippets`, `--context-sentences`

## Output Structure

```
output/
├── results/
│   ├── submission.csv              # Competition format
│   ├── csv_classification_cache.json
│   └── csv_classification_report.json
├── evaluation/
│   └── evaluation_results.json    # Performance metrics
└── analysis/
    ├── submission_analysis.json   # Detailed analysis
    ├── submission_analysis.txt    # Human-readable report
    └── submission_analysis.csv    # Key statistics
```

## Use Cases

### Development and Testing
```bash
# Quick test with small subset
python scripts/1-process_data.py data/test.csv --create-subset 10
python scripts/2-classify_papers.py data/test_subset.csv --limit-rows 5
```

### Production Classification
```bash
# Full dataset processing
python scripts/2-classify_papers.py data/test.csv --output-dir results/production
```

### Performance Analysis
```bash
# With ground truth evaluation
python scripts/3-evaluate_results.py results/submission.csv train.csv
python scripts/4-analyze_results.py results/submission.csv --detailed
```

## Tips

1. **Start Small**: Always test with `--create-subset` before running full datasets
2. **Use Caching**: Results are cached automatically - rerun scripts to resume from failures  
3. **Monitor Progress**: Use `--verbose` for detailed logging
4. **Check Outputs**: Each step suggests the next step in its output
5. **Ground Truth**: Step 3 works without ground truth for basic analysis

## Troubleshooting

- **API Errors**: Check OpenAI API key and rate limits
- **Memory Issues**: Use `--limit-rows` for large datasets
- **Missing Files**: Scripts validate input files and provide clear error messages
- **NaN Values**: Handled automatically in CSV processing