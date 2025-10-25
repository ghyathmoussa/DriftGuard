# DriftGuard Demo Guide

This guide shows you how to run various demos of DriftGuard's capabilities.

## Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

## Available Demos

### 1. Simple LLM Drift Detection 🎯

**Best for:** Quick understanding of drift detection

```bash
python demo_simple.py
```

**What it demonstrates:**
- Basic LLM drift detector setup
- Comparing similar vs different text distributions
- Understanding drift scores and thresholds

**Expected output:**
- No drift when texts are semantically similar
- Drift detected when topics change significantly

---

### 2. Comprehensive Feature Demo 🚀

**Best for:** Understanding all DriftGuard capabilities

```bash
python demo_comprehensive.py
```

**What it demonstrates:**
- Data drift detection (numerical & categorical)
- Concept drift detection with ADWIN
- Performance monitoring and comparison
- Training vs Production metrics

**Expected output:**
- Statistical test results for different features
- Drift points detected in streaming data
- Performance comparison tables

---

### 3. Real-World Chatbot Scenario 🤖

**Best for:** Seeing practical application

```bash
python demo_chatbot.py
```

**What it demonstrates:**
- Week-by-week monitoring simulation
- Detecting topic shifts in customer queries
- Real-world drift scenarios (e.g., COVID-19 impact)
- Performance tracking over time

**Expected output:**
- Weekly drift analysis
- Actionable insights and recommendations
- Performance trend summary

---

### 4. Original Example Demos

**LLM Demo:**
```bash
python examples/llm_demo.py
```

**Traditional ML Monitoring:**
```bash
python examples/monitoring.py
```

**Concept Drift Visualization:**
```bash
python examples/concept_drift.py
```

---

## Running Tests

To verify everything works correctly:

```bash
# Run LLM tests
python -m pytest tests/llm.py -v

# Or use unittest
python tests/llm.py
```

---

## Interactive Dashboard

To launch the interactive monitoring dashboard:

```bash
python src/visualization/example.py
```

Then open your browser to `http://localhost:8050`

---

## Understanding the Output

### Drift Scores

- **Drift Score < Threshold**: ✅ No drift detected (data distribution is similar)
- **Drift Score > Threshold**: 🚨 Drift detected (data has changed significantly)

### Performance Metrics

**For Traditional ML:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC

**For LLM:**
- ROUGE (text similarity)
- BLEU (translation quality)
- BERTScore (semantic similarity)
- Perplexity (language model quality)

### P-Values (Statistical Tests)

- **p-value < 0.05**: Statistically significant drift
- **p-value >= 0.05**: No significant drift

---

## Customizing Demos

You can modify the demo files to test with your own data:

### Example: Custom Drift Detection

```python
from src.llm.llm_drift import LLMDriftDetector

# Initialize
detector = LLMDriftDetector(drift_threshold=0.15)

# Set your baseline
your_reference_texts = ["your", "training", "data"]
detector.set_reference(your_reference_texts)

# Test production data
your_production_texts = ["your", "production", "data"]
result = detector.detect_drift(your_production_texts)

print(f"Drift: {result['drift_detected']}, Score: {result['drift_score']}")
```

---

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root:

```bash
export PYTHONPATH="${PYTHONPATH}:/home/ghyath/projects/mlops/DriftGuard"
```

Or add to your script:
```python
import sys
sys.path.insert(0, '/home/ghyath/projects/mlops/DriftGuard')
```

### Model Download Issues

The first run may take time to download transformer models. Ensure you have internet connection.

### Memory Issues

For large datasets, consider:
- Reducing batch sizes
- Using a smaller embedding model
- Processing data in chunks

---

## Next Steps

1. ✅ Run all demos to understand features
2. 📊 Check `logs/metrics_drift.log` for logged data
3. 🔧 Modify demos with your own data
4. 🚀 Integrate into your ML pipeline
5. 📈 Set up monitoring dashboard for production

---

## Questions?

- Check the main README.md
- Review source code documentation
- Contact: gheathmousa@gmail.com

