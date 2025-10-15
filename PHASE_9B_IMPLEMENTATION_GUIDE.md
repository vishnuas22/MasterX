# Phase 9B Implementation Guide
## ML Training & Integration - Google Colab Instructions

**Date:** October 15, 2025  
**Status:** 🚀 Ready for Training  
**Target:** <100ms latency, >85% accuracy

---

## What's Been Created ✅

### 1. Neural Models (`/app/backend/services/emotion/neural_models.py`)
- ✅ PADRegressor (268 lines)
- ✅ LearningReadinessNet (236 lines)
- ✅ InterventionNet (194 lines)
- ✅ TemperatureScaler (87 lines)
- ✅ EmotionNeuralModels (unified wrapper)

### 2. Training Script (`/app/backend/train_emotion_40_colab.py`)
- ✅ 40-emotion classifier training
- ✅ Multi-task learning (emotion + PAD)
- ✅ Google Colab optimized
- ✅ Checkpoint saving
- ✅ Validation metrics

---

## Google Colab Training Instructions

### Step 1: Setup Colab Notebook

```python
# Install dependencies
!pip install transformers torch torchvision torchaudio
!pip install datasets scikit-learn tqdm

# Clone or upload your code
from google.colab import drive
drive.mount('/content/drive')

# Upload training script
from google.colab import files
uploaded = files.upload()  # Upload train_emotion_40_colab.py

# Upload neural_models.py
uploaded = files.upload()  # Upload neural_models.py
```

### Step 2: Download GoEmotions Dataset

```python
# Download GoEmotions dataset
!wget https://github.com/google-research/google-research/raw/master/goemotions/data/train.tsv
!wget https://github.com/google-research/google-research/raw/master/goemotions/data/dev.tsv
!wget https://github.com/google-research/google-research/raw/master/goemotions/data/test.tsv

# Create data directory
!mkdir -p /content/data
!mv *.tsv /content/data/
```

### Step 3: Modify Training Script for Real Data

Replace the synthetic data section (lines 550-560) with:

```python
# Load GoEmotions dataset
import pandas as pd

def load_goemotions(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['text', 'emotions', 'id']
    
    texts = []
    emotions = []
    
    for _, row in df.iterrows():
        text = row['text']
        # GoEmotions has multiple labels, take first one
        emotion_list = row['emotions'].split(',')
        
        for emotion_name in emotion_list:
            if emotion_name in GOEMOTIONS_TO_40_EMOTIONS:
                mapped_emotion = GOEMOTIONS_TO_40_EMOTIONS[emotion_name]
                texts.append(text)
                emotions.append(mapped_emotion)
                break
    
    return texts, emotions

# Load train and validation
train_texts, train_emotions = load_goemotions('/content/data/train.tsv')
val_texts, val_emotions = load_goemotions('/content/data/dev.tsv')
```

### Step 4: Run Training

```python
# Run training script
!python train_emotion_40_colab.py

# Monitor GPU usage
!nvidia-smi
```

### Step 5: Download Trained Model

```python
# Download best model
from google.colab import files

files.download('/content/models/emotion_classifier_40_best.pt')
files.download('/content/models/emotion_classifier_40_final.pt')
```

---

## What to Train (Priority Order)

### Priority 1: Main Emotion Classifier ⭐⭐⭐
**Script:** `train_emotion_40_colab.py`  
**Dataset:** GoEmotions (58k examples)  
**Time:** 2-4 hours on T4 GPU  
**Target:** >85% accuracy on 40 emotions

**Deliverable:**
- `emotion_classifier_40_best.pt` (main model weights)
- Validation accuracy report
- Confusion matrix

### Priority 2: PAD Regressor Training ⭐⭐
**Method:** Joint training with main classifier (already included)  
**Dataset:** Same GoEmotions + synthetic PAD annotations  
**Time:** Included in main training  
**Target:** MAE <0.1 on PAD scores

**Deliverable:**
- PAD regressor weights (included in main checkpoint)
- PAD score validation metrics

### Priority 3: Neural Models Training ⭐
**Models:** ReadinessNet, InterventionNet  
**Dataset:** Need to create effectiveness labels  
**Time:** 1-2 hours additional  
**Status:** Can be done later (Phase 9C)

**Note:** For now, use the untrained/random initialization. We can train these separately with effectiveness data later.

### Priority 4: Temperature Scaling ⭐
**Method:** Post-hoc calibration on validation set  
**Time:** 5 minutes  
**Implementation:** Already in TemperatureScaler.calibrate()

---

## Expected Training Results

### Target Metrics

| Metric | Target | Acceptable | Needs Work |
|--------|--------|------------|------------|
| Accuracy | >85% | >80% | <80% |
| F1 Score | >0.83 | >0.78 | <0.78 |
| Latency | <100ms | <200ms | >200ms |
| PAD MAE | <0.1 | <0.15 | >0.15 |

### GPU Recommendations

| GPU | Training Time | Batch Size | Notes |
|-----|---------------|------------|-------|
| T4 | 3-4 hours | 32 | Good balance |
| V100 | 1.5-2 hours | 64 | Faster |
| A100 | 45-60 min | 128 | Best |
| CPU | 24+ hours | 8 | Not recommended |

---

## After Training: Integration Steps

### Step 1: Upload Trained Model to MasterX

```bash
# On your local machine, after downloading from Colab
scp emotion_classifier_40_best.pt user@masterx-server:/app/backend/models/lightweight_emotion/
```

Or upload via GitHub:
```bash
git add emotion_classifier_40_best.pt
git commit -m "Add trained 40-emotion classifier"
git push
```

### Step 2: Update emotion_transformer.py

The loading code is already there (lines 408-424):

```python
model_path = "/app/backend/models/lightweight_emotion/emotion_classifier_40.pt"

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    self.classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded trained classifier from {model_path}")
```

Just rename your file or update the path.

### Step 3: Update emotion_engine.py (Remove Hardcoded Values)

**File:** `/app/backend/services/emotion/emotion_engine.py`

**Lines to replace:** 601-858

**Current (hardcoded):**
```python
# Lines 601-622: Hardcoded emotion lists
positive_emotions = [
    EmotionCategory.JOY.value,
    EmotionCategory.EXCITEMENT.value,
    # ...
]
```

**Replace with (learned):**
```python
# Use learned PAD scores from PADRegressor
pad_scores = self.neural_models.pad_regressor(emotion_embedding)
emotional_factor = pad_scores[0].item()  # Use pleasure dimension
```

**Lines 836-858: Hardcoded valence mappings**

**Replace with:**
```python
def _calculate_emotion_valence(self, emotion_embedding: torch.Tensor) -> float:
    """Calculate emotional valence using learned PAD regressor."""
    with torch.no_grad():
        pad_scores = self.neural_models.pad_regressor(emotion_embedding)
        return pad_scores[0].item()  # Pleasure dimension
```

### Step 4: Test Integration

```python
cd /app/backend
python3 << 'EOF'
import asyncio
from services.emotion.emotion_engine import EmotionEngine

async def test():
    engine = EmotionEngine()
    await engine.initialize()
    
    result = await engine.analyze_emotion(
        user_id="test",
        text="I'm so frustrated with this problem!",
        context={},
        behavioral_data={}
    )
    
    print(f"Emotion: {result.metrics.primary_emotion}")
    print(f"PAD: {result.metrics.valence:.2f}")
    print(f"Readiness: {result.metrics.learning_readiness}")
    print(f"Time: {result.metrics.analysis_time_ms:.1f}ms")

asyncio.run(test())
EOF
```

---

## Troubleshooting

### Issue 1: OOM (Out of Memory)
**Solution:**
- Reduce batch_size: 32 → 16 → 8
- Use gradient accumulation
- Enable FP16 training

```python
# Add to training script
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Issue 2: Low Accuracy (<80%)
**Solutions:**
1. Train longer (20 epochs instead of 10)
2. Lower learning rate (2e-5 → 1e-5)
3. Add more data augmentation
4. Use RoBERTa instead of BERT

### Issue 3: Slow Training
**Solutions:**
1. Use V100 or A100 GPU (faster than T4)
2. Increase batch size if memory allows
3. Enable torch.compile() in model
4. Use mixed precision (FP16)

### Issue 4: Model Not Loading
**Solution:**
```python
# Check checkpoint structure
checkpoint = torch.load('model.pt')
print(checkpoint.keys())

# Load with error handling
try:
    model.load_state_dict(checkpoint['model_state_dict'])
except KeyError:
    model.load_state_dict(checkpoint)  # Direct state dict
```

---

## Validation Checklist

Before deploying trained model:

- [ ] Accuracy >85% on test set
- [ ] F1 score >0.83
- [ ] All 40 emotions represented in predictions
- [ ] PAD scores in valid range [0, 1]
- [ ] Model file size <1GB
- [ ] Latency <200ms on CPU (for testing)
- [ ] No errors on edge cases (empty text, very long text)

---

## Next Steps After Training

### Immediate (Phase 9B Completion)
1. ✅ Train 40-emotion classifier on Colab
2. ✅ Download trained weights
3. ✅ Upload to MasterX server
4. ✅ Update emotion_engine.py
5. ✅ Test integration
6. ✅ Measure latency

### Short-term (Phase 9C)
1. Train ReadinessNet on effectiveness data
2. Train InterventionNet on outcome data
3. Calibrate TemperatureScaler
4. Full system testing

### Long-term (Phase 10)
1. Deploy to GPU environment
2. Achieve <100ms latency target
3. A/B testing with users
4. Continuous model improvement

---

## Files Created Summary

```
/app/backend/services/emotion/
├── neural_models.py           ✅ NEW (700+ lines)
│   ├── PADRegressor
│   ├── LearningReadinessNet
│   ├── InterventionNet
│   ├── TemperatureScaler
│   └── EmotionNeuralModels

/app/backend/
├── train_emotion_40_colab.py  ✅ NEW (700+ lines)
│   ├── Training configuration
│   ├── Dataset loader
│   ├── Multi-task model
│   ├── Training loop
│   └── Validation

/app/
├── PHASE_9B_IMPLEMENTATION_GUIDE.md  ✅ THIS FILE
└── PHASE_9A_VERIFICATION_REPORT.md   ✅ PREVIOUS
```

---

## Questions & Support

### For Training Issues:
1. Check GPU availability: `!nvidia-smi`
2. Monitor memory: `torch.cuda.memory_summary()`
3. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`

### For Integration Issues:
1. Check model loading: View `/var/log/supervisor/backend.err.log`
2. Test imports: `python -c "from services.emotion.neural_models import *"`
3. Verify paths: `ls -la /app/backend/models/lightweight_emotion/`

---

## Success Criteria

Phase 9B is complete when:
- ✅ 40-emotion classifier trained (>85% accuracy)
- ✅ Trained model integrated and loading
- ✅ emotion_engine.py updated (no hardcoded values)
- ✅ End-to-end test passing
- ✅ Latency measured and documented

---

**Ready to train on Google Colab!** 🚀

Upload these files to Colab:
1. `train_emotion_40_colab.py`
2. `neural_models.py`
3. Follow instructions above
4. Train with GPU
5. Download weights
6. Integrate and test

Good luck with training! Let me know results.
