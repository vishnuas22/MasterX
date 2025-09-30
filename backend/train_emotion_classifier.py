"""
Fine-tune BERT for Educational Emotion Detection

This script fine-tunes BERT on the GoEmotions dataset + educational augmentation
to achieve 85-90% accuracy on emotion classification for learning contexts.

Dataset: GoEmotions (58k examples) + Educational Augmentation
Model: BERT-base-uncased (110M parameters)
Target Accuracy: 85-90%
Training Time: ~2-3 hours on GPU, ~8-10 hours on CPU

Usage:
    python train_emotion_classifier.py --epochs 3 --batch-size 16
"""

import asyncio
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EDUCATIONAL EMOTION MAPPING
# ============================================================================

# Map GoEmotions categories to our 13 educational emotions
EMOTION_MAPPING = {
    # Confusion & Struggle
    'confusion': 'confusion',
    'disappointment': 'confusion',
    'nervousness': 'anxiety',
    'fear': 'fear',
    'embarrassment': 'anxiety',
    
    # Frustration
    'annoyance': 'frustration',
    'anger': 'frustration',
    'disapproval': 'frustration',
    
    # Positive Learning
    'joy': 'joy',
    'excitement': 'excitement',
    'love': 'satisfaction',
    'gratitude': 'satisfaction',
    'admiration': 'satisfaction',
    'approval': 'confidence',
    'pride': 'confidence',
    'realization': 'breakthrough_moment',
    
    # Engagement
    'curiosity': 'engagement',
    'surprise': 'engagement',
    'amusement': 'engagement',
    'desire': 'engagement',
    'optimism': 'engagement',
    
    # Disengagement
    'sadness': 'struggle',
    'grief': 'struggle',
    'remorse': 'struggle',
    'relief': 'neutral',
    'caring': 'neutral',
    'neutral': 'neutral'
}

# Our 13 educational emotion categories
EDUCATIONAL_EMOTIONS = [
    'confusion', 'frustration', 'anxiety', 'fear',
    'joy', 'excitement', 'satisfaction', 'confidence',
    'breakthrough_moment', 'struggle', 'boredom', 'engagement', 'neutral'
]

EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EDUCATIONAL_EMOTIONS)}
ID_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_ID.items()}


# ============================================================================
# EDUCATIONAL AUGMENTATION DATA
# ============================================================================

# Real educational scenarios for data augmentation
EDUCATIONAL_AUGMENTATION = [
    # Confusion
    ("I don't understand what this means", "confusion"),
    ("This concept is really confusing me", "confusion"),
    ("I'm lost on this topic", "confusion"),
    ("Can you explain this again?", "confusion"),
    ("I'm not following this explanation", "confusion"),
    ("What does this even mean?", "confusion"),
    ("I'm confused about the relationship between these concepts", "confusion"),
    ("This doesn't make sense to me", "confusion"),
    ("I'm having trouble understanding the difference", "confusion"),
    ("Could you clarify this part?", "confusion"),
    
    # Frustration
    ("This is so frustrating, I've been stuck on this for hours", "frustration"),
    ("I keep getting this wrong no matter what I try", "frustration"),
    ("Why isn't this working?", "frustration"),
    ("I've tried everything and nothing works", "frustration"),
    ("This is impossible to understand", "frustration"),
    ("I'm so frustrated with this problem", "frustration"),
    ("No matter how many times I review this, I don't get it", "frustration"),
    ("This is driving me crazy", "frustration"),
    ("I can't figure this out at all", "frustration"),
    ("Why is this so difficult?", "frustration"),
    
    # Anxiety
    ("I'm worried I won't be able to learn this", "anxiety"),
    ("What if I fail this test?", "anxiety"),
    ("I'm nervous about the exam", "anxiety"),
    ("I don't think I'm smart enough for this", "anxiety"),
    ("Everyone else seems to get it but I don't", "anxiety"),
    ("I'm scared I'm falling behind", "anxiety"),
    ("This is making me really anxious", "anxiety"),
    ("I'm not sure I can do this", "anxiety"),
    ("What if I never understand this topic?", "anxiety"),
    ("I'm afraid of making mistakes", "anxiety"),
    
    # Fear
    ("I'm afraid I'll never understand this", "fear"),
    ("This terrifies me", "fear"),
    ("I'm scared of this subject", "fear"),
    ("I have a fear of failing", "fear"),
    
    # Joy
    ("I love learning this!", "joy"),
    ("This is so much fun!", "joy"),
    ("I'm really enjoying this topic", "joy"),
    ("This makes me happy", "joy"),
    ("I'm having a great time learning this", "joy"),
    
    # Excitement
    ("This is amazing! I can't wait to learn more!", "excitement"),
    ("I'm so excited about this concept!", "excitement"),
    ("Wow! This is incredible!", "excitement"),
    ("I'm thrilled to be learning this!", "excitement"),
    ("This is absolutely fascinating!", "excitement"),
    ("I can't believe how cool this is!", "excitement"),
    
    # Satisfaction
    ("I'm proud of myself for figuring this out", "satisfaction"),
    ("That felt really good to complete", "satisfaction"),
    ("I'm satisfied with my progress", "satisfaction"),
    ("This is rewarding", "satisfaction"),
    ("I feel accomplished", "satisfaction"),
    ("I'm happy with what I've learned", "satisfaction"),
    
    # Confidence
    ("I think I'm getting the hang of this", "confidence"),
    ("I feel more confident now", "confidence"),
    ("I believe I can do this", "confidence"),
    ("I'm starting to understand this better", "confidence"),
    ("I feel good about my understanding", "confidence"),
    ("I'm confident I can solve this", "confidence"),
    ("I know I can figure this out", "confidence"),
    
    # Breakthrough moment
    ("Oh! I finally get it!", "breakthrough_moment"),
    ("Everything just clicked!", "breakthrough_moment"),
    ("Aha! Now I understand!", "breakthrough_moment"),
    ("It all makes sense now!", "breakthrough_moment"),
    ("I finally figured it out!", "breakthrough_moment"),
    ("Now I see how it all connects!", "breakthrough_moment"),
    ("This just came together for me!", "breakthrough_moment"),
    ("Wow, I finally understand this concept!", "breakthrough_moment"),
    
    # Struggle
    ("I'm struggling with this material", "struggle"),
    ("This is really challenging for me", "struggle"),
    ("I'm having a hard time with this", "struggle"),
    ("I'm not doing well with this topic", "struggle"),
    ("I keep making mistakes", "struggle"),
    
    # Boredom
    ("This is boring", "boredom"),
    ("I'm not interested in this", "boredom"),
    ("This isn't engaging at all", "boredom"),
    ("I'm losing interest", "boredom"),
    ("This is tedious", "boredom"),
    
    # Engagement
    ("This is interesting", "engagement"),
    ("I'm curious about this", "engagement"),
    ("Tell me more about this", "engagement"),
    ("I want to learn more", "engagement"),
    ("This caught my attention", "engagement"),
    ("I'm engaged in this topic", "engagement"),
    ("This is fascinating", "engagement"),
    ("I'm paying close attention", "engagement"),
    
    # Neutral
    ("Let's continue", "neutral"),
    ("Okay, I'm following", "neutral"),
    ("I see", "neutral"),
    ("Alright", "neutral"),
    ("Next topic please", "neutral"),
    ("Moving on", "neutral"),
]


# ============================================================================
# DATASET CLASS
# ============================================================================

class EmotionDataset(Dataset):
    """Custom dataset for emotion classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of emotion labels (as IDs)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MODEL CLASS
# ============================================================================

class EmotionBERTClassifier(nn.Module):
    """BERT-based emotion classifier."""
    
    def __init__(self, bert_model_name: str = 'bert-base-uncased', num_classes: int = 13, dropout: float = 0.1):
        """
        Initialize classifier.
        
        Args:
            bert_model_name: Pre-trained BERT model name
            num_classes: Number of emotion classes
            dropout: Dropout probability
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each emotion class
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ============================================================================
# DATA LOADING
# ============================================================================

def load_goemotions_dataset(limit: int = None) -> List[Tuple[str, str]]:
    """
    Load GoEmotions dataset from HuggingFace.
    
    Args:
        limit: Optional limit on number of samples
        
    Returns:
        List of (text, emotion) tuples
    """
    logger.info("Loading GoEmotions dataset...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('go_emotions', 'simplified')
        
        # Convert to our format
        data = []
        for split in ['train', 'validation']:
            for item in dataset[split]:
                text = item['text']
                # Get emotion name from label
                emotion_id = item['labels'][0] if item['labels'] else 27  # 27 is neutral
                emotion_name = dataset[split].features['labels'].feature.names[emotion_id]
                
                # Map to our educational emotions
                mapped_emotion = EMOTION_MAPPING.get(emotion_name, 'neutral')
                data.append((text, mapped_emotion))
                
                if limit and len(data) >= limit:
                    break
            
            if limit and len(data) >= limit:
                break
        
        logger.info(f"Loaded {len(data)} samples from GoEmotions")
        return data
        
    except Exception as e:
        logger.warning(f"Could not load GoEmotions: {e}")
        logger.info("Using educational augmentation data only")
        return []


def prepare_dataset(use_goemotions: bool = True, augment: bool = True) -> Tuple[List[str], List[int]]:
    """
    Prepare complete training dataset.
    
    Args:
        use_goemotions: Whether to include GoEmotions data
        augment: Whether to include educational augmentation
        
    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    emotions = []
    
    # Add GoEmotions if available
    if use_goemotions:
        try:
            goemotions_data = load_goemotions_dataset()
            for text, emotion in goemotions_data:
                texts.append(text)
                emotions.append(emotion)
        except Exception as e:
            logger.warning(f"Skipping GoEmotions: {e}")
    
    # Add educational augmentation
    if augment:
        logger.info(f"Adding {len(EDUCATIONAL_AUGMENTATION)} educational examples")
        for text, emotion in EDUCATIONAL_AUGMENTATION:
            texts.append(text)
            emotions.append(emotion)
    
    # Convert emotions to IDs
    labels = [EMOTION_TO_ID[emotion] for emotion in emotions]
    
    logger.info(f"Total dataset size: {len(texts)} samples")
    
    # Print emotion distribution
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    logger.info("Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {emotion}: {count}")
    
    return texts, labels


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for emotion detection')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--output-dir', type=str, default='/app/backend/models/emotion_bert_educational', help='Output directory')
    parser.add_argument('--skip-goemotions', action='store_true', help='Skip GoEmotions dataset')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    texts, labels = prepare_dataset(
        use_goemotions=not args.skip_goemotions,
        augment=True
    )
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    
    # Create datasets and dataloaders
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    logger.info("Initializing model...")
    model = EmotionBERTClassifier(num_classes=len(EDUCATIONAL_EMOTIONS))
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_accuracy = 0
    training_history = []
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_dataloader, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc * 100
        })
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            logger.info(f"✅ New best accuracy: {val_acc*100:.2f}% - Saving model...")
            
            # Save model
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            # Save classification report
            report = classification_report(
                val_labels,
                val_preds,
                target_names=EDUCATIONAL_EMOTIONS,
                output_dict=True
            )
            
            with open(output_dir / 'classification_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"\nClassification Report (Best Model):")
            logger.info(classification_report(
                val_labels,
                val_preds,
                target_names=EDUCATIONAL_EMOTIONS
            ))
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Save emotion mappings
    with open(output_dir / 'emotion_mappings.json', 'w') as f:
        json.dump({
            'emotion_to_id': EMOTION_TO_ID,
            'id_to_emotion': ID_TO_EMOTION,
            'emotions': EDUCATIONAL_EMOTIONS
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_accuracy*100:.2f}%")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
