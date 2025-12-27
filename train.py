"""
PlantVillage Disease Classification - Training Script
Author: Eşref Batuhan Simsar
Dataset: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
from pathlib import Path

# Configuration
class Config:
    DATA_DIR = "/path/to/plantvillage/dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    IMG_SIZE = 224
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_MIXED_PRECISION = True
    
    # Classes
    CLASSES = [
        "Apple___Apple_scab",
        "Apple___Black_rot", 
        "Apple___Cedar_apple_rust",
        "Apple___healthy"
    ]

# Data Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_dataloaders():
    """Create train, validation, and test dataloaders"""
    # Load full dataset
    full_dataset = datasets.ImageFolder(Config.DATA_DIR)
    
    # Split dataset (70% train, 15% val, 15% test)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_model(from_scratch=True):
    """
    Create ConvNeXt-Tiny model
    
    Args:
        from_scratch: If True, train from scratch. If False, use transfer learning.
    """
    if from_scratch:
        # Train from scratch
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(768, Config.NUM_CLASSES)
    else:
        # Transfer learning
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Freeze early layers
        for param in model.features[:6].parameters():
            param.requires_grad = False
        # Replace classifier
        model.classifier[2] = nn.Linear(768, Config.NUM_CLASSES)
    
    return model.to(Config.DEVICE)

def train_epoch(model, train_loader, criterion, optimizer, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        if Config.USE_MIXED_PRECISION and scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            if Config.USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_model(from_scratch=True):
    """Main training function"""
    print(f"\n{'='*70}")
    print(f"Training Mode: {'From Scratch' if from_scratch else 'Transfer Learning'}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Create model
    model = create_model(from_scratch=from_scratch)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.USE_MIXED_PRECISION else None
    
    # Training loop
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/best_model_{'scratch' if from_scratch else 'transfer'}.pth")
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Test
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model, best_val_acc

if __name__ == "__main__":
    # Train from scratch
    print("\n🔥 Training from scratch...")
    model_scratch, acc_scratch = train_model(from_scratch=True)
    
    # Train with transfer learning
    print("\n\n🚀 Training with transfer learning...")
    model_transfer, acc_transfer = train_model(from_scratch=False)
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"From Scratch:      {acc_scratch:.2f}%")
    print(f"Transfer Learning: {acc_transfer:.2f}%")
    print(f"Improvement:       {acc_transfer - acc_scratch:.2f}%")
    print("="*70)
