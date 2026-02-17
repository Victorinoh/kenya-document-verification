"""
Data Loader for Document Verification System
Loads and preprocesses images for CNN training
"""
import cv2
import numpy as np
from pathlib import Path
import random


class DocumentDataLoader:
    """
    Loads document images for training/validation/testing
    """
    
    # Class labels
    DOCUMENT_TYPES = {
        'national_ids': 0,
        'kcse_certificates': 1,
        'passports': 2
    }
    
    AUTHENTICITY = {
        'genuine': 0,
        'fake': 1
    }
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.data = []  # List of (filepath, doc_type_label, auth_label)
    
    def load_split(self, split='train'):
        """
        Load all images for a given split (train/validation/test)
        
        Args:
            split: 'train', 'validation', or 'test'
        
        Returns:
            List of (filepath, doc_type_label, authenticity_label)
        """
        self.data = []
        doc_types = ['national_ids', 'kcse_certificates', 'passports']
        
        if split in ['train', 'validation']:
            # Genuine images
            for doc in doc_types:
                if split == 'train':
                    folder = Path(f"data/augmented/train/{doc}")
                else:
                    folder = Path(f"data/augmented/validation/{doc}")
                
                if folder.exists():
                    for img_path in folder.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.png']:
                            self.data.append((
                                str(img_path),
                                self.DOCUMENT_TYPES[doc],
                                self.AUTHENTICITY['genuine']
                            ))
            
            # Fake images
            for doc in doc_types:
                if split == 'train':
                    folder = Path(f"data/augmented/train/fake/{doc}")
                else:
                    folder = Path(f"data/augmented/validation/fake/{doc}")
                
                if folder.exists():
                    for img_path in folder.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.png']:
                            self.data.append((
                                str(img_path),
                                self.DOCUMENT_TYPES[doc],
                                self.AUTHENTICITY['fake']
                            ))
        
        elif split == 'test':
            for auth in ['genuine', 'fake']:
                for doc in doc_types:
                    folder = Path(f"data/augmented/test/{auth}/{doc}")
                    if folder.exists():
                        for img_path in folder.iterdir():
                            if img_path.suffix.lower() in ['.jpg', '.png']:
                                self.data.append((
                                    str(img_path),
                                    self.DOCUMENT_TYPES[doc],
                                    self.AUTHENTICITY[auth]
                                ))
        
        random.shuffle(self.data)
        print(f"✅ Loaded {len(self.data)} images for {split} split")
        return self.data
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed numpy array (224, 224, 3) normalized to [0, 1]
        """
        # Load image
        img = cv2.imread(str(image_path))
        
        if img is None:
            # Return blank image if loading fails
            return np.zeros((*self.image_size, 3), dtype=np.float32)
        
        # Resize to target size
        img = cv2.resize(img, self.image_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_batch(self, batch_size=32, split='train'):
        """
        Load a batch of images and labels
        
        Returns:
            images: numpy array (batch_size, 224, 224, 3)
            doc_labels: numpy array (batch_size,)
            auth_labels: numpy array (batch_size,)
        """
        if not self.data:
            self.load_split(split)
        
        # Pick random batch
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        
        images = []
        doc_labels = []
        auth_labels = []
        
        for img_path, doc_label, auth_label in batch:
            img = self.preprocess_image(img_path)
            images.append(img)
            doc_labels.append(doc_label)
            auth_labels.append(auth_label)
        
        return (
            np.array(images),
            np.array(doc_labels),
            np.array(auth_labels)
        )
    
    def get_class_distribution(self):
        """Show class distribution in loaded data"""
        if not self.data:
            print("No data loaded. Call load_split() first.")
            return
        
        doc_counts = {0: 0, 1: 0, 2: 0}
        auth_counts = {0: 0, 1: 0}
        
        for _, doc_label, auth_label in self.data:
            doc_counts[doc_label] += 1
            auth_counts[auth_label] += 1
        
        doc_names = {v: k for k, v in self.DOCUMENT_TYPES.items()}
        
        print("\n📊 Class Distribution:")
        print("  Document Types:")
        for label, count in doc_counts.items():
            pct = count / len(self.data) * 100
            print(f"    {doc_names[label]}: {count} ({pct:.1f}%)")
        
        print("  Authenticity:")
        print(f"    Genuine: {auth_counts[0]} ({auth_counts[0]/len(self.data)*100:.1f}%)")
        print(f"    Fake:    {auth_counts[1]} ({auth_counts[1]/len(self.data)*100:.1f}%)")


def test_data_loader():
    """Quick test of the data loader"""
    print("=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    loader = DocumentDataLoader(image_size=(224, 224))
    
    # Load training data
    print("\n1. Loading training data...")
    train_data = loader.load_split('train')
    loader.get_class_distribution()
    
    # Load a batch
    print("\n2. Loading a batch of 8 images...")
    images, doc_labels, auth_labels = loader.load_batch(batch_size=8, split='train')
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {doc_labels.shape}")
    print(f"   Pixel range: [{images.min():.2f}, {images.max():.2f}]")
    
    # Load validation data
    print("\n3. Loading validation data...")
    loader2 = DocumentDataLoader()
    val_data = loader2.load_split('validation')
    loader2.get_class_distribution()
    
    print("\n✅ Data loader working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loader()