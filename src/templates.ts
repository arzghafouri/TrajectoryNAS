import { ProjectTemplate } from './types';

export const PYTHON_TEMPLATES: ProjectTemplate[] = [
  {
    id: 'pytorch-cnn',
    title: 'شبکه عصبی عمیق PyTorch (مدل CNN)',
    description: 'پروژه استاندارد پایتون برای آموزش شبکه عصبی کانولوشنی روی تصاویر با PyTorch، بهینه‌ساز AdamW و نمودارهای دقت.',
    category: 'Deep Learning',
    icon: 'Cpu',
    hyperparameters: {
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: 'AdamW',
      modelArchitecture: 'ResNet-18 Custom'
    },
    files: [
      {
        name: 'main.py',
        language: 'python',
        content: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvNeuralNet

def train_model():
    print("🚀 Initializing PyTorch Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dummy dataset loader or MNIST
    print("📦 Loading dataset...")
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ConvNeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("🎯 Model Architecture created successfully!")
    print(model)

if __name__ == "__main__":
    train_model()
`
      },
      {
        name: 'model.py',
        language: 'python',
        content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
`
      },
      {
        name: 'requirements.txt',
        language: 'text',
        content: `torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
`
      },
      {
        name: 'README.md',
        language: 'markdown',
        content: `# PyTorch CNN Image Classifier

این پروژه یک ساختار استاندارد پایتون برای آموزش مدل‌های بینایی ماشین با PyTorch است.

## نحوه اجرا:
\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`
`
      }
    ]
  },
  {
    id: 'huggingface-llm',
    title: 'تنظیم دقیق (Fine-Tuning) با Hugging Face',
    description: 'اسکریپت پایتون برای بارگذاری مدل‌های ترنسفورمر، توکنایزر و آموزش روی دیتاست‌های سفارشی با Hugging Face Transformers.',
    category: 'NLP & LLM',
    icon: 'Sparkles',
    hyperparameters: {
      epochs: 3,
      batchSize: 8,
      learningRate: 0.00005,
      optimizer: 'AdamW (HFC)',
      modelArchitecture: 'distilbert-base-uncased'
    },
    files: [
      {
        name: 'main.py',
        language: 'python',
        content: `from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def main():
    print("🤗 Loading Hugging Face Transformers Pipeline...")
    model_name = "distilbert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    print(f"Model {model_name} loaded successfully with 2 output labels.")
    
    # Sample training data
    data = {
        "text": ["This is an amazing AI model!", "Terrible experience, very buggy.", "Love using Python for AI.", "Completely broken and useless."],
        "label": [1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("✨ Dataset tokenized and ready for training!")

if __name__ == "__main__":
    main()
`
      },
      {
        name: 'requirements.txt',
        language: 'text',
        content: `transformers>=4.30.0
datasets>=2.12.0
torch>=2.0.0
accelerate>=0.20.0
pandas>=2.0.0
`
      },
      {
        name: 'README.md',
        language: 'markdown',
        content: `# Hugging Face Fine-Tuning Script

پروژه پایتون برای دسته‌بندی متن و تنظیم دقیق مدل‌های زبان بزرگ (LLM).
`
      }
    ]
  },
  {
    id: 'fastapi-serving',
    title: 'سرویس مدل هوش مصنوعی با FastAPI',
    description: 'پیاده‌سازی وب‌سرویس سریع و مدرن پایتون با FastAPI برای اینفرنس لحظه‌ای (Inference) مدل‌های یادگیری ماشین.',
    category: 'Deployment',
    icon: 'Server',
    hyperparameters: {
      epochs: 1,
      batchSize: 1,
      learningRate: 0.01,
      optimizer: 'Uvicorn ASGI',
      modelArchitecture: 'FastAPI REST API'
    },
    files: [
      {
        name: 'app.py',
        language: 'python',
        content: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AI Model Serving API", version="1.0.0")

class InferenceRequest(BaseModel):
    text: str
    max_length: int = 100

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    status: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Model Serving API is running active."}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    # Simulate model inference
    return {
        "prediction": f"Processed: {request.text[:30]}...",
        "confidence": 0.954,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
`
      },
      {
        name: 'requirements.txt',
        language: 'text',
        content: `fastapi>=0.95.0
uvicorn>=0.22.0
pydantic>=2.0.0
`
      },
      {
        name: 'README.md',
        language: 'markdown',
        content: `# FastAPI AI Serving Endpoint

راه‌اندازی سرویس API برای مدل‌های هوش مصنوعی با داکیومنت‌سازی خودکار Swagger UI.
`
      }
    ]
  },
  {
    id: 'langchain-rag',
    title: 'سیستم پرسش و پاسخ هوشمند (RAG با LangChain)',
    description: 'سیستم جستجو و بازیابی اطلاعات پیشرفته (RAG) با LangChain، چانک‌بندی متون و اتصال به مدل‌های زبانی.',
    category: 'LLM Agents',
    icon: 'BookOpen',
    files: [
      {
        name: 'rag_pipeline.py',
        language: 'python',
        content: `from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

def initialize_rag():
    print("📚 Initializing LangChain RAG Pipeline...")
    
    # Sample Knowledge base
    sample_text = """
    PyAI Studio is an advanced development environment for Python AI engineers.
    It supports PyTorch, Hugging Face, FastAPI, and LangChain frameworks.
    You can train models, visualize loss curves, and export clean Python code.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = text_splitter.split_text(sample_text)
    print(f"Split document into {len(chunks)} semantic chunks.")
    print("🔗 Vector Store & Retriever ready for queries!")

if __name__ == "__main__":
    initialize_rag()
`
      },
      {
        name: 'requirements.txt',
        language: 'text',
        content: `langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.2
`
      },
      {
        name: 'README.md',
        language: 'markdown',
        content: `# LangChain RAG Pipeline

پروژه پایتون برای پیاده‌سازی سیستم‌های پرسش از اسناد با هوش مصنوعی.
`
      }
    ]
  }
];
