import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, '..', 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

df = pd.read_csv('train.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

means = train_df[['Age', 'Fare']].mean()
stds = train_df[['Age', 'Fare']].std()
for col in ['Age', 'Fare']:
    train_df[col] = (train_df[col] - means[col]) / stds[col]
    test_df[col] = (test_df[col] - means[col]) / stds[col]

X_train = torch.tensor(train_df.drop('Survived', axis=1).values, dtype=torch.float32)
y_train = torch.tensor(train_df['Survived'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(test_df.drop('Survived', axis=1).values, dtype=torch.float32)
y_test = torch.tensor(test_df['Survived'].values, dtype=torch.float32).reshape(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x)

model = Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses, train_accs, test_accs = [], [], []
epochs = 800

print("开始训练...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        train_preds = torch.sigmoid(model(X_train))
        train_acc = ((train_preds >= 0.5).float() == y_train).float().mean().item()
        train_accs.append(train_acc)

        test_preds = torch.sigmoid(model(X_test))
        test_acc = ((test_preds >= 0.5).float() == y_test).float().mean().item()
        test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    if (epoch + 1) % 10 == 0:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.title('Train and Test Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS_DIR, 'training_curve.png'))
        plt.close()

print(f"\n最终测试集准确率: {test_accs[-1]:.4f}")
torch.save(model.state_dict(), 'model.pth')
torch.save({'means': means, 'stds': stds}, 'preprocess.pth')
print("模型和预处理参数已直接保存至 src 目录下")

print("\n--- 请输入乘客信息进行预测 ---")
pclass = int(input("舱位(1/2/3): "))
sex = input("性别(male/female): ")
sex_num = 0 if sex == 'male' else 1
age = float(input("年龄: "))
age_norm = (age - means['Age']) / stds['Age']
sibsp = int(input("兄弟姐妹/配偶数: "))
parch = int(input("父母/子女数: "))
fare = float(input("票价: "))
fare_norm = (fare - means['Fare']) / stds['Fare']
embarked = input("登船港口(S/C/Q): ")
embarked_num = {'S': 0, 'C': 1, 'Q': 2}[embarked]

features = np.array([[pclass, sex_num, age_norm, sibsp, parch, fare_norm, embarked_num]], dtype=np.float32)

model.eval()
with torch.no_grad():
    output = model(torch.tensor(features))
    prob = torch.sigmoid(output).item()
    pred = 1 if prob >= 0.5 else 0

print(f"\n预测结果: {'幸存 (1)' if pred == 1 else '遇难 (0)'}")
print(f"幸存概率: {prob:.4f}")
