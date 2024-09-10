import torch
import matplotlib.pyplot as plt
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOSS_HISTORY_PATH = "loss_history.pkl"

loss_history = []

if os.path.exists(LOSS_HISTORY_PATH):
  with open(LOSS_HISTORY_PATH, 'rb') as f:
    loss_history = pickle.load(f)

plt.plot(loss_history)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
