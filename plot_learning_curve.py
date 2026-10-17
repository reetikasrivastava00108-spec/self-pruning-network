
import numpy as np
import matplotlib.pyplot as plt


# Load the metrics recorded during training.
history = np.loadtxt(
    "learning_curve.csv",
    delimiter=",",
    skiprows=1
)

# Read the columns from the CSV file.
epochs = history[:, 0]
train_loss = history[:, 1]
train_accuracy = history[:, 2]
test_loss = history[:, 3]
test_accuracy = history[:, 4]

# Plot the loss curves.
plt.figure()
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# Plot the accuracy curves.
plt.figure()
plt.plot(epochs, train_accuracy * 100, label="Training Accuracy")
plt.plot(epochs, test_accuracy * 100, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved loss_curve.png and accuracy_curve.png")
