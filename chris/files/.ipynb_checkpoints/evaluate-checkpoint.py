import torch
from sklearn.metrics import accuracy_score, classification_report

def test_model(model, test_loader, device):
    """
    Evaluates the model on the test set.
    
    Args:
    - model: Trained model
    - test_loader: DataLoader for test data
    - device: 'cuda' or 'cpu'
    
    Returns:
    - test_acc: Test accuracy
    - report: Classification report
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=["Normal", "Pneumonia"])

    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print("\nClassification Report:\n", report)

    return test_acc, report
