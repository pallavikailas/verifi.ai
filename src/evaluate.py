from sklearn.metrics import accuracy_score, f1_score

def evaluate(predictions, labels):
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1
