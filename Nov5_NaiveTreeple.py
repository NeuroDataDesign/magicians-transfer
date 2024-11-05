from treeple import RandomForestClassifier
import numpy as np

class NaiveTransferRandomForest:
    def __init__(self, n_estimators=1000, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs
        )
        self.is_trained = False

    def pre_train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        print("Pre-training completed.")

    def transfer_train(self, X, y):
        if self.is_trained is False:
            raise RuntimeError("Pre_train() first.")
        self.model.partial_fit(X, y)
        print("Transfer training completed.")

    def predict(self, X, y):
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Test accuracy: {accuracy*100:.2f}%")
        return accuracy
