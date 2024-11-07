from treeple import RandomForestClassifier
from treeple.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score

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
        #self.model.fit(X, y)
        self.model.partial_fit(X, y)
        print("Transfer training completed.")

    def predict(self, X, y):
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"Test accuracy: {accuracy*100:.2f}%")
        return accuracy

class LifeLongTrees:
    def __init__(self, n_estimators=1000):
        self.n_estimators = n_estimators
        self.trees = [DecisionTreeClassifier(max_features="sqrt") for _ in range(n_estimators)]
    def fit(self, x, y):
        for tree in self.trees:
            tree.fit(x, y)
    def predict(self,x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        return predictions

class LifeLongForest:   #for multitask
    def __init__(self):
        self.encoders = []
        self.tasks = []
    def add_task(self, n_estimators):
        forest = LifeLongTrees(n_estimators = n_estimators)
        self.encoders.append(forest)
        self.tasks.append(len(self.tasks))
    def fit(self, x, y, task_idx):
        if task_idx < len(self.encoders):
            forest = self.encoders[task_idx]
            forest.fit(x, y)
    def vote(self, x):
        all_trees_predictions = []
        for forest in self.encoders:
            tree_predictions = forest.predict(x)  #(n_trees, n_samples)
            all_trees_predictions.append(tree_predictions)
        return np.vstack(all_trees_predictions)  #(total_n_trees, n_samples)
    def decode(self, x, y):
        all_tree_votes = self.vote(x)  #(total_n_trees, n_samples)
        final_predictions = mode(all_tree_votes, axis=0)[0].flatten()   #(n_samples, )
        accuracy = accuracy_score(y, final_predictions)
        return accuracy