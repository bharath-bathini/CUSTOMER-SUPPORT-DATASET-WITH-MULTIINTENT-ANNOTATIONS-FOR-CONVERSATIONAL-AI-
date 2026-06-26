import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

class MetricsCalculator:
    def __init__(self, labels, category):
        """
        Initialize MetricsCalculator with labels and category.
        
        Args:
            labels: List of class labels
            category: Name of the category (e.g., 'intent', 'sentiment', etc.)
        """
        self.labels = labels
        self.category = category
        self.precision = []
        self.recall = []
        self.fscore = []
        self.accuracy = []

        self.metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        self.class_report_df = pd.DataFrame()
        self.class_performance_dfs = {}

    def calculate_metrics(self, algorithm, predict, y_test, y_score=None):
        """
        Calculate and save performance metrics, confusion matrix, ROC curve, and text images.
        
        Args:
            algorithm: Name of the algorithm (e.g., 'Decision Tree', 'Random Forest')
            predict: Predicted labels
            y_test: True labels
            y_score: Predicted probabilities/scores for ROC curve (optional)
        """
        
        # Create directory structure: results/classifier/category
        self.base_dir = f'results/{algorithm}/{self.category}'
        os.makedirs(self.base_dir, exist_ok=True)
            
        categories = self.labels

        # Calculate overall metrics
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100

        # Append to DataFrame
        metrics_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Accuracy': [a],
            'Precision': [p],
            'Recall': [r],
            'F1-Score': [f]
        })
        self.metrics_df = pd.concat([self.metrics_df, metrics_entry], ignore_index=True)

        # Print metrics
        print(f"{algorithm} Accuracy  : {a:.2f}")
        print(f"{algorithm} Precision : {p:.2f}")
        print(f"{algorithm} Recall    : {r:.2f}")
        print(f"{algorithm} FScore    : {f:.2f}")

        # Save metrics text image
        metrics_text = (
            f"{algorithm} Accuracy  : {a:.2f}\n"
            f"{algorithm} Precision : {p:.2f}\n"
            f"{algorithm} Recall    : {r:.2f}\n"
            f"{algorithm} FScore    : {f:.2f}"
        )
        plt.figure(figsize=(6, 2))
        plt.text(0.01, 0.5, metrics_text, fontsize=12, family='monospace', va='center')
        plt.axis('off')
        plt.savefig(f"{self.base_dir}/{algorithm.replace(' ', '_')}_metrics_text.png", bbox_inches='tight')
        plt.close()

        # Classification report
        CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)
        print(f"{algorithm} Classification Report")
        print(f"{algorithm}\n{classification_report(y_test, predict, target_names=[str(c) for c in categories])}\n")

        # Save classification report image
        cr_text = f"{algorithm}\n{classification_report(y_test, predict, target_names=[str(c) for c in categories])}"
        plt.figure(figsize=(8, 4))
        plt.text(0.01, 0.5, cr_text, fontsize=10, family='monospace', va='center')
        plt.axis('off')
        plt.savefig(f"{self.base_dir}/{algorithm.replace(' ', '_')}_classification_report.png", bbox_inches='tight')
        plt.close()

        # Classification report dataframe
        cr_df = pd.DataFrame(CR).transpose()
        cr_df['Algorithm'] = algorithm
        self.class_report_df = pd.concat([self.class_report_df, cr_df], ignore_index=False)

        # Class-specific performance
        for category in categories:
            class_entry = pd.DataFrame({
                'Algorithm': [algorithm],
                'Precision': [CR[str(category)]['precision'] * 100],
                'Recall': [CR[str(category)]['recall'] * 100],
                'F1-Score': [CR[str(category)]['f1-score'] * 100],
                'Support': [CR[str(category)]['support']]
            })

            if str(category) not in self.class_performance_dfs:
                self.class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

            self.class_performance_dfs[str(category)] = pd.concat([self.class_performance_dfs[str(category)], class_entry], ignore_index=True)

        # Plot confusion matrix
        plt.figure()
        ax = sns.heatmap(confusion_matrix(y_test, predict), xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, len(categories)])
        plt.title(f"{algorithm} Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(f"{self.base_dir}/{algorithm.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        # ROC Curve Plot
        if y_score is None:
            print("[WARNING] y_score is None. Cannot plot ROC.")
            return

        n_classes = len(categories)
        plt.figure(figsize=(10, 8))

        # Binary classification
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])  # probability for class 1
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {categories[1]} (AUC = {roc_auc:.2f})")

        # Multiclass classification
        else:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title(f"{algorithm} ROC Curve{'s' if n_classes > 2 else ''} (One-vs-Rest)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.base_dir}/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.close()

    def overall_classifiers_comparision(self):
        """Plot and save overall performance comparison across classifiers."""
        melted_df = self.metrics_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
        
        save_dir = f'results/{self.category}'
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Algorithm", y="Score", hue="Metric", data=melted_df)
        plt.title("Overall Performance Comparison of Classifiers")
        plt.ylabel("Score (%)")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(f"{save_dir}/1Overall_performance_comparison.png")

        plt.close()

        df = melted_df.drop_duplicates()

        # Pivot for better readability
        pivot_df = df.pivot(index="Algorithm", columns="Metric", values="Score")
        pivot_df = pivot_df.sort_values(by="Accuracy", ascending=True)
        return pivot_df

    def class_specific_classifiers_comparision(self):
        """Plot and save class-specific performance comparison across classifiers."""
        for class_label, df in self.class_performance_dfs.items():
            melted_df = df.melt(id_vars="Algorithm", value_vars=["Precision", "Recall", "F1-Score"],
                                var_name="Metric", value_name="Score")

            save_dir = f'results/{self.category}'
            os.makedirs(save_dir, exist_ok=True)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Algorithm", y="Score", hue="Metric", data=melted_df)
            plt.title(f"Performance Comparison for Class {class_label}")
            plt.ylabel("Score (%)")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{class_label}_performance_comparison.png")
            plt.close()
        all_dfs = []
        for label, df in self.class_performance_dfs.items():
            df = df.copy()
            df["Label"] = label
            all_dfs.append(df)

        # Combine all into one DataFrame
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Reorder columns
        final_df = final_df[["Label", "Algorithm", "Precision", "Recall", "F1-Score", "Support"]]

        # Sort by Label then Algorithm
        final_df = final_df.sort_values(by=["Label", "Algorithm"]).reset_index(drop=True)

        return final_df
