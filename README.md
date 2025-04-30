📊 Multimodal Credit Scoring Using LLN, CNN & Honey Bee Optimization

This project presents a hybrid deep learning solution for credit score classification using multimodal data fusion. It combines the strength of Liquid Neural Networks (LLN) and Convolutional Neural Networks (CNN) to model both sequential and tabular banking data. The model is further optimized using the Honey Bee Algorithm (HBO) for hyperparameter tuning, resulting in a robust and adaptive credit risk prediction framework.

🔍 Problem Statement

Credit scoring is crucial in assessing the financial trustworthiness of borrowers. Traditional scoring models struggle to integrate diverse types of financial data. This project addresses that limitation by:

Utilizing multiple data modalities (sequential, categorical, numerical)

Applying advanced neural architectures and fusion techniques

Enabling adaptive hyperparameter tuning through nature-inspired optimization

🚀 Project Objectives

Build a multimodal deep learning architecture for credit scoring.

Apply LLN for sequential behavior modeling and CNN for pattern recognition in tabular features.

Design and compare early, intermediate, and late fusion strategies.

Use the Honey Bee Optimizer to fine-tune neural architecture and learning parameters.

Evaluate performance using multiple statistical and classification metrics.

🧠 Key Features

🔄 Multimodal Fusion Strategies

Early Fusion: Combines raw LLN and CNN input features before training.

Intermediate Fusion: Merges hidden representations of LLN and CNN mid-network (default approach).

Late Fusion: Combines independent model predictions for final decision via averaging.

🐝 Honey Bee Optimization

Custom implementation of the HBO algorithm is used to optimize:

GRU unit size for LLN

Filter size and kernel size for CNN

Dense layer dimensions

Learning rate

📊 Evaluation Metrics

Accuracy (up to 80% on intermediate fusion)

F1-Score: 0.6275 (weighted)

AUPRC: 0.6709 (Area Under Precision-Recall Curve)

ROC-AUC: Multi-class ROC curves plotted and scored

Confusion Matrix: Provides class-level error analysis

📁 Project Structure

├── Historical_Banking_Data.csv              # Raw input dataset
├── cnn_numerical_tabular_dataset.csv        # Tabular subset for CNN model
├── lln_cnn_early_fusion_model.h5            # Saved early fusion model
├── lln_cnn_intermediate_fusion_model.h5     # Saved intermediate fusion model
├── lln_cnn_late_fusion_model.h5             # Saved late fusion model
├── multimodal_fusion.py / notebook.ipynb    # Training script or notebook
├── README.md                                # Project documentation


📈 Sample Results (Intermediate Fusion)

Metric	Value
Accuracy	~80%
Weighted F1-Score	0.6275
AUPRC	0.6709
ROC-AUC (Class 0)	0.82
ROC-AUC (Class 1)	0.77
ROC-AUC (Class 2)	0.79




🛠️ Tools & Technologies

Python (3.8+)

TensorFlow & Keras

NumPy, Pandas, Matplotlib

Scikit-learn for evaluation

Custom Honey Bee Optimizer (Python)


📚 Learnings & Impact

Learned to build multimodal deep learning pipelines using real-world financial data.

Designed and tuned liquid time-constant networks using GRU layers.

Understood and benchmarked fusion strategies for model integration.

Applied custom metaheuristics for deep learning optimization.


📜 License

This project is open-source under the MIT License.
