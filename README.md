# 2025-Y2-S1-MLB-B7G2-04

AIML Project: Breast Cancer Dataset Preprocessing

Overview:

This project focuses on preprocessing the Breast Cancer Wisconsin (Diagnostic) dataset using various Artificial Intelligence and Machine Learning (AIML) techniques. The dataset is sourced from Kaggle and includes features related to breast tumor characteristics (e.g., radius_mean, texture_mean) with a binary target variable 'diagnosis' (M for malignant, B for benign).
The preprocessing steps are implemented in individual Jupyter notebooks for each technique, with an integrated pipeline combining them. Techniques include handling missing data, imputation, encoding categorical variables, normalization/scaling, feature selection, and dimensionality reduction. The goal is to prepare the data for downstream modeling tasks like classification, while demonstrating robustness and interpretability through EDA visualizations.
Key libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn (for imputation, encoding, scaling, selection, PCA).

Dataset Details:

•	Source: Kaggle dataset wasiqaliyasir/breast-cancer-dataset (Apache 2.0 license).

•	Description: Contains 569 samples with 32 columns (including 'id' and 'diagnosis'). Features are computed from digitized images of fine needle aspirates (FNA) of breast masses, describing cell nuclei properties (e.g., mean, standard error, worst values for radius, texture, etc.).

•	Size: ~569 rows after loading (may vary if simulating missing data).

•	Target: 'diagnosis' (M = Malignant, B = Benign).

•	Location in Repo: Raw data is downloaded dynamically via Kaggle API in the notebooks (not committed due to size; use your own credentials to download). Processed outputs are saved in /data/ or /results/outputs/.

•	Notes: The dataset has no inherent missing values, but notebooks simulate them for demonstration (e.g., 5% in 'radius_mean' for imputation testing). One empty column ('Unnamed: 32') is dropped where present.


Group Member Roles:

Each member contributed a specific preprocessing technique.

•	Member 1 (IT24101668 - Handling Missing Data): Imputed numerical features with mean and dropped rows with missing 'diagnosis' / 'id'. Justification: Ensures complete dataset without bias. Includes heatmaps for missing values before/after.

•	Member 2 (IT24100556 - Encoding Categorical Variables): Encoded the 'diagnosis' column using LabelEncoder (M=1, B=0). Justification: Converts categorical target to numerical for ML compatibility. Includes countplot visualization showing class imbalance (more benign cases).

•	Member 3 (IT24101709 - Data Imputation with KNN): Implemented KNN imputation on simulated missing values in 'radius_mean'. Justification: Enhances robustness for incomplete datasets in cancer diagnostics. Includes histograms for before/after comparison.

•	Member 4 (IT24101674 - Normalization/Scaling): Scaled 'radius_mean' using MinMaxScaler (to [0,1] range). Justification: Normalizes features for scale-sensitive algorithms. Includes histograms for before/after distribution preservation.

•	Member 5 (IT24101803 - Feature Engineering: Feature Selection): Applied SelectKBest with ANOVA F-test (f_classif) to select top 5 features. Justification: Reduces redundancy by identifying discriminative features (e.g., high scores for 'radius_worst'). Includes barplot of feature importance scores.

•	Member 6 (IT24101669 - Feature Engineering: Dimension Reduction): Performed PCA to reduce to 2 components after standardization. Justification: Captures variance while reducing dimensionality for visualization/efficiency. Includes scatter plot showing class separation.

•	Integrated Pipeline (Combined Work): Combines all techniques: Download, mean imputation, encoding, MinMax scaling (all features), SelectKBest (k=5), PCA (n=2). Saves processed CSV and visualizes PCA scatter plot.


How to Run the Code:

1.	Prerequisites:
    
  o	Python 3.x environment (e.g., Google Colab, Jupyter Notebook, or local setup).
  
  o	Install dependencies: Run pip install pandas, numpy, matplotlib, seaborn, scikit-learn, kaggle in your environment.
  
  o	Kaggle API Setup: The notebooks download the dataset using hardcoded credentials (username: "harikaran123r", key: "0e4de6becd2e7a91773fd195c8878f3a"). For security and to avoid rate limits, replace with your own: 

    	Sign up on Kaggle, generate an API token from your account settings.
    
    	Update the kaggle_credentials dict in each notebook's first cell with your username and key.
    
    	Ensure the token is placed in ~/.kaggle/kaggle.json (chmod 600 for permissions).

2.	Running Individual Notebooks:
    
  o	Open a notebook (e.g., IT_Number_Preprocessing_technique.ipynb).
  
  o	Run cells sequentially. The first cell downloads/unzips the dataset to /content/data/.
  
  o	Outputs: Prints dataset stats, visualizations (e.g., histograms, plots), and interpretations.
  
  o	Note: If dataset appears empty (as in some outputs), verify Kaggle download or file path.

3.	Running the Integrated Pipeline:
	 
  o	Open group_pipeline.ipynb.
  
  o	Run all cells: Downloads data, applies all preprocessing steps, saves processed_breast_cancer_data.csv to /content/data/.
  
  o	Outputs: Visualizations (e.g., PCA scatter) and processed file.

4.	Results and Outputs:

  o	Visualizations: Saved or displayed inline (e.g., histograms in PNG/JPEG format under /results/eda_visualizations).
  
  o	Logs: Printed in notebooks (optional: redirect to /results/logs/).
  
  o	Final Outputs: Processed dataset in /results/outputs/ (e.g., CSV after pipeline).

5.	Troubleshooting:
	
  o	Dataset Download Fails: Check Kaggle API token validity or internet connection. Manually download from Kaggle and place in /content/data/Breast_cancer_dataset.csv.
  
  o	Empty Dataset Warning: Ensure the ZIP unzips correctly; the CSV might be named differently (e.g., 'breast-cancer.csv' – update paths if needed).
  
  o	GPU/colab-specific: Some notebooks (e.g., Encoding) request GPU, but it's optional.
  
  o	Reproducibility: Set seeds (e.g., np.random.seed(42)) for simulated missing data.
