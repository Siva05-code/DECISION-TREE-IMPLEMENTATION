# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SIVAKARTHICK B
I
*INTERN ID*: : CT08FYO

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH


**Wine Quality Prediction using Decision Tree Classifier**

### **Project Overview**
Wine quality assessment is an essential task in the wine industry, as it helps in evaluating different wine samples based on their physicochemical properties. This project utilizes **Machine Learning** techniques, particularly the **Decision Tree Classifier**, to predict the quality of red wine based on various chemical attributes. The dataset used for this project is the **Wine Quality Dataset**, which contains features such as acidity, sugar levels, pH, alcohol content, and other factors that influence wine quality.

### **Tools and Technologies Used**
For this project, the following tools and technologies were employed:
1. **Programming Language:** Python
2. **Data Handling & Analysis:** Pandas, NumPy
3. **Data Visualization:** Matplotlib, Seaborn
4. **Machine Learning:** Scikit-learn (DecisionTreeClassifier, train_test_split, StandardScaler, accuracy_score, classification_report, confusion_matrix, plot_tree)
5. **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix

### **Dataset Description**
The dataset comprises multiple physicochemical attributes that affect wine quality. The key features include:
- **Fixed acidity:** The amount of non-volatile acids in the wine.
- **Volatile acidity:** The level of acetic acid, which affects the wine's taste and smell.
- **Citric acid:** Adds freshness and enhances flavor.
- **Residual sugar:** The sugar content after fermentation.
- **Chlorides:** The amount of salt in wine.
- **Free sulfur dioxide & Total sulfur dioxide:** Important for preventing spoilage.
- **Density:** Determines the alcohol content.
- **pH:** The acidity level of wine.
- **Sulphates:** Influences microbial stability.
- **Alcohol:** Affects the body and taste of the wine.
- **Quality:** The target variable, which ranges from 3 to 8.

To simplify the classification process, wine quality was **categorized into three classes**:
- **Low Quality (3-5)**
- **Medium Quality (6)**
- **High Quality (7-8)**

### **Implementation Steps**
1. **Data Preprocessing:**
   - Load the dataset and check for missing values.
   - Convert wine quality into categorical labels (Low, Medium, High).
   - Map categorical labels to numerical values (0, 1, 2).

2. **Feature Engineering:**
   - Select all columns except `quality` and `quality_label` as feature variables (X).
   - Use `quality_label` as the target variable (y).
   
3. **Data Splitting:**
   - Split the dataset into **80% training** and **20% testing** sets.
   - Use **stratified sampling** to maintain the class balance.

4. **Feature Scaling:**
   - Normalize the features using `StandardScaler` to ensure uniformity and improve model performance.

5. **Model Training:**
   - Use the **Decision Tree Classifier** with **Gini impurity** as the criterion and a max depth of 10 to prevent overfitting.
   - Train the model on the processed training dataset.

6. **Model Evaluation:**
   - Test the model on unseen test data.
   - Compute **accuracy, classification report, and confusion matrix**.
   - Visualize the **decision tree structure** to understand decision-making criteria.

### **Results and Insights**
- The model achieved an accuracy of approximately **85-90%**, indicating strong predictive power.
- The **classification report** provided **precision, recall, and F1-score** for each class.
- The **confusion matrix** revealed correct and incorrect classifications, helping to identify misclassifications.
- The **decision tree visualization** showed how different features contribute to predicting wine quality.

### **Applications of this Project**
This wine quality prediction system can be applied in multiple domains:
1. **Wine Industry:**
   - Helps winemakers assess and improve wine quality.
   - Can be used for automated wine grading based on chemical composition.

2. **Food and Beverage Sector:**
   - Assists in ensuring consistent quality for consumers.
   - Provides insights into how various chemical properties influence taste.

3. **Quality Control & Standardization:**
   - Aids in developing **quality benchmarks** for different wine types.
   - Ensures compliance with regulatory standards.

4. **Retail & E-Commerce:**
   - Can be integrated into wine-selling platforms for recommending quality-based selections.
   - Helps consumers make informed purchasing decisions.

5. **Educational and Research Purposes:**
   - Useful for students and researchers in machine learning, chemistry, and food science.
   - Provides a real-world dataset for studying classification techniques.

### **Conclusion**
This project effectively demonstrates the power of **Decision Tree Classification** in predicting wine quality based on its physicochemical properties. The model's accuracy and insights make it a valuable tool for winemakers, retailers, and researchers. By integrating advanced machine learning techniques and expanding datasets, this system can be further refined for more **precise and reliable wine quality predictions**.


### **Output**


![Image](https://github.com/user-attachments/assets/a07387a2-59ab-4fc2-8e30-3c3b6bacc803)


<img width="394" alt="Image" src="https://github.com/user-attachments/assets/8be49ee1-cc47-458e-9306-c66d1e7941d4" />
