# NLP-Based Language Detection Project

## **Overview**
Language detection is a Natural Language Processing (NLP) task that involves identifying the language of a given text. Using machine learning models, this project predicts the language of user-provided text based on a trained dataset of multilingual samples.

---

## **Project Workflow**

### 1. **Data Collection**
- The dataset (`languages.csv`) contains two columns:
  - `Text`: Text samples in multiple languages.
  - `Language`: The corresponding language of each text sample.
- Example dataset:
  ```csv
  Text,Language
  "Bonjour",French
  "Hello",English
  "Hola",Spanish
  ```

### 2. **Data Preprocessing**
- **Objective**: Convert text into numerical features for machine learning.
- Steps:
  1. **Tokenization**: Break text into smaller units (e.g., words).
  2. **Vectorization**:
     - Use **Bag-of-Words (BoW)** or **TF-IDF** to represent text numerically.
  3. **Splitting the Dataset**:
     - Split data into **Training**, **Validation**, and **Test** sets:
       ```python
       from sklearn.model_selection import train_test_split

       # First split: Training and Temporary (Validation + Test)
       X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.33, random_state=42)
       
       # Second split: Validation and Test
       X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
       ```
  4. **Vectorizing Text Data**:
     ```python
     from sklearn.feature_extraction.text import CountVectorizer

     cv = CountVectorizer()
     X_train_cv = cv.fit_transform(X_train)
     X_val_cv = cv.transform(X_val)
     X_test_cv = cv.transform(X_test)
     ```

---

### 3. **Model Selection & Training**
- Models used:
  - **Naive Bayes**
  - **Support Vector Machine (SVM)**
  - **Random Forest**
  - **Neural Network (MLPClassifier)**

- **Training and Evaluation**:
  ```python
  from sklearn.metrics import accuracy_score

  results = {}
  for name, model in models.items():
      print(f"Training {name}...")
      model.fit(X_train_cv, y_train)

      # Validate the model
      y_val_pred = model.predict(X_val_cv)
      val_accuracy = accuracy_score(y_val, y_val_pred)
      print(f"{name} Validation Accuracy: {val_accuracy * 100:.2f}%")

      # Test the model
      y_test_pred = model.predict(X_test_cv)
      test_accuracy = accuracy_score(y_test, y_test_pred)
      print(f"{name} Test Accuracy: {test_accuracy * 100:.2f}% \n")

      results[name] = test_accuracy

  # Display model performance
  print("\nModel Comparison:")
  for name, accuracy in results.items():
      print(f"{name}: Test Accuracy = {accuracy * 100:.2f}% \n")
  ```

---

### 4. **Language Prediction for User Input**
- Predict the language using the best-performing model:
  ```python
  # Select the best model
  best_model_name = max(results, key=results.get)
  best_model = models[best_model_name]

  # Get user input and predict language
  user_input = input("Enter a text: ").strip()
  if user_input:
      data = cv.transform([user_input]).toarray()
      output = best_model.predict(data)
      print(f"Predicted Language by {best_model_name}: {output[0]}")
  else:
      print("No input provided. Please enter some texts.")
  ```

---

## **Tools and Libraries**
- **Python Libraries**:
  - `pandas`, `numpy`: Data manipulation.
  - `sklearn`: Machine learning models and utilities.
  - `CountVectorizer`: Text vectorization.
- **Machine Learning Models**:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Neural Network (MLPClassifier).

---

## **Challenges and Solutions**
- **Handling Mixed Languages**: Ensure the dataset contains examples of mixed-language texts for better generalization.
- **Short Texts**: Use models like Naive Bayes, which perform well with sparse data.
- **Data Imbalance**: Balance the dataset to avoid bias towards dominant languages.

---

## **Project Outcome**
- The project builds a trained language detection model that can:
  - Accurately identify the language of user-provided text.
  - Handle multilingual datasets efficiently.
  - Provide insights into model performance with validation and test accuracy metrics.

---

## **Future Improvements**
- Add support for additional languages.
- Incorporate deep learning models like transformers (e.g., BERT) for better accuracy.
- Develop a web-based interface for user interaction.

---

## **References**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- NLP Tutorials and Textbooks.
