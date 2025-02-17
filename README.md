# GraphClassify 

GraphClassify is a FastAPI-based GraphQL service that benchmarks multiple classification models on the Breast Cancer dataset. It provides real-time performance insights, including accuracy, precision, recall, and F1-score. 

## Features 
- Benchmarks multiple ML classifiers
- GraphQL-powered API for flexible queries
- FastAPI backend for high-performance requests
- Provides accuracy, precision, recall, and F1-score metrics
- Uses the Breast Cancer dataset for evaluation

## Installation 

Clone the repository and install the dependencies:

```sh
# Clone the repository
git clone https://github.com/yourusername/GraphClassify.git
cd GraphClassify

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Running the Project 

```sh
uvicorn main:app --host 127.0.0.1 --port 8000
```

The API will be available at: **`http://127.0.0.1:8000/graphql`**

## GraphQL Query Example 

You can use the GraphQL Playground or send a query via a tool like Postman:

```graphql
query {
  getModelPerformance(modelName: "RandomForestClassifier") {
    model
    accuracy
    precision
    recall
    f1_score
  }
}
```

## Models Used 
- Logistic Regression
- Random Forest
- SVM (Support Vector Machine)
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors
- Gradient Boosting
- And more...

## Contributing 
Contributions are welcome! Feel free to fork the repo and submit a PR.

