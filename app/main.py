from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import strawberry
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, Perceptron
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

models = [
    LogisticRegression(),
    RidgeClassifierCV(),
    RidgeClassifier(),
    SVC(),
    NuSVC(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    HistGradientBoostingClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier(),
    BaggingClassifier(),
    KNeighborsClassifier(),
    NearestCentroid(),
    BernoulliNB(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    CalibratedClassifierCV(SVC()), 
    Perceptron(),
    DummyClassifier()
]

model_results = []
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=1),
        "recall": recall_score(y_test, y_pred, zero_division=1),
        "f1_score": f1_score(y_test, y_pred, zero_division=1),
    })

# Define GraphQL Schema
@strawberry.type
class ModelResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@strawberry.type
class Query:
    @strawberry.field
    def get_model_performance(self, model_name: str = None) -> list[ModelResult]:
        if model_name:
            for result in model_results:
                if result["model"] == model_name:
                    return [ModelResult(**result)]
            raise ValueError("Model not found")
        return [ModelResult(**result) for result in model_results]

schema = strawberry.Schema(query=Query)

app = FastAPI()

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
