import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def housing_model(file):
    df = pd.read_csv(file)
    X = df[["area", "bedrooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict price using the other columns.
    y_pred = model.predict(X_test)

    #  Evaluate it using Mean Absolute Error (MAE).
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    # Scatter plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.show()

