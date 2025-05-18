import pandas as pd
def clean_and_summary(file):
    df = pd.read_csv(file) 
    numeric_col = ["English", "Science", "Math"]

    # Fills any missing numeric values with the mean of the column.
    df[numeric_col] = df[numeric_col].fillna(df[numeric_col].mean())

    #  Converts Gender into binary values. 
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    # Returns a summary DataFrame showing average scores per gender. 
    
    average_score = df.groupby("Gender")[numeric_col].mean()
    return average_score
