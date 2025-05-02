import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("dengAI Dashboard")
section = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Prediction", "Forecast"])


# ---------- Load Data ----------
@st.cache_data
def load_data():
    train = pd.read_csv("../data/01_raw/dengue_features_train.csv")
    labels = pd.read_csv("../data/01_raw/dengue_labels_train.csv")
    train = train.merge(labels, on=["city", "year", "weekofyear"])
    model = joblib.load("../data/05_models/model.pkl")
    final_model = joblib.load("../data/05_models/final_model.pkl")
    validation = pd.read_parquet("../data/03_model_input/validation_data.parquet")
    X_test = pd.read_parquet("../data/03_model_input/X_test.parquet")
    y_test = pd.read_csv("../data/03_model_input/y_test.csv")
    return train, model, final_model, validation, X_test, y_test


train, model, final_model, validation, X_test, y_test = load_data()

# ---------- Overview ----------
if section == "Overview":
    st.title("🦟 dengAI Competition Dashboard")
    st.markdown(
        """
    Predicting the number of dengue cases using environmental data.

    - Two cities: San Juan (SJ), Iquitos (IQ)
    - Weather and climate variables: temperature, humidity, precipitation, etc.
    - Goal: Forecast `total_cases`
    """
    )

    st.subheader("Dataset Samples")
    st.write(train.head())

# ---------- EDA ----------
elif section == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    city = st.selectbox("Select city", train["city"].unique())

    city_data = train[train["city"] == city]

    st.subheader("Dengue Cases Over Time")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(pd.to_datetime(city_data["week_start_date"]), city_data["total_cases"])
    ax.set_title(f"Dengue Cases in {city}")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = city_data.drop(
        columns=["city", "year", "weekofyear", "week_start_date"]
    ).corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
    st.pyplot(fig)

# ---------- Model Prediction ----------
elif section == "Model Prediction":
    st.title("🤖 Model Prediction")

    # Predict
    preds = model.predict(X_test)

    # Prepare the results
    results = X_test.copy()
    results["predicted_cases"] = preds
    results["actual_cases"] = y_test.values  # Align via index

    # Create a new 'year-week' column for plotting
    results["year-week"] = (
        results["year"].astype(str) + "-W" + results["weekofyear"].astype(str)
    )

    # Select city
    city = st.selectbox("Select city for modeling", ["sj", "iq"], key="model_city")
    city_code = 1 if city == "sj" else 2

    # Filter for the selected city and reset index
    results_city = results[results["city"] == city_code].reset_index(drop=True)

    # Sort by the 'year-week' column to ensure the data is ordered by time
    results_city = results_city.sort_values(by="year-week")

    # Get the train data for the selected city (assuming you have a `train` DataFrame)
    train_city = train[train["city"] == city].reset_index(drop=True)
    train_city["year-week"] = (
        train_city["year"].astype(str) + "-W" + train_city["weekofyear"].astype(str)
    )
    train_city = train_city.sort_values(
        by="year-week"
    )  # Sort to maintain correct order

    # Plotting: year-week as x-axis
    st.subheader(f"Actual vs Predicted for {city.upper()}")
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot the train data in the background (with transparency)
    ax.plot(
        train_city["year-week"],
        train_city["total_cases"],
        label="Train Data",
        color="blue",
        alpha=0.3,
    )

    # Plot the actual and predicted values for test data as scatter points
    ax.scatter(
        results_city["year-week"],
        results_city["actual_cases"],
        label="Actual (Test)",
        color="blue",
        zorder=5,
        s=10,
    )
    ax.scatter(
        results_city["year-week"],
        results_city["predicted_cases"],
        label="Predicted (Test)",
        color="orange",
        zorder=5,
        s=10,
    )

    # Improve x-axis labels for readability (rotate labels)
    step = 50  # Adjust this value if you need more or fewer labels
    ax.set_xticks(train_city["year-week"][::step])  # Show every 5th label
    ax.set_xticklabels(train_city["year-week"][::step], rotation=45, ha="right")

    # Set labels and legend
    ax.set_xlabel("Year-Week")
    ax.set_ylabel("Cases")
    ax.legend()

    # Display the plot
    st.pyplot(fig)


elif section == "Forecast":
    st.title("📈 Forecast on Test Set")

    # Generate predictions
    test_copy = validation.copy()
    test_copy["predicted_cases"] = model.predict(validation)

    # Create 'year-week' column
    test_copy["year-week"] = (
        test_copy["year"].astype(str) + "-W" + test_copy["weekofyear"].astype(str)
    )

    # Select city
    city = st.selectbox("Select city for modeling", ["sj", "iq"], key="model_city")
    city_code = 1 if city == "sj" else 2
    city_data = test_copy[test_copy["city"] == city_code].reset_index(drop=True)

    # ------------------ User Upload ------------------
    st.subheader("🔄 Compare with Your Own Predictions")
    uploaded_file = st.file_uploader(
        "Upload your prediction file (CSV or Parquet)", type=["csv", "parquet"]
    )
    print(uploaded_file)

    user_preds = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_preds = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                user_preds = pd.read_parquet(uploaded_file)

            # Create 'year-week' if needed
            if "year-week" not in user_preds.columns and {
                "year",
                "weekofyear",
            }.issubset(user_preds.columns):
                user_preds["year-week"] = (
                    user_preds["year"].astype(str)
                    + "-W"
                    + user_preds["weekofyear"].astype(str)
                )

            # Filter by city
            if "city" in user_preds.columns:
                user_preds = user_preds[user_preds["city"] == city]

            # change col name
            user_preds.rename(columns={"total_cases": "predicted_cases"}, inplace=True)

            print(user_preds)

            # Merge on 'year-week' to align
            merged = pd.merge(
                city_data[["year-week", "predicted_cases"]],
                user_preds[["year-week", "predicted_cases"]],
                on="year-week",
                how="left",
                suffixes=("_ours", "_user"),
            )

            # Plot both series using Streamlit's line_chart for consistent look
            st.subheader(f"📊 Forecast Comparison for {city.upper()}")
            comparison_df = merged.set_index("year-week")[
                ["predicted_cases_ours", "predicted_cases_user"]
            ]
            comparison_df.columns = [
                "Our Prediction",
                "Your Prediction",
            ]  # Clean up column names
            st.line_chart(comparison_df)

        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    else:
        # Default chart if no file uploaded
        st.subheader(f"Predicted Cases for {city.upper()}")
        st.line_chart(city_data.set_index("year-week")[["predicted_cases"]])
