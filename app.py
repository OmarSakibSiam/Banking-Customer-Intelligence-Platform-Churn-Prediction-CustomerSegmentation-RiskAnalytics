from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Banking Customer Intelligence Platform
# XGBoost churn prediction + K-Means customer segmentation
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Banking Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "banking_customer_app.pkl"


# ----------------------------- Styling ------------------------------

st.markdown(
    """
    <style>
    /* ---------- Clean black background, white text ---------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #000000;
        color: #FFFFFF;
    }
    [data-testid="stHeader"] { background: #000000; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: #0B0B0B;
        border-right: 1px solid #2A2A2A;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }

    /* All text: pure white, fully opaque (no faded / blurred look) */
    h1, h2, h3, h4, h5, h6,
    .stApp p, .stApp li, .stApp label, .stApp span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    .stApp [data-testid="stSidebar"] [role="radiogroup"] label { opacity: 1 !important; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 13px 15px;
    }
    div[data-testid="stMetric"] * { color: #FFFFFF !important; }

    /* Form container */
    [data-testid="stForm"] {
        background: #070707;
        border: 1px solid #2A2A2A;
        border-radius: 12px;
    }

    /* Inputs and select boxes */
    div[data-baseweb="input"], div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        background: #111111 !important;
        border-color: #444444 !important;
    }
    input, textarea {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        background: transparent !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] button { color: #FFFFFF !important; }
    div[data-baseweb="select"] svg, div[data-baseweb="input"] svg { fill: #FFFFFF !important; }

    /* Dropdown menus */
    div[data-baseweb="popover"], div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"], [role="listbox"] {
        background: #111111 !important;
    }
    [role="option"], [role="option"] * { color: #FFFFFF !important; }
    [role="option"]:hover { background: #2A2A2A !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #0B0B0B;
        border: 1px solid #2A2A2A;
        border-radius: 10px;
    }
    [data-testid="stExpander"] * { color: #FFFFFF !important; }

    /* Dividers */
    hr { border-color: #2A2A2A !important; }

    .small-note { color: #FFFFFF; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Load model ----------------------------

@st.cache_resource(show_spinner="Loading churn and segmentation models...")
def load_artifacts(path: str):
    return joblib.load(path)


if not ARTIFACT_PATH.exists():
    st.error(
        "The model file was not found. Place "
        "`banking_customer_app.pkl` in the same folder as `app.py`."
    )
    st.code(
        "Banking-Customer-Intelligence/\n"
        "├── app.py\n"
        "├── banking_customer_app.pkl\n"
        "└── requirements.txt",
        language="text",
    )
    st.stop()

try:
    artifacts = load_artifacts(str(ARTIFACT_PATH))
except Exception as exc:
    st.error("The model artifact could not be loaded.")
    st.exception(exc)
    st.warning(
        "This pickle was created with specific library versions. "
        "Use the same scikit-learn, pandas, XGBoost and Python-compatible "
        "versions used in the notebook, then regenerate the artifact if needed."
    )
    st.stop()


# ----------------------------- Helpers -------------------------------

def get_transformed_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved ColumnTransformer and restore transformed column names."""
    transformer = artifacts["transformer"]
    transformed = transformer.transform(raw_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = artifacts.get("transformed_feature_names")
    if feature_names is None:
        feature_names = transformer.get_feature_names_out()

    feature_names = list(feature_names)

    if transformed.shape[1] != len(feature_names):
        raise ValueError(
            "The transformed feature count does not match the saved feature names. "
            "Check that the preprocessing transformer and model were saved together."
        )

    return pd.DataFrame(transformed, columns=feature_names)


def make_customer_input(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
) -> pd.DataFrame:
    """Build the exact raw feature row expected by the saved transformer."""
    input_values = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": float(estimated_salary),
        # Derived automatically; the user does not enter this separately.
        "HasZeroBalance": int(float(balance) == 0.0),
    }

    raw_columns = artifacts.get("raw_input_columns")
    if raw_columns is None:
        raw_columns = list(artifacts["transformer"].feature_names_in_)
    raw_columns = list(raw_columns)

    missing = [column for column in raw_columns if column not in input_values]
    if missing:
        raise ValueError(
            "The saved model expects input columns not provided by this app: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [[input_values[column] for column in raw_columns]],
        columns=raw_columns,
    )


def predict_churn(transformed_df: pd.DataFrame):
    """Return predicted churn probability, class and threshold."""
    model = artifacts["churn_model"]
    threshold = float(artifacts.get("threshold", 0.5))

    # Preserve feature names if the XGBoost model was trained with a DataFrame.
    if hasattr(model, "feature_names_in_"):
        expected_names = list(model.feature_names_in_)
        model_input = transformed_df[expected_names]
    else:
        model_input = transformed_df.to_numpy()

    probabilities = model.predict_proba(model_input)[0]
    classes = list(model.classes_)

    if 1 in classes:
        churn_index = classes.index(1)
    else:
        # For a standard binary classifier, class 1 is the positive class.
        churn_index = 1

    churn_probability = float(probabilities[churn_index])
    predicted_class = int(churn_probability >= threshold)

    return churn_probability, predicted_class, threshold


def predict_customer_cluster(transformed_df: pd.DataFrame):
    """Scale the five saved clustering features and predict the K-Means cluster."""
    scaler = artifacts["scaler"]
    kmeans = artifacts["kmeans"]
    cluster_features = list(artifacts["cluster_features"])

    missing = [column for column in cluster_features if column not in transformed_df.columns]
    if missing:
        raise ValueError(
            "The saved clustering features are missing after transformation: "
            + ", ".join(missing)
        )

    cluster_input = transformed_df[cluster_features]
    scaled_input = scaler.transform(cluster_input)
    cluster_id = int(kmeans.predict(scaled_input)[0])

    names = artifacts.get("cluster_names", {})
    cluster_name = names.get(cluster_id, names.get(str(cluster_id), f"Cluster {cluster_id}"))

    return cluster_id, cluster_name


def get_cluster_profile_row(cluster_id: int):
    profile = artifacts.get("cluster_profile")
    if profile is None or not isinstance(profile, pd.DataFrame):
        return None

    if cluster_id in profile.index:
        return profile.loc[cluster_id]

    # Also handle a profile whose index was converted to strings.
    if str(cluster_id) in profile.index:
        return profile.loc[str(cluster_id)]

    return None


def display_cluster_profile(profile_row):
    if profile_row is None:
        st.info("No saved cluster profile was found in the model artifact.")
        return

    preferred = [
        ("Customer_Count", "Customers"),
        ("Average_Age", "Average age"),
        ("Average_Balance", "Average balance"),
        ("Median_Balance", "Median balance"),
        ("Average_Products", "Average products"),
        ("Active_Member_Rate", "Active member rate"),
        ("Zero_Balance_Rate", "Zero-balance rate"),
        ("Churn_Rate", "Historical cluster churn rate"),
    ]

    available = [(key, label) for key, label in preferred if key in profile_row.index]
    if not available:
        st.dataframe(profile_row.to_frame("Cluster profile"), use_container_width=True)
        return

    metric_cols = st.columns(min(4, len(available)))
    for index, (key, label) in enumerate(available):
        value = profile_row[key]
        if pd.isna(value):
            display_value = "N/A"
        elif key in ("Customer_Count",):
            display_value = f"{int(value):,}"
        elif key in ("Average_Balance", "Median_Balance"):
            display_value = f"{float(value):,.2f}"
        elif key in ("Active_Member_Rate", "Zero_Balance_Rate", "Churn_Rate"):
            display_value = f"{float(value):.2f}%"
        else:
            display_value = f"{float(value):.2f}"

        metric_cols[index % len(metric_cols)].metric(label, display_value)


# ----------------------------- Navigation ----------------------------

st.sidebar.title("🏦 Banking Intelligence")
st.sidebar.caption("Churn Prediction & Customer Segmentation")
page = st.sidebar.radio(
    "Navigation",
    ["Customer Analyzer", "Cluster Profiles", "About the Model"],
)

st.sidebar.divider()
st.sidebar.caption(f"Prediction model: {artifacts.get('churn_model_name', 'XGBoost')}")
st.sidebar.caption("Segmentation model: K-Means")


# ----------------------------- Page 1 --------------------------------

if page == "Customer Analyzer":
    st.title("Banking Customer Intelligence")
    st.markdown(
        "Enter a customer's details to estimate churn probability and identify "
        "the customer's behavioural segment."
    )

    st.subheader("Customer information")

    with st.form("customer_details"):
        left, right = st.columns(2)

        with left:
            credit_score = st.number_input(
                "Credit Score", min_value=350, max_value=850, value=650, step=1
            )
            geography = st.selectbox(
                "Geography",
                artifacts.get("categories", {}).get(
                    "Geography", ["France", "Spain", "Germany"]
                ),
            )
            gender = st.selectbox(
                "Gender",
                artifacts.get("categories", {}).get("Gender", ["Female", "Male"]),
            )
            age = st.number_input(
                "Age", min_value=18, max_value=92, value=38, step=1
            )
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=10, value=5, step=1
            )

        with right:
            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                max_value=250898.09,
                value=75000.0,
                step=1000.0,
                format="%.2f",
            )
            num_products = st.selectbox(
                "Number of Products", options=[1, 2, 3, 4], index=0
            )
            has_credit_card = st.selectbox(
                "Has Credit Card", options=["Yes", "No"], index=0
            )
            is_active_member = st.selectbox(
                "Active Member", options=["Yes", "No"], index=0
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=11.58,
                max_value=199992.48,
                value=100000.0,
                step=1000.0,
                format="%.2f",
            )

        submitted = st.form_submit_button(
            "Analyze Customer", type="primary", use_container_width=True
        )

    if submitted:
        try:
            raw_customer = make_customer_input(
                credit_score=credit_score,
                geography=geography,
                gender=gender,
                age=age,
                tenure=tenure,
                balance=balance,
                num_products=num_products,
                has_credit_card=(has_credit_card == "Yes"),
                is_active_member=(is_active_member == "Yes"),
                estimated_salary=estimated_salary,
            )

            transformed_customer = get_transformed_dataframe(raw_customer)

            churn_probability, churn_class, threshold = predict_churn(
                transformed_customer
            )

            cluster_id, cluster_name = predict_customer_cluster(
                transformed_customer
            )

            st.divider()
            st.header("Customer analysis results")

            result_left, result_right = st.columns(2)

            with result_left:
                st.subheader("Churn prediction")
                st.metric(
                    "Estimated churn probability",
                    f"{churn_probability * 100:.2f}%",
                )
                st.progress(churn_probability)

                if churn_class == 1:
                    st.error("Prediction: Likely to churn")
                else:
                    st.success("Prediction: Likely to stay")

                st.caption(
                    f"Decision threshold: {threshold:.2f}. "
                    "The prediction is based on the saved XGBoost classifier."
                )

            with result_right:
                st.subheader("Customer segment")
                st.metric("K-Means cluster", f"Cluster {cluster_id}")
                st.success(cluster_name)
                st.caption(
                    "The segment is assigned using the saved scaler and K-Means model."
                )

            st.divider()
            st.subheader("Behavioural profile of this segment")
            profile_row = get_cluster_profile_row(cluster_id)
            display_cluster_profile(profile_row)

            st.info(
                "The churn probability is the model's estimate for this individual customer. "
                "The cluster churn rate shown above is the historical average for the entire "
                "cluster; these are different measurements."
            )

            with st.expander("View the submitted customer information"):
                st.dataframe(raw_customer, use_container_width=True)

        except Exception as exc:
            st.error("The customer could not be analyzed. Check the saved model artifacts.")
            st.exception(exc)


# ----------------------------- Page 2 --------------------------------

elif page == "Cluster Profiles":
    st.title("Customer Segment Profiles")
    st.markdown(
        "Explore the three K-Means customer groups identified during your notebook analysis."
    )

    profile = artifacts.get("cluster_profile")
    cluster_names = artifacts.get("cluster_names", {})

    if isinstance(profile, pd.DataFrame):
        profile_display = profile.copy()

        if "Cluster_Name" not in profile_display.columns:
            profile_display.insert(
                0,
                "Cluster_Name",
                [
                    cluster_names.get(index, cluster_names.get(str(index), f"Cluster {index}"))
                    for index in profile_display.index
                ],
            )

        st.dataframe(profile_display, use_container_width=True)

        st.subheader("Explore an individual segment")
        cluster_ids = sorted(
            [int(value) for value in profile.index if str(value).isdigit()]
        )
        if not cluster_ids:
            cluster_ids = list(cluster_names.keys())

        if cluster_ids:
            selected_cluster = st.selectbox("Select cluster", cluster_ids)
            selected_name = cluster_names.get(
                selected_cluster,
                cluster_names.get(str(selected_cluster), f"Cluster {selected_cluster}"),
            )
            st.subheader(selected_name)

            row = get_cluster_profile_row(selected_cluster)
            display_cluster_profile(row)
    else:
        st.warning("The saved artifact does not contain a cluster_profile DataFrame.")


# ----------------------------- Page 3 --------------------------------

else:
    st.title("About the Models")

    st.subheader("Churn prediction — XGBoost")
    st.write(
        "The application uses the saved XGBoost classifier to estimate the probability "
        "that a customer will churn. The saved ColumnTransformer applies the same "
        "categorical encoding and feature preparation used during model training."
    )

    st.subheader("Customer segmentation — K-Means")
    st.write(
        "The application applies the saved preprocessing output, selects the five "
        "clustering features, applies the fitted StandardScaler and uses the fitted "
        "K-Means model to assign a cluster. The cluster number is then translated "
        "into the business name created during notebook analysis."
    )

    st.subheader("Important interpretation")
    st.warning(
        "Predictions are estimates based on the trained model and supplied dataset. "
        "They are not guarantees of future customer behaviour. A cluster's historical "
        "churn rate is not the same as an individual customer's predicted churn probability."
    )

    st.caption(
        "This application uses the saved model artifact. It does not retrain models "
        "when a visitor submits customer information."
    )
