import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set wider sidebar width
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="true"] ~ div[data-testid="stVerticalBlock"] {
        margin-left: 400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="XGBoost Classifier App", layout="wide")

# Header with centered text and right-aligned logo
logo_path = r"C:\Users\u1341435\OneDrive - MMC\Desktop\OpenCoesione\LOGO MARSH.jpg"
logo = Image.open(logo_path)

col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="margin-bottom: 0;">Marsh Advisory</h1>
            <h3 style="margin-top: 0;">XGBOOST_CLASSIFIER</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_right:
    st.image(logo, width=120)

# Default file path
default_file_path = r"C:\Users\U1341435\OneDrive - MMC\Desktop\OpenCoesione\Nuovo_Input_FESR_esteso.xlsx"

# Sidebar: File upload or use default
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("File loaded from upload")
else:
    try:
        df = pd.read_excel(default_file_path)
        st.sidebar.info(f"Loaded default file from:\n{default_file_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to load default file: {e}")
        st.stop()

# Show data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Predefined input columns
predefined_input_columns = [
    "OC_TEMA_SINTETICO",
    "OC_COD_TIPO_AIUTO",
    "OC_MACROAREA",
    "COD_TIPO_PROCED_ATTIVAZIONE",
    "PROVINCIA_BENEFICIARIO",
    "OC_DESCR_FORMA_GIU_BENEFICIARIO",
    "COD_OB_TEMATICO",
    "COD_PRIORITA_INVEST",
    "Cluster_SLL_REGIONE_DIMENSIONE",
    "DIMENSIONE ",
    "PRIV/TOT(LORDO)",
    "DAREP/TOT(LORDO)",
    "STATEST/TOT(LORDO)",
    "PUBB/TOT(LORDO)",
    "CONT_COMUNI",
    "CLASSIF_AREA_INTERNA",
    "DIMENSIONI_TEMP_PREVISTE",
    "CUP_NATURA_TIPOLOGIA",
    "CUP_SETTORE_SOTTOSETTORE_CATEGORIA"
]

# Sidebar: Select input columns
st.sidebar.header("Select Input Columns")
selected_columns = st.sidebar.multiselect(
    "Choose input columns for the model",
    options=predefined_input_columns,
    default=predefined_input_columns
)

if len(selected_columns) == 0:
    st.warning("Please select at least one input column to proceed.")
    st.stop()

# Sidebar: Select target variable
st.sidebar.header("Select Target to Predict")
target_var = st.sidebar.selectbox(
    "Target variable",
    options=['%AVANZ. FISICO ', '%SPESO']
)

# Button to start training
if st.sidebar.button("Run Model Training and Evaluation"):

    with st.spinner("Preprocessing data and training model..."):

        df_filtered = df.copy()

        # Common filtering
        df_filtered = df_filtered[df_filtered['OC_TOTALE_INDICATORI'] <= 4]
        df_filtered = df_filtered[~df_filtered['CUP_COD_NATURA'].astype(str).str.contains('8')]

        # Create combined columns
        df_filtered['CUP_NATURA_TIPOLOGIA'] = df_filtered['CUP_COD_NATURA'].astype(str) + "_" + df_filtered['CUP_COD_TIPOLOGIA'].astype(str)
        df_filtered['CUP_SETTORE_SOTTOSETTORE_CATEGORIA'] = (
            df_filtered['CUP_COD_SETTORE'].astype(str) + "_" +
            df_filtered['CUP_COD_SOTTOSETTORE'].astype(str) + "_" +
            df_filtered['CUP_COD_CATEGORIA'].astype(str)
        )

        # Prepare target-specific filtering and target class creation
        if target_var == '%AVANZ. FISICO ':
            df_filtered = df_filtered[(df_filtered['%AVANZ. FISICO '] >= 0) & (df_filtered['%AVANZ. FISICO '] <= 0.99)]
            df_filtered = df_filtered[df_filtered['%AVANZ. FISICO '].notna()]
            df_filtered = df_filtered[df_filtered['%AVANZ. FISICO '] != 'n/d']
            df_filtered['%AVANZ. FISICO '] = pd.to_numeric(df_filtered['%AVANZ. FISICO '], errors='coerce')
            df_filtered = df_filtered[df_filtered['%AVANZ. FISICO '].notna()]

            scaler = MinMaxScaler()
            df_filtered['target_normalized'] = scaler.fit_transform(df_filtered['%AVANZ. FISICO '].values.reshape(-1, 1))

            def initial_segment_target(val):
                if val <= 0.5:
                    return "avanzamento fisico sotto 50%"
                elif val <= 0.95:
                    return "avanzamento fisico 51-95%"
                else:
                    return "avanzamento fisico 96-100%"

            target_display_name = "AVANZAMENTO FISICO"

        else:  # target_var == '%SPESO'
            df_filtered = df_filtered[(df_filtered['%SPESO'] >= 0) & (df_filtered['%SPESO'] <= 0.99)]
            df_filtered = df_filtered[df_filtered['%SPESO'].notna()]
            df_filtered = df_filtered[df_filtered['%SPESO'] != 'n/d']
            df_filtered['%SPESO'] = pd.to_numeric(df_filtered['%SPESO'], errors='coerce')
            df_filtered = df_filtered[df_filtered['%SPESO'].notna()]

            scaler = MinMaxScaler()
            df_filtered['target_normalized'] = scaler.fit_transform(df_filtered['%SPESO'].values.reshape(-1, 1))

            def initial_segment_target(val):
                if val <= 0.60:
                    return "avanzamento finanziario sotto 60%"
                elif val <= 0.95:
                    return "avanzamento finanziario 61-95%"
                else:
                    return "avanzamento finanziario 96-100%"

            target_display_name = "AVANZAMENTO FINANZIARIO"

        df_filtered['target_class'] = df_filtered['target_normalized'].apply(initial_segment_target)

        le_target = LabelEncoder()
        df_filtered['target_class_enc'] = le_target.fit_transform(df_filtered['target_class'])

        # Prepare X and y
        X = df_filtered[selected_columns]
        y = df_filtered['target_class_enc']

        # Columns to encode (intersection with selected columns)
        columns_to_encode_all = [
            "OC_MACROAREA",
            "PROVINCIA_BENEFICIARIO",
            "OC_DESCR_FORMA_GIU_BENEFICIARIO",
            "OC_TEMA_SINTETICO",
            "OC_COD_TIPO_AIUTO",
            "COD_OB_TEMATICO",
            "COD_PRIORITA_INVEST",
            "Cluster_SLL_REGIONE_DIMENSIONE",
            "COD_TIPO_PROCED_ATTIVAZIONE",
            "CLASSIF_AREA_INTERNA",
            "CUP_NATURA_TIPOLOGIA",
            "CUP_SETTORE_SOTTOSETTORE_CATEGORIA"
        ]
        columns_to_encode = [col for col in columns_to_encode_all if col in selected_columns]

        oof_preds = np.zeros((len(X), len(le_target.classes_)))
        oof_true = np.array(y)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        importances_list = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            st.write(f"Training fold {fold}...")
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

            target_encoder = ce.TargetEncoder(cols=columns_to_encode)
            X_train_enc = target_encoder.fit_transform(X_train, y_train)
            X_val_enc = target_encoder.transform(X_val)

            class_majority = le_target.transform([le_target.classes_[-1]])[0]  # last class as majority

            mask_majority = y_train == class_majority
            mask_minority = y_train != class_majority

            X_majority = X_train_enc[mask_majority]
            y_majority = y_train[mask_majority]

            X_minority = X_train_enc[mask_minority]
            y_minority = y_train[mask_minority]

            count_majority = len(y_majority)

            minority_classes = y_minority.unique()
            sampling_strategy = {cls: count_majority for cls in minority_classes if sum(y_minority == cls) < count_majority}

            if len(y_minority) > 0 and len(sampling_strategy) > 0:
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_minority_res, y_minority_res = smote.fit_resample(X_minority, y_minority)
            else:
                X_minority_res, y_minority_res = X_minority, y_minority

            X_combined = pd.concat([
                pd.DataFrame(X_majority, columns=X_train_enc.columns),
                pd.DataFrame(X_minority_res, columns=X_train_enc.columns)
            ], ignore_index=True)
            y_combined = pd.concat([
                pd.Series(y_majority),
                pd.Series(y_minority_res)
            ], ignore_index=True)

            target_majority_size = int(count_majority * 0.9)
            rus = RandomUnderSampler(sampling_strategy={class_majority: target_majority_size}, random_state=42)
            X_train_balanced, y_train_balanced = rus.fit_resample(X_combined, y_combined)

            clf = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(le_target.classes_),
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            clf.fit(X_train_balanced, y_train_balanced)

            feat_imp_fold = pd.Series(clf.feature_importances_, index=selected_columns)
            importances_list.append(feat_imp_fold)

            y_val_proba = clf.predict_proba(X_val_enc)
            oof_preds[val_idx] = y_val_proba

        avg_importance = pd.concat(importances_list, axis=1).mean(axis=1)
        avg_importance = avg_importance.sort_values(ascending=False).head(20)

        y_pred = np.argmax(oof_preds, axis=1)

        predicted_classes = le_target.inverse_transform(y_pred)

        # Map predicted classes to numeric values for display (adjust as needed)
        if target_var == '%AVANZ. FISICO ':
            def map_pred(val):
                if val == "avanzamento fisico sotto 50%":
                    return 1
                elif val == "avanzamento fisico 51-95%":
                    return 0.5
                else:
                    return 0
        else:
            def map_pred(val):
                if val == "avanzamento finanziario sotto 60%":
                    return 1
                elif val == "avanzamento finanziario 61-95%":
                    return 0.5
                else:
                    return 0

        mapped_pred = np.array([map_pred(c) for c in predicted_classes])

        # Display feature importance first
        st.subheader("Top 20 Average Feature Importance")
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        sns.barplot(x=avg_importance.values, y=avg_importance.index, ax=ax_fi)
        ax_fi.set_xlabel("Average Importance")
        ax_fi.set_ylabel("Feature")
        ax_fi.set_title("Average Feature Importance Across Folds (Top 20)")
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
            """, unsafe_allow_html=True)
        st.pyplot(fig_fi)
        st.markdown("</div>", unsafe_allow_html=True)

        # Display accuracy and macro F1-score
        st.subheader(f"Classification Metrics for target: {target_display_name} (Cross-Validation)")

        acc = accuracy_score(oof_true, y_pred)
        f1_macro = f1_score(oof_true, y_pred, average='macro')
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Macro F1-score:** {f1_macro:.4f}")

        # Display classification report as a table
        report_dict = classification_report(oof_true, y_pred, target_names=le_target.classes_, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df[['precision', 'recall', 'f1-score', 'support']] = report_df[['precision', 'recall', 'f1-score', 'support']].apply(pd.to_numeric, errors='coerce')
        report_df = report_df.round(3)

        st.subheader("Classification Report")
        st.dataframe(report_df)

        # Display confusion matrix last
        cm = confusion_matrix(oof_true, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title('Confusion Matrix (CV)')
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
            """, unsafe_allow_html=True)
        st.pyplot(fig_cm)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Select input columns and target variable, then click 'Run Model Training and Evaluation' to start.")