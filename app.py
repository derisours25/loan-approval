
# Smart Loan Approval DSS
# RF + TOPSIS + Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Smart Loan DSS", layout="wide")

st.title("Smart Loan Approval DSS")
st.write("Random Forest + TOPSIS Recommendation Dashboard")

uploaded_file = st.file_uploader("Upload loan_approval.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df["loan_status"] = df["loan_status"].astype(str).str.strip()

    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    df["loan_income_ratio"] = df["loan_amount"] / df["income_annum"]
    df["asset_loan_ratio"] = df["total_assets"] / df["loan_amount"]
    df["high_risk"] = np.where(df["cibil_score"] < 600, 1, 0)


    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive", "Analytics", "Smart Prediction", "TOPSIS"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applicants", len(df))
        c2.metric("Approved", (df["loan_status"] == "Approved").sum())
        c3.metric("Rejected", (df["loan_status"] == "Rejected").sum())
        c4.metric("Avg CIBIL", round(df["cibil_score"].mean(), 0))

    with tab2:
        col1, col2 = st.columns(2)

        cibil_analysis = (
            df.groupby("cibil_score")
            .size()
            .reset_index(name="total_applicants")
            .sort_values("cibil_score")
        )

        fig1 = px.line(
            cibil_analysis,
            x="cibil_score",
            y="total_applicants",
            markers=True,
            title="Applicant Distribution by CIBIL Score",
        )
        col1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.pie(
            df,
            names="loan_status",
            color="loan_status",
            color_discrete_map={
                "Approved": "#2563EB",
                "Rejected": "#DC2626",
            },
            title="Approval Distribution",
        )
        col2.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(
            df,
            x="loan_status",
            y="loan_amount",
            color="loan_status",
            color_discrete_map={
                "Approved": "#2563EB",
                "Rejected": "#DC2626",
            },
            title="Loan Amount Distribution",
        )
        st.plotly_chart(fig3, use_container_width=True)

    ml_df = df.copy()
    le = LabelEncoder()
    for col in ["education", "self_employed", "loan_status"]:
        ml_df[col] = le.fit_transform(ml_df[col])

    X = ml_df.drop(
    columns=[
        "loan_status",
        "loan_id"
    ],
    errors="ignore"
)
    y = ml_df["loan_status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    with tab3:

        st.header("Smart Loan Approval Prediction")

        st.markdown("""
        Masukkan profil pemohon pinjaman untuk memperoleh
        prediksi persetujuan pinjaman menggunakan Random Forest.
        """)

        col1, col2 = st.columns(2)

        with col1:

            dep = st.number_input(
                "Dependents",
                min_value=0,
                max_value=10,
                value=2
            )

            edu = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"]
            )

            self_emp = st.selectbox(
                "Self Employed",
                ["Yes", "No"]
            )

            income = st.number_input(
                "Annual Income",
                min_value=0,
                value=5000000
            )

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0,
                value=1000000
            )

            loan_term = st.number_input(
                "Loan Term (Months)",
                min_value=1,
                value=12
            )

        with col2:

            cibil = st.slider(
                "CIBIL Score",
                300,
                900,
                700
            )

            res = st.number_input(
                "Residential Asset",
                min_value=0,
                value=1000000
            )

            com = st.number_input(
                "Commercial Asset",
                min_value=0,
                value=500000
            )

            lux = st.number_input(
                "Luxury Asset",
                min_value=0,
                value=500000
            )

            bank = st.number_input(
                "Bank Asset",
                min_value=0,
                value=500000
            )

        if st.button("Predict Applicant", use_container_width=True):

            total_assets = res + com + lux + bank

            input_df = pd.DataFrame({

                "no_of_dependents": [dep],
                "education": [1 if edu == "Graduate" else 0],
                "self_employed": [1 if self_emp == "Yes" else 0],
                "income_annum": [income],
                "loan_amount": [loan_amount],
                "loan_term": [loan_term],
                "cibil_score": [cibil],
                "residential_assets_value": [res],
                "commercial_assets_value": [com],
                "luxury_assets_value": [lux],
                "bank_asset_value": [bank],
                "total_assets": [total_assets],
                "loan_income_ratio": [loan_amount / max(income, 1)],
                "asset_loan_ratio": [total_assets / max(loan_amount, 1)],
                "high_risk": [1 if cibil < 600 else 0]

            })

            prediction = model.predict(
                scaler.transform(input_df)
            )[0]

            probabilities = model.predict_proba(
                scaler.transform(input_df)
            )[0]

            approval_probability = probabilities[0] * 100

            prediction_label = (
                "Approved"
                if approval_probability >= 50
                else "Rejected"
            )

            if approval_probability >= 80:
                risk = "Low Risk"
                risk_icon = "🟢"

            elif approval_probability >= 60:
                risk = "Medium Risk"
                risk_icon = "🟡"

            else:
                risk = "High Risk"
                risk_icon = "🔴"

            st.divider()

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Prediction",
                prediction_label
            )

            k2.metric(
                "Approval Probability",
                f"{approval_probability:.2f}%"
            )

            k3.metric(
                "Risk Category",
                f"{risk_icon} {risk}"
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=approval_probability,
                    title={
                        "text": "Approval Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#EF4444"
                            },
                            {
                                "range": [50, 80],
                                "color": "#FACC15"
                            },
                            {
                                "range": [80, 100],
                                "color": "#22C55E"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.progress(
                int(approval_probability)
            )

            st.caption(
                f"Approval Confidence: {approval_probability:.2f}%"
            )

            st.subheader("Decision Explanation")

            reasons = []

            if cibil >= 750:
                reasons.append(
                    "Excellent credit score"
                )

            elif cibil >= 650:
                reasons.append(
                    "Acceptable credit score"
                )

            else:
                reasons.append(
                    "Low credit score increases risk"
                )

            if income >= loan_amount * 3:
                reasons.append(
                    "Strong repayment capability"
                )

            if total_assets >= loan_amount:
                reasons.append(
                    "Assets sufficiently cover the requested loan"
                )

            if approval_probability < 50:
                reasons.append(
                    "Overall financial profile requires review"
                )

            for reason in reasons:
                st.success(reason)

            st.subheader("Risk Assessment")

            if risk == "Low Risk":

                st.success("""
                Applicant demonstrates strong financial stability
                and low probability of default.
                """)

            elif risk == "Medium Risk":

                st.warning("""
                Applicant shows moderate risk.
                Additional verification is recommended.
                """)

            else:

                st.error("""
                Applicant has elevated credit risk.
                Loan approval should be reviewed carefully.
                """)


       

        st.caption(
            f"Approval Confidence: {approval_probability:.2f}%"
            )

        with tab4:

            st.header("TOPSIS Recommendation Engine")

            st.markdown("""
            TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
            digunakan untuk menentukan kandidat peminjam terbaik berdasarkan
            kombinasi beberapa kriteria finansial.
            """)

            criteria = [
                "income_annum",
                "cibil_score",
                "total_assets",
                "asset_loan_ratio",
                "loan_income_ratio"
            ]

            decision_matrix = df[criteria].values

            normalized = (
                decision_matrix /
                np.sqrt((decision_matrix**2).sum(axis=0))
            )

            weights = np.array([
                0.25,
                0.30,
                0.20,
                0.15,
                0.10
            ])

            weighted = normalized * weights

            ideal_positive = np.max(weighted, axis=0)
            ideal_negative = np.min(weighted, axis=0)

            d_pos = np.sqrt(
                ((weighted - ideal_positive)**2).sum(axis=1)
            )

            d_neg = np.sqrt(
                ((weighted - ideal_negative)**2).sum(axis=1)
            )

            score = d_neg / (d_pos + d_neg)

            ranking = df.copy()

            ranking["TOPSIS_Score"] = score

            ranking["Recommendation"] = np.where(
                ranking["TOPSIS_Score"] >= 0.80,
                "Highly Recommended",
                np.where(
                    ranking["TOPSIS_Score"] >= 0.60,
                    "Recommended",
                    "Needs Review"
                )
            )

            ranking = ranking.sort_values(
                "TOPSIS_Score",
                ascending=False
            ).reset_index(drop=True)

            ranking.insert(
                0,
                "Rank",
                range(1, len(ranking)+1)
            )

            # =========================
            # BEST CANDIDATE CARD
            # =========================

            best = ranking.iloc[0]

            st.success(
                f"""
                Best Candidate Recommendation

                Rank : #{best['Rank']}

                Income : {best['income_annum']:,.0f}

                CIBIL Score : {best['cibil_score']}

                TOPSIS Score : {best['TOPSIS_Score']:.3f}

                Recommendation :
                {best['Recommendation']}
                """
            )

            # =========================
            # SCORE GUIDE
            # =========================

            st.info("""
            TOPSIS Score Interpretation

            0.80 - 1.00
            Highly Recommended

            0.60 - 0.79
            Recommended

            Below 0.60
            Needs Review
            """)

            # =========================
            # TOP 10 TABLE
            # =========================

            st.subheader("Top 10 Ranked Applicants")

            st.dataframe(
                ranking[
                    [
                        "Rank",
                        "income_annum",
                        "loan_amount",
                        "cibil_score",
                        "TOPSIS_Score",
                        "Recommendation"
                    ]
                ].head(10),
                use_container_width=True
            )

            # =========================
            # BAR CHART
            # =========================

            fig = px.bar(
                ranking.head(10),
                x="TOPSIS_Score",
                y="Rank",
                orientation="h",
                color="TOPSIS_Score",
                title="Top 10 TOPSIS Ranking"
            )

            fig.update_layout(
                yaxis=dict(
                    autorange="reversed"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =========================
            # PIE CHART
            # =========================

            fig2 = px.pie(
                ranking.head(20),
                names="Recommendation",
                title="Recommendation Distribution"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            # =========================
            # DOWNLOAD
            # =========================

            csv = ranking.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Full Ranking CSV",
                csv,
                "ranking.csv",
                "text/csv"
            )
