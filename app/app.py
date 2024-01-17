import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from natixis.clustering_model.similar_bond_prediction import (
    get_nearest_rows_with_proximity_scores,
)
from natixis.deep_model.prediction import predict

# Initialize session state elements


def main():
    """
    The main function that runs the Streamlit application.

    This function is responsible for setting up the Streamlit application and defining the app's behavior.
    """
    # Set CSS
    with open("app/assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Set sidebar
    with st.sidebar:
        # Set Natixis logo
        st.image("app/assets/natixis.png")

        # Text markdown settings
        st.markdown(
            """
        <style>
        .sidebar-title {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 30px; /* Ajustez la hauteur en fonction de vos besoins */
            color: #5c1c74;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Title of the app
        st.markdown("---")
        st.markdown(
            """
        <div style="text-align: center;">
            <h1 class="sidebar-title" style="margin-bottom: 10px;">SECONDARY BOND TRADING TOOL</h1>
            <p class="subtitle" style="margin-bottom: 30px;"> An HEC-Natixis application</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # App logo
        st.image("app/assets/logo.png")

        # Reset button
        st.markdown("---")
        st.button("Reset session history")

    # Bond trading tool
    st.title("📈 :violet[Secondary Bond Trading Tool]")

    option = st.radio(
        "Select functionality:",
        ("Recommend clients for an ISIN code", "Recommend similar bonds"),
    )

    if option == "Recommend clients for an ISIN code":
        isin = st.text_input("Enter Natixis ISIN code:")
        size = st.text_input("Enter bond size (in M€):")
        b_side = st.radio(
            "Choose Natixis side (buyer or seller):",
            ["Buyer", "Seller"],
            horizontal=True,
        )
        n_clients = st.select_slider(
            "Select the number of clients you are looking for:", range(1, 11)
        )

        if st.button("Recommend clients"):
            recommended_clients, probabilities, viz_df = predict(
                isin, b_side, n_clients, size
            )
            # Get clients
            results_df = pd.DataFrame(
                {
                    "Client": recommended_clients,
                    "Investment probability": probabilities * 100,
                }
            )
            results_df.index += 1
            st.dataframe(results_df, use_container_width=True)
            # Plot visualization
            if not viz_df.empty:
                st.subheader(f"Latest RFQs for ISIN: {isin} and Natixis side: {b_side}")
                fig, ax = plt.subplots()
                ax.plot(
                    viz_df["Deal_Date"],
                    viz_df["company_short_name"],
                    marker="o",
                    linestyle="",
                    color="green",
                )
                plt.xticks(rotation=90, ha="right")
                ax.set_xlabel("Date")
                ax.set_ylabel("Company")
                st.pyplot(fig)
            else:
                st.warning(f"No positive signals found for ISIN: {isin}")

    if option == "Recommend similar bonds":
        isin = st.text_input("Enter Natixis ISIN code:")
        size = st.text_input("Enter bond size (in M€):")
        b_side = st.radio(
            "Choose Natixis side (buyer or seller):",
            ["Buyer", "Seller"],
            horizontal=True,
        )
        n_reco = st.select_slider(
            "Select the number of bonds you are looking for:", range(1, 11)
        )
        if st.button("Recommend bonds"):
            recommended_bonds, scores = get_nearest_rows_with_proximity_scores(
                isin, n_reco
            )
            results_df = pd.DataFrame(
                {"Bonds": recommended_bonds, "Proximity score": scores}
            )
            results_df.reset_index(drop=True, inplace=True)
            results_df.index += 1
            st.dataframe(results_df, use_container_width=True)


if __name__ == "__main__":
    main()
