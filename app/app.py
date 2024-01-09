import streamlit as st

# Initialize session state elements

def main():
    """
    The main function that runs the Streamlit application.

    This function is responsible for setting up the Streamlit application and defining the app's behavior.

    Returns:
        None
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
        unsafe_allow_html=True)

        # Title of the app
        st.markdown("---")
        st.markdown(
            """
        <div style="text-align: center;">
            <h1 class="sidebar-title" style="margin-bottom: 10px;">SECONDARY BOND TRADING TOOL</h1>
            <p class="subtitle" style="margin-bottom: 30px;"> An HEC-Natixis application</p>
        </div>
        """,
        unsafe_allow_html=True)

        # App logo
        st.image("app/assets/logo.png")

        # Reset button
        st.markdown("---")
        st.button("Reset session history")

    # Bond trading tool
    st.title("📈 :violet[Secondary Bond Trading Tool]")

    option = st.radio("Select functionality:", ("Recommend clients for an ISIN code", "Recommend bonds for a client"))

    if option == "Recommend clients for an ISIN code":
        isin_code = st.text_input("Enter Natixis ISIN code:")
        if st.button("Recommend clients"):
            recommended_clients = None
            st.write("Recommended clients:", recommended_clients)
    elif option == "Recommend bonds for a client":
        client_name = st.text_input("Enter client name:")
        if st.button("Recommend bonds"):
            recommended_bonds = None
            st.write("Recommended bonds:", recommended_bonds)

if __name__ == "__main__":
    main()