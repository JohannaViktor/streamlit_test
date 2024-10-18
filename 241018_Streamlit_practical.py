#241018_Streamlit_practical
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#functions
def create_scatter_plot(data_df, year):

    custom_data=['headcount_ratio_upper_mid_income_povline']
    data = data_df.copy().dropna(axis=0)

    fig = px.scatter(
        data[data['year'] == year],
        x='GDP per capita',
        y='Life Expectancy (IHME)',
        hover_name=data.index,
        log_x=True,
        size='headcount_ratio_upper_mid_income_povline',
        color=data.index,
        title=f'Life Expectancy vs GDP per capita ({year})',
        labels={'GDP per capita': 'GDP per capita (log scale)',
                'Life Expectancy (IHME)': 'Life Expectancy (years)',
                'headcount_ratio_upper_mid_income_povline': 'Poverty (%)'},
        custom_data=custom_data
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Life Expectancy: %{y:.1f} years<br>GDP per capita: $%{x:,.0f}<br>Poverty: %{customdata[0]:,.2f}% <extra></extra>"
    )

    return fig

def get_life_prediction(data_df):
    # ALE Prediction model
    st.header("Predict Life Expectancy")
    st.write("This model uses timestamp, GDP per capita, and poverty rates to predict Life Expectancy.")
    feature_names = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    X = data_df[feature_names].values
    y = data_df['Life Expectancy (IHME)'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #model = LinearRegression()
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)

    st.write(f"#### Model Performance:")
    st.write(f"Mean Squared Error: {mse:.4f}")
    #st.write(f"R-squared Score: {r2:.4f}")

    gdp_input = st.number_input(f"### Enter the GDP per capita (dollars):", min_value=0.0, max_value=100000.0, value=12000.0)
    poverty_input = st.number_input("Enter the Poverty Rate (%):", min_value=0.0, max_value=100.0, value=25.0)
    year_input = st.number_input("Year of prediction:", min_value=1920, max_value=2016, value=2016)

    if st.button("Predict"):
        prediction = model.predict([[gdp_input, poverty_input, year_input]])
        st.write(f"Predicted Life Expectancy:: {prediction[0]:.3f} Years")




st.set_page_config(layout="wide")
st.header("Worldwide Analysis of Quality of Life and Economic Factors")

st.write("""
    This app enables you to explore the relationships between poverty, 
            life expectancy, and GDP across various countries and years. 
            Use the panels to select options and interact with the data.
    """)
#data
df = pd.read_csv("gdp_lifeexp_pov.csv", index_col=0)#change file path
df['year'] = df['year'].astype(int)
#tabs
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])
with tab1:
    st.header("Global Overview - Key Statistics")
             
    year = st.slider("Select Year for Visualization", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=2016)


    frozen_df = df[df['year']==year]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Global Average Life Expectancy", f"{frozen_df['Life Expectancy (IHME)'].mean():.1f} years")
    with col2:
        st.metric("Global Median GDP per capita", f"${frozen_df['GDP per capita'].median():,.0f}")
    with col3:
        st.metric("Global Poverty Average", f"{frozen_df['headcount_ratio_upper_mid_income_povline'].mean():,.0f}%")
    with col4:
        st.metric("Number of Countries", f"{frozen_df.index.nunique()}")

    fig = create_scatter_plot(frozen_df, year)
    st.plotly_chart(fig, use_container_width=True)
    
    get_life_prediction(df)

with tab3:
        st.header("Data Source")
        
        #year range filter
        year_range = st.slider("Select Year Range", 
                                        min_value=int(df['year'].min()), 
                                        max_value=int(df['year'].max()),
                                        value=(int(df['year'].min()), int(df['year'].max())))

        # Filter data based on selections
        filtered_df = df[(df['year'].between(year_range[0], year_range[1]))]

        
        #country filter
        countries = st.multiselect(
            "Choose countries", list(df.index.unique()), ["China", "Argentina"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = filtered_df.loc[countries]
            #show data
            st.write(data)
        
         # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='filtered_development_data.csv',
            mime='text/csv',
        )
