
import streamlit as st
import pandas as pd
import numpy as np
import time as time
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





df = pd.read_csv("/Users/happy/Desktop/HDAG project/Affordable Housing by Town.zip", encoding='latin1')




st.title("HDAG Homework - Check Your Housing Affordability!")

st.subheader("Give us your info and we'll tell you if you can afford a house")

st.header("Enter Your Info")
name = st.text_input("What shall we call you?")

town = st.text_input("Which town do you live in?")


if (name, town) != ("", ""): 
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

if town in df['Town'].values:
    st.write(f"Slay, {name}. We have some useful info about {town}!")
    st.header("Diving into more details")
    data1 = st.selectbox("What would you like to know about your town?", options=["2010 Census Units", 'Percent Affordable', 'Government Assisted', 'Tenant Rental Assistance', 'Deed Restricted Units'])
    progress2 = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress2.progress(i + 1)
    st.write(f"Great, here's {data1} over the years:")
    st.line_chart(df[df['Town'] == town][['Year', data1]].set_index('Year'))
    if st.button("Show Regression Analysis Of Your Town's Percent Affordable"):
        X_lr = df[df['Town'] == town][['Year']]
        y_lr = df[df['Town'] == town][['Percent Affordable']]

        # Split into training and test sets
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.3, random_state=2024)

        # Fit the linear regression model
        linreg = LinearRegression().fit(X_train_lr, y_train_lr)

        # Evaluate the model
        train_r2 = linreg.score(X_train_lr, y_train_lr)
        test_r2 = linreg.score(X_test_lr, y_test_lr)
        train_mse = mean_squared_error(y_train_lr, linreg.predict(X_train_lr))
        test_mse = mean_squared_error(y_test_lr, linreg.predict(X_test_lr))

        # Plot the regression line
        plt.figure(figsize=(8,5))
        plt.scatter(X_train_lr, y_train_lr, label='Train', alpha=0.5)
        plt.scatter(X_test_lr, y_test_lr, label='Test', alpha=0.5)
        x_lin = np.linspace(X_lr.min(), X_lr.max(), 100).reshape(-1,1)
        plt.plot(x_lin, linreg.predict(x_lin), color='red', label='Regression Line')
        plt.xlabel('Year')
        plt.ylabel('Total Assisted Units')
        plt.title('Single Variable Linear Regression')
        plt.legend()
        plt.show()
        st.pyplot(plt)
    
    
elif town == "":
    st.write("Please enter a town name.")

elif town not in df['Town'].values:
    st.write(f"Sorry {name}... we don't recognize {town} in our data. Maybe try capitalizing it?")


