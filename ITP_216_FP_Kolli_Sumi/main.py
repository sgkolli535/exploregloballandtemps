"""
Sumi Kolli
Fall 2021
Section 32081
Final Project

"""
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

"""

    routes to home page for user to enter information they want to visualize
   params: none
   returns: render template home.html

"""


@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("home.html")


"""

    routes to page with graph if user enters valid city, otherwise routes to home page again
   params: none
   returns: render template chart.html or home.html

"""


@app.route("/client", methods=["POST", "GET"])
def client():
    # upon form submission
    if request.method == "POST":
        # checks if city exists in database
        if check_city(request.form["city"]):
            return redirect(url_for("chart", city=request.form["city"], dates=request.form["dates"]))
        # otherwise routes to homepage
        else:
            return redirect(url_for("home"))


"""

    routes to page with graph on it and has button to route to page to see graph of predictions too
   params: city, dates
   returns: graph.html or redirect to chart_predicted if button is clicked

"""


@app.route("/chart/<city>/<dates>", methods=["POST", "GET"])
def chart(city, dates):
    # upon form submission
    if request.method == "POST":
        # redirect to page with graph with predictions
        return redirect(url_for("chart_predicted", city=city))
    return render_template("graph.html", city=city, dates=dates)


"""

    sends image of generated graph based on user input to graph.html
   params: city, dates
   returns: image file of generated graph

"""


@app.route("/fig/<city>/<dates>")
def fig(city, dates):
    # call create_chart function with user input information
    figure = create_chart(city, dates)
    img = BytesIO()
    # save figure
    figure.savefig(img)
    img.seek(0)
    # send figure as image/png
    return send_file(img, mimetype='image/png')


"""

    routes to page with graph of temperature predictions based on MLP 
   params: city
   returns: render template predicted.html

"""


@app.route("/chart_predicted/<city>")
def chart_predicted(city):
    return render_template("predicted.html", city=city)


"""

    sends image of generated graph based on user input to predicted.html
   params: city
   returns: image file of generated graph

"""


@app.route("/fig_predicted/<city>")
def fig_predicted(city):
    # call ml_chart function with user input information
    figure = ml_chart(city)
    img = BytesIO()
    # save figure
    figure.savefig(img)
    img.seek(0)
    # send figure as as image/png
    return send_file(img, mimetype='image/png')


"""

    checks if user's inputted city is in database
   params: city
   returns: True if in database and false otherwise

"""


def check_city(city):
    # generate list of all unique cities in database
    df = pd.read_csv("GlobalLandTemperaturesByMajorCityupdated.csv")
    cities = list(df["City"].unique())
    if city in cities:
        return True
    else:
        return False


"""

    creates chart with user input
   params: city, dates
   returns: figure of chart

"""


def create_chart(city, dates):
    # read csv and save as dataframe
    df = pd.read_csv("GlobalLandTemperaturesByMajorCityupdated.csv")
    # save rows of specific city
    city_df = df[df["City"] == city]
    # depending on user's chosen dates, further specify the rows to visualize
    if dates == "1900 - 1910":
        city_df = city_df.iloc[0:132]
    elif dates == "1911 - 1920":
        city_df = city_df.iloc[132:252]
    elif dates == "1921 - 1930":
        city_df = city_df.iloc[252:372]
    elif dates == "1931 - 1930":
        city_df = city_df.iloc[372:492]
    elif dates == "1941 - 1950":
        city_df = city_df.iloc[492:612]
    elif dates == "1951 - 1960":
        city_df = city_df.iloc[612:732]
    elif dates == "1961 - 1970":
        city_df = city_df.iloc[732:852]
    elif dates == "1971 - 1980":
        city_df = city_df.iloc[852:972]
    elif dates == "1981 - 1990":
        city_df = city_df.iloc[972:1092]
    elif dates == "1991 - 2000":
        city_df = city_df.iloc[1092:1212]
    elif dates == "2001 - 2013":
        city_df = city_df.iloc[1212:]
    # generate plot of average temperature over dates
    fige, ax = plt.subplots(1, 1)
    ax.plot(city_df["dt"], city_df["AverageTemperature"], color="blue", label="Average Temperature")
    # add legends, titles, and labels
    ax.legend()
    ax.set(title=city)
    ax.set_xlabel('Date (Monthly)')
    ax.set(ylabel='Average Temperature')
    # set ticks on x-axis to appear every 12 data points and rotate 45 for readability
    ax.set_xticks(city_df["dt"][::12])
    ax.set_xticklabels(city_df["dt"][::12], rotation=45)
    # make a twin plot on same graph to also graph average temperature uncertainty over dates
    ax2 = ax.twinx()
    ax2.plot(city_df["dt"], city_df["AverageTemperatureUncertainty"], color="orange", label="Average Temperature "
                                                                                            "Uncertainty")
    # add legends and labels
    ax2.legend()
    ax2.set(ylabel='Average Temperature Uncertainty')
    ax2.set(xlabel='Date (Monthly)')
    # set ticks on x-axis to appear every 12 data points and rotate 45 for readability
    ax2.set_xticks(city_df["dt"][::12])
    ax2.set_xticklabels(city_df["dt"][::12], rotation=45)
    # add space to bottom to reveal x-axis label
    plt.gcf().subplots_adjust(bottom=0.15)
    return fige


"""

    creates MLP chart with user input of city
   params: city
   returns: figure of chart

"""


def ml_chart(city):
    # read csv and save as dataframe
    df = pd.read_csv("GlobalLandTemperaturesByMajorCityupdated.csv")
    # save rows of specific city
    df = df[df["City"] == city]
    # change dt values from string, to datetime object, back to string without slashes, and finally to int
    df['dt'] = pd.to_datetime(df['dt'], format='%m/%d/%y')
    df['dt'] = df['dt'].dt.strftime('%m%d%y')
    df['dt'] = df['dt'].astype(int)
    # change average temperature and average temperature values to int
    df['AverageTemperature'] = df['AverageTemperature'].astype(int)
    df['AverageTemperatureUncertainty'] = df['AverageTemperatureUncertainty'].astype(int)
    # set x and y sets to predict average temperature from dates and average temperature uncertainty
    x_temp = df[['dt', 'AverageTemperatureUncertainty']]
    y_temp = df['AverageTemperature']
    # use test train split
    x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp)
    # scale x_train and x_test sets
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    # call mlp and fit the model to data
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(x_train, y_train)
    # generate predictions from x_test set
    predictions = mlp.predict(x_test)
    # make histogram of predictions to find most likely average temperatures in city
    fige, ax = plt.subplots(1, 1)
    ax.hist(x=predictions, label='Average Temperature Counts')
    # add legend, title, and labels
    ax.legend()
    ax.set(title=city)
    ax.set_xlabel('Average Temperature')
    ax.set(ylabel='Count')
    # add space to bottom to reveal x-axis label
    plt.gcf().subplots_adjust(bottom=0.15)
    return fige


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
