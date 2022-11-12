import io

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


if __name__ == "__main__":

    # Wetterdaten
    API_LINK_WETTER = 'https://daten.sg.ch/api/records/1.0/search/?dataset=wetterdaten-klimamessnetz-kanton-stgallen-tageswerte&q=date%3A%5B2021-04-19+TO+2021-08-31%5D&rows=150&sort=-date&facet=station_location&facet=date&refine.station_location=St.+Gallen&refine.date=2021&exclude.station_location=S%C3%A4ntis&exclude.station_location=Bad+Ragaz'
    req_wetter = requests.get(API_LINK_WETTER)
    raw_data_wetter = req_wetter.json()
    sonnenscheindauer_tagessumme = [entry["fields"]["sre000d0"] / 60 if entry["fields"]["sre000d0"] > 0 else 1 for entry in raw_data_wetter["records"]]
    globalstrahlung_tagesmittel = [entry["fields"]["gre000d0"] if entry["fields"]["gre000d0"] > 0 else 1 for entry in raw_data_wetter["records"]]
    energiedichte_sonnenstrahlung = [a * b for a, b in zip(sonnenscheindauer_tagessumme, globalstrahlung_tagesmittel)]
    print('Globalstahlung Tagesmittel: ', globalstrahlung_tagesmittel)
    print('Sonnenscheindauer Tagessumme: ', sonnenscheindauer_tagessumme)
    print('Energiedichte: ', energiedichte_sonnenstrahlung)

    # Solaranlagen Daten
    data_frame = pd.read_csv('stromproduktion-der-solaranlagen-der-stgaller-stadtwerke2.csv', delimiter=';')
    data_frame = data_frame.loc[:, ['DateTime (Local Time)', 'Name', 'Additional Energy Export', 'Fläche in m2']]
    data_frame['DateTime (Local Time)'] = data_frame['DateTime (Local Time)'].str.split('T').str[0]
    data_frame = data_frame.dropna()
    data_frame = data_frame.sort_values(by=['DateTime (Local Time)'])
    energiedichte_solaranlagen = []
    old_date = data_frame['DateTime (Local Time)'][0]
    my_sum = 0
    for index, row in data_frame.iterrows():
        if row['DateTime (Local Time)'] == old_date:
            my_sum += row['Additional Energy Export'] / row['Fläche in m2']
        else:
            old_date = row['DateTime (Local Time)']
            energiedichte_solaranlagen.append(my_sum)
            my_sum = 0
    #leistung.append(my_sum)
    print('Energiedichte Solaranlagen: ', energiedichte_solaranlagen)

    # Lineare Regression
    model = LinearRegression().fit(np.array(energiedichte_sonnenstrahlung).reshape([-1, 1]), energiedichte_solaranlagen)
    b0 = model.intercept_
    b1 = model.coef_
    r2 = model.score(np.array(energiedichte_sonnenstrahlung).reshape([-1, 1]), energiedichte_solaranlagen)
    print('R^2-Wert: ', r2)

    # Polinomische Regression
    n = 4
    X = np.array(energiedichte_sonnenstrahlung).reshape([-1, 1])
    poly = PolynomialFeatures(degree=n)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, energiedichte_solaranlagen)
    lin2 = LinearRegression()
    lin2.fit(X_poly, energiedichte_solaranlagen)

    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    xi = np.linspace(np.min(energiedichte_sonnenstrahlung), np.max(energiedichte_sonnenstrahlung), 10)
    line = xi * b1 + b0
    plt.plot(energiedichte_sonnenstrahlung, energiedichte_solaranlagen, 'o')
    plt.plot(xi, line, label='lineare Regression')
    plt.plot(xi, lin2.predict(poly.fit_transform(np.array(xi).reshape([-1, 1]))), label=f'polinomische Regression (n={n})')
    plt.xlabel("Energiedichte der Sonnenstahlung [$Wh/m^2$]")
    plt.ylabel("Energiedichte aus den Solaranlagen [$Wh/m^2$]")
    plt.legend()
    plt.grid()
    plt.show()

    # Predictor
    print('\nSolarenergie Vorhersage')
    print('--------------------------------')
    sonnenscheindauer_input = float(input('Bitte geben Sie die Sonnenscheindauer [h] für einen Tag ein: '))
    globalstrahlung_input = float(input('Bitte geben Sie die durchschnittliche Globalstrahlung [W/m^2] für einen Tag ein: '))
    energiedichte_output = float(sonnenscheindauer_input*globalstrahlung_input*b1 + b0)
    print(f'Geschätzte Energiedichte der Solaranlagen der St. Galler Stadtwerke: {0.001*energiedichte_output:.3f} [kWh/m^2]')
