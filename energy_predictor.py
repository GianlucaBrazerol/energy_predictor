import io

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    # Wetterdaten
    API_LINK_WETTER = 'https://daten.sg.ch/api/records/1.0/search/?dataset=wetterdaten-klimamessnetz-kanton-stgallen-tageswerte&q=date%3A%5B2021-04-19+TO+2021-08-31%5D&rows=150&sort=-date&facet=station_location&facet=date&refine.station_location=St.+Gallen&refine.date=2021&exclude.station_location=S%C3%A4ntis&exclude.station_location=Bad+Ragaz'
    req_wetter = requests.get(API_LINK_WETTER)
    raw_data_wetter = req_wetter.json()
    sonnenscheindauer_tagessumme = [entry["fields"]["sre000d0"] / 60 if entry["fields"]["sre000d0"] > 0 else 1 for entry in raw_data_wetter["records"]]
    globalstrahlung_tagesmittel = [entry["fields"]["gre000d0"] if entry["fields"]["gre000d0"] > 0 else 1 for entry in raw_data_wetter["records"]]
    energiedichte = [a*b for a, b in zip(sonnenscheindauer_tagessumme, globalstrahlung_tagesmittel)]
    print('Globalstahlung Tagesmittel: ', globalstrahlung_tagesmittel)
    print('Sonnenscheindauer Tagessumme: ', sonnenscheindauer_tagessumme)
    print('Energiedichte: ', energiedichte)

    # Solaranlagen Daten
    data_frame = pd.read_csv('stromproduktion-der-solaranlagen-der-stgaller-stadtwerke2.csv', delimiter=';')
    data_frame = data_frame.loc[:, ['DateTime (Local Time)', 'Name', 'Additional Energy Export']]
    data_frame['DateTime (Local Time)'] = data_frame['DateTime (Local Time)'].str.split('T').str[0]
    data_frame = data_frame.dropna()
    data_frame = data_frame.sort_values(by=['DateTime (Local Time)'])
    leistung = []
    old_date = data_frame['DateTime (Local Time)'][0]
    my_sum = 0
    for index, row in data_frame.iterrows():
        if row['DateTime (Local Time)'] == old_date:
            my_sum += row['Additional Energy Export']
        else:
            old_date = row['DateTime (Local Time)']
            leistung.append(my_sum)
            my_sum = 0
    #leistung.append(my_sum)
    print('Leistung: ', leistung)

    # Lineare Regression
    model = LinearRegression().fit(np.array(energiedichte).reshape([-1, 1]), leistung)
    b0 = model.intercept_
    b1 = model.coef_
    print(b0, b1)

    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(energiedichte, leistung)
    ax.plot(energiedichte, b0 + b1*leistung)
    plt.xlabel("Energiedichte der Sonnenstahlung [$Wh/m^2$]")
    plt.ylabel("Energie aus den Solaranlagen [$Wh$]")
    plt.grid()
    plt.show()
