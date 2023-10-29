from flask import Flask, Response, request
from pathlib import Path
import csv
import pandas as pd
from random import randrange
from datetime import date

app = Flask(__name__)

@app.route('/')
def index():
    return '''
        <html>
            <body>
                <h1>Generate Data CSV</h1>
                <form action="/generate" method="post">
                    <label for="series">Series:</label>
                    <input type="number" id="series" name="series" value="1"><br><br>
                    <label for="nr_random_lines">Number of random lines:</label>
                    <input type="number" id="nr_random_lines" name="nr_random_lines" value="7"><br><br>
                    <label for="min_power">Minimum power:</label>
                    <input type="number" id="min_power" name="min_power" value="5"><br><br>
                    <label for="max_power">Maximum power:</label>
                    <input type="number" id="max_power" name="max_power" value="1190"><br><br>
                    <label for="min_time">Minimum time:</label>
                    <input type="number" id="min_time" name="min_time" value="1050"><br><br>
                    <label for="max_time">Maximum time:</label>
                    <input type="number" id="max_time" name="max_time" value="5000"><br><br>
                    <label for="min_pressure">Minimum pressure:</label>
                    <input type="number" id="min_pressure" name="min_pressure" value="100"><br><br>
                    <label for="max_pressure">Maximum pressure:</label>
                    <input type="number" id="max_pressure" name="max_pressure" value="350"><br><br>
                    <input type="submit" value="Generate CSV">
                </form>
            </body>
        </html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    series = int(request.form['series'])
    nr_random_lines = int(request.form['nr_random_lines'])
    min_power = int(request.form['min_power'])
    max_power = int(request.form['max_power'])
    min_time = int(request.form['min_time'])
    max_time = int(request.form['max_time'])
    min_pressure = int(request.form['min_pressure'])
    max_pressure = int(request.form['max_pressure'])

    power=[]
    line_time=[]
    pressure=[]
    line_passes=[]
    defocus = []

    print("TODAY's DATE:",str(date.today()))

    for x in range(nr_random_lines):
        powr = randrange(min_power, max_power, 1) # in mW
        tm = randrange(min_time, max_time, 1) # in ms
        pr = randrange(min_pressure, max_pressure, 10) # in psi
        passes = randrange(1, 10, 1) #passes
        defoc = randrange(-300, 0,1)/100
        for i in range(9):
            power.append(powr)
            line_time.append(tm)
            pressure.append(pr)
            line_passes.append(passes)
            defocus.append(defoc)

    print(f"POWER: ({min_power}, {max_power})\n",
          f"TIME: ({min_time}, {max_time})\n",
          f"PRESSURE: ({min_pressure}, {max_pressure})")

    data_header=['power','time','pressure','passes','defocus','ratio', 'resistance']

    with open('dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_header)
        writer.writerows(zip(power, line_time, pressure,line_passes,defocus))

    df2=pd.read_csv('dataset.csv')
    df2=df2.drop_duplicates() #keep only the unique rows
    df2.to_csv('data.csv',index=False) #this is what will be read by mlrMBO in the R code

    path = Path('data.csv')
    return Response(
        path.read_bytes(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename={path.name}'}
    )

if __name__ == '__main__':
    app.run(port=8080, debug=True)