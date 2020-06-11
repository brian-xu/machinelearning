import os

import wget


def download_mpg():
    if 'auto-mpg.data' not in os.listdir():
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')
        contents = open('auto-mpg.data', 'r').read()
        with open('auto-mpg.data', 'w') as f:
            contents = 'MPG Cylinders Displacement Horsepower Weight Acceleration Model Origin Name\n' + contents
            f.write(contents)


def download_wine():
    if 'wine.data' not in os.listdir():
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
        contents = open('wine.data', 'r').read()
        with open('wine.data', 'w') as f:
            contents = 'Alcohol,Malic Acid,Ash,Ash Alkalinity,Magnesium,Total Phenols,Flavanoids,Nonflavanoid Phenols,' \
                       'Protoanthocyanins,Color Intensity,Hue,OD280,OD315,Proline\n' + contents
            f.write(contents)


def download_grades():
    if 'student-mat.csv' not in os.listdir():
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip')
        import zipfile
        with zipfile.ZipFile('student.zip') as z:
            z.extract('student-mat.csv')
        os.remove('student.zip')


def download_digits():
    if 'optdigits.tra' not in os.listdir():
        wget.download('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra')
    if 'optdigits.tes' not in os.listdir():
        wget.download('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes')
