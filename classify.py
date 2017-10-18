import numpy as np
import csv
import yaml
import pickle
from classifier.crossvalidation import Crossvalidation as clcv
from classifier.knn import Knn as clf

# trisetiobr@gmail.com

def main():
    with open('./system_config.json') as f:
        config = yaml.load(f)['classifier']

    classifier = config['name']
    indir = config['indir']
    outdir = config['outdir']
    parameters = config['parameters']
            
    # load feature
    with open(indir,'rb') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    X = []
    y = []
    for i in range(1, len(data)):
        X.append(data[i][0:-1])
        y.append(data[i][-1])

    cv = clcv(X,y)
    cv.split(k=5, stratified=True)
    classifier = clf(X,y,3)
    classifier.k_search(cv, parameters['k'], log=True)
    pickle.dump( classifier, open( outdir, "wb" ) )

if __name__ == '__main__':
    main()