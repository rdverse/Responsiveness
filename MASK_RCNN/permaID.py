import pandas as pd 
import shutil
import os
import numpy as np
import sys

def rename_conv(prev, alt, path):

    prevDir = os.path.join(path, prev)
    prevCSV = os.path.join(path, prev + '.csv')

    altDir = os.path.join(path, alt)
    try:
        altCSV = os.path.join(path, alt+ '.csv')
    except:
        pass

    shutil.move(prevDir, altDir)
    try:
        shutil.move(prevCSV, altCSV)
    except:
        pass

if __name__=="__main__":
    # count of folders/csv's
    PATH = sys.argv[1]
    print(PATH)
    

    PATHds = os.path.join(PATH, 'ds')
    PATHres = os.path.join(PATH, 'chrisPP')

    folderNames = sorted([int(i) for i in os.listdir(PATHds)])
   
    count = len(folderNames)

    alternateNames = list()
   
    for i in folderNames:
        response = input("What is the alternate name for %d ?" %int(i))
        alternateNames.append(response)
    
    nalphabets = list('abcdefghijklmnopqrstuvwxyz'[:count])
    
    folderNames = np.array(folderNames).reshape(-1,1)
    nalphabets = np.array(nalphabets).reshape(-1,1)
    alternateNames = np.array(alternateNames).reshape(-1,1)
    data = np.hstack([folderNames,nalphabets,alternateNames])
    df = pd.DataFrame(data = data,columns=["original", 
                                        "alias", 
                                       "alternate"])


    for path in [PATHds, PATHres]:
        for orig, alias in df[["original", "alias"]].values:
            rename_conv(orig, alias, path)


        for alias, alternate in df[["alias", "alternate"]].values:
            rename_conv(alias, alternate, path)