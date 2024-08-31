from django.shortcuts import render
print(__doc__)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math 
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import svm, datasets
sns.set(color_codes=True)
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import sys
from django.contrib import messages
from django.db import models
import os
'''def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/index.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'core/index.html')'''
from subprocess import run, PIPE ,Popen, call,check_output
def button(request):
    return render(request,'index.html')
def output(request):
    return render(request,'index.html')


def external(request):
    file=request.FILES['myfile']
    #messages.success(request, 'Form submission successful')
    #doc = request.FILES #returns a dict-like object
    #file = doc['myfile']
    f="/Users/krsingh/Desktop/datasets/"+str(file)
    #p=Popen(['python3', 'forestfire.py',f],cwd="~",stdout=PIPE ,shell=False)
    #output = check_output(["forestfire.py", "--", f])
    output=Popen(['python3','/Users/krsingh/fyp_project/FYP/FYP/forestfire.py',f],stdout=PIPE,universal_newlines=True)
    l=[]
    for line in output.stdout.readlines():
        l.append(line)
    print(len(l))
    return render(request, 'index.html', {'Accuracy_LSVM':l[1],'Precision_LSVM':l[2],'Recall_LSVM':l[3],'Time_LSVM':l[4],'Accuracy_PSVM':l[5],'Precision_PSVM':l[6],'Recall_PSVM':l[7],'Time_PSVM':l[8]})
    #,Accuracy_lr':l[4],'Precision_lr':l[5],'Recall_lr':l[6],'Accuracy_ANN':l[7],'Precision_ANN':l[8],'Recall_ANN':l[9]})

def alert(request):
    return render(request,'pg2.html')

def prediction(request):
    return render(request,"prediction.html")
def validation(request):
    return render(request,"validation.html")

def predictor(request):
    return render(request,"index.html")