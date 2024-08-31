import sys
import math

class FWICLASS:
    def __init__(self,temp,rhum,wind,prcp):
        self.h= rhum
        self.t= temp
        self.w= wind
        self.p= prcp

    def FFMCcalc(self,ffmc0):
        rf=1
        mo = (147.2*(101.0 - ffmc0))/(59.5 + ffmc0)
        if (self.p > 0.5):
            rf = self.p - 0.5
            if(mo > 150.0):
                mo = (mo+42.5*rf*math.exp(-100.0/(251.0-mo))*(1.0 - math.exp(-6.93/rf))) + (.0015*(mo - 150.0)**2)*math.sqrt(rf)
        elif mo <= 150.0:
            mo = mo+42.5*rf*math.exp(-100.0/(251.0-mo))*(1.0 - math.exp(-6.93/rf))
        if(mo > 250.0):
            mo = 250.0

        ed = .942*(self.h**.679) + (11.0*math.exp((self.h-100.0)/10.0))+0.18*(21.1-self.t) *(1.0 - 1.0/math.exp(.1150 * self.h))

        if(mo < ed):
            ew = .618*(self.h**.753) + (10.0*math.exp((self.h-100.0)/10.0))+ .18*(21.1-self.t)*(1.0 - 1.0/math.exp(.115 * self.h))

            if(mo <= ew):
                kl = .424*(1.0-((100.0-self.h)/100.0)**1.7)+(.0694*math.sqrt(self.w)) *(1.0 - ((100.0 - self.h)/100.0)**8)
                kw = kl * (.581 * math.exp(.0365 * self.t))
                m = ew - (ew - mo)/10.0**kw
            elif mo > ew:
                m = mo
        elif(mo == ed):
            m = mo
        elif mo > ed:
            kl =.424*(1.0-(self.h/100.0)**1.7)+(.0694*math.sqrt(self.w))* (1.0-(self.h/100.0)**8)
            kw = kl * (.581*math.exp(.0365*self.t))
            m = ed + (mo-ed)/10.0 ** kw
        ffmc = (59.5 * (250.0 -m)) / (147.2 + m)
        if (ffmc > 101.0):
            ffmc = 101.0
        if (ffmc <= 0.0):
            ffmc = 0.0
        return ffmc

    def DMCcalc(self,dmc0,mth):
        el = [6.5,7.5,9.0,12.8,13.9,13.9,12.4,10.9,9.4,8.0,7.0,6.0]
        t = self.t
        if (t < -1.1):
            t = -1.1
        rk = 1.894*(t+1.1) * (100.0-self.h) * (el[mth-1]*0.0001)
        if self.p > 1.5:
            ra= self.p
            rw = 0.92*ra - 1.27
            wmi = 20.0 + 280.0/math.exp(0.023*dmc0)
        
            if dmc0 <= 33.0:
                b = 100.0 /(0.5 + 0.3*dmc0)
            elif dmc0 > 33.0:
                if dmc0 <= 65.0:
                    b = 14.0 - 1.3*math.log(dmc0)
                elif dmc0 > 65.0:
                    b = 6.2 * math.log(dmc0) - 17.2
            wmr = wmi + (1000*rw) / (48.77+b*rw)
            pr = 43.43 * (5.6348 - math.log(wmr-20.0))
        elif self.p <= 1.5:
            pr = dmc0
        if (pr<0.0):
            pr = 0.0
        dmc = pr + rk
        if(dmc<= 1.0):
            dmc = 1.0
        return dmc
        
    def DCcalc(self,dc0,mth):
        dc=0
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        t = self.t
        if(t < -2.8):
            t = -2.8
        pe = (0.36*(t+2.8) + fl[mth-1] )/2
        if pe <= 0.0:
            pe = 0.0
        if (self.p > 2.8):
            ra = self.p
            rw = 0.83*ra - 1.27
            smi = 800.0 * math.exp(-dc0/400.0)
            dr = dc0 - 400.0*math.log( 1.0+((3.937*rw)/smi))
            if (dr > 0.0):
                dc = dr + pe
        elif self.p <= 2.8:
            dc = dc0 + pe
        return dc
    
    def ISIcalc(self,ffmc):
        mo = 147.2*(101.0-ffmc) / (59.5+ffmc)
        ff = 19.115*math.exp(mo*-0.1386) * (1.0+(mo**5.31)/49300000.0)
        isi = ff * math.exp(0.05039*self.w)
        return isi

    def BUIcalc(self,dmc,dc):
        if dmc <= 0.4*dc:
            bui = (0.8*dc*dmc) / (dmc+0.4*dc)
        else:
            bui = dmc-(1.0-0.8*dc/(dmc+0.4*dc))*(0.92+(0.0114*dmc)**1.7)
        if bui <0.0:
            bui = 0.0
        return bui

    def FWIcalc(self,isi,bui):
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626*bui**0.809 + 2.0)
        else:
            bb = 0.1*isi*(1000.0/(25. + 108.64/math.exp(0.023*bui)))
        if(bb <= 1.0):
            fwi = bb
        else:
            fwi = math.exp(2.72 * (0.434*math.log(bb))**0.647)
        return fwi

def main():
    import sys
    ffmc0= 85.0
    dmc0= 6.0
    dc0= 15.0
    my_csv_in = sys.argv[1]

    with open(my_csv_in, 'r') as f_in:
        print("opened")
        next(f_in)
        #test.toPandas().to_csv('C:/Users/HARITA/Desktop/datasets/predicted.csv')
        with open('/Users/krsingh/Desktop/datasets/testset3.csv', 'w') as f_out:
            h=["Year","Month","Day","FFMC","DMC","DC","ISI","BUI","Temp","RH","Wind","Rain","FWI","Intensity","Fire"]
            hd=','.join(h)
            f_out.write(hd)
            f_out.write("\r")
            for line in f_in:
                l=line.rstrip().split(',')
                yr=l[0]
                mth=l[1]
                day=l[2]
                temp=float(l[3])
                rhum=float(l[4])
                wind=float(l[5])
                prcp=float(l[6])
                if rhum> 100.0:
                    rhum=100.0
                mth=int(mth)
                fwisystem= FWICLASS(temp,rhum,wind,prcp)
                ffmc = fwisystem.FFMCcalc(ffmc0)
                dmc = fwisystem.DMCcalc(dmc0,mth)
                dc = fwisystem.DCcalc(dc0,mth)
                isi = fwisystem.ISIcalc(ffmc)
                bui = fwisystem.BUIcalc(dmc,dc)
                fwi = fwisystem.FWIcalc(isi,bui)
                fire=0
                intensity=0
                '''if fwi> 1.0000:
                    fire=1
                if fwi<5.0000:
                    intensity=0
                elif fwi<=3.0000:
                    intensity=1
                elif fwi>3 and fwi<=7.5000:
                    intensity=2
                elif fwi>7.5000 and fwi<=12.0000:
                    intensity=3
                elif fwi>12.0000 and fwi<=24:
                    intensity=3
                elif fwi>24:
                    intensity=4'''
                if fwi> 5.0000:
                    fire=1
                if fwi<5.0000:
                    intensity=0
                elif fwi<=10.0000:
                    intensity=1
                elif fwi<=17.0000:
                    intensity=2
                elif fwi<24:
                    intensity=3
                elif fwi >24:
                    intensity=4
                l=[str(yr),str(mth),str(day),str(round(ffmc,4)),str(round(dmc,4)),str(round(dc,4)),str(round(isi)),str(round(bui,4)),str(round(temp,4)),str(round(rhum,4)),str(round(wind,4)),str(round(prcp,4)),str(round(fwi,4)),str(intensity),str(fire)]
                ffmc0 = ffmc
                dmc0 = dmc
                dc0 = dc
                d=','.join(l)
                f_out.write(d)
                #print(d)
                f_out.write("\r")






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
    import findspark

    findspark.init()

    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline

    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.feature import QuantileDiscretizer
    #import Spark and MLlib packages
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS

    #import data analysis packages
    import numpy as np
    import pandas as pd
    import sklearn

    from pandas import Series, DataFrame
    from sklearn import svm
    from sklearn.svm import SVC
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics

    from numpy import array
    from timeit import default_timer as timer
    #import data visualization packages
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')


    '''my_csv_in = sys.argv[1]
    my_csv_out = r'/Users/krsingh/Desktop/datasets/forestfirestest2.csv'

    with open(my_csv_in, 'r') as f_in:
        print("opened")
        with open(my_csv_out, 'w') as f_out:
            for line in f_in:
                f_out.write(line)
                f_out.write("\r")
    '''

    import random

    dataframets= pandas.read_csv(r"/Users/krsingh/Desktop/datasets/testset3.csv")
    dataframetr = pandas.read_csv(r"/Users/krsingh/Desktop/datasets/newtraintdata7.csv")

    #dataframetr.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
    #dataframetr.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
    featuredatatr=dataframetr.iloc[:, :14]
    targetvaluestr=dataframetr.iloc[:,14:]
    #dataframets.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
    #dataframets.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
    featuredatats=dataframets.iloc[:, :14]
    targetvaluests=dataframets.iloc[:,14:]

    x_train=featuredatatr
    x_test=featuredatats
    y_train=targetvaluestr
    y_test=targetvaluests
    # SVM regularizaion parameter
    C = 1.0

    #Let's use different Kernel, whic is nothing but x_test
    #the functions mapping data to hyper dimension

    #SVC with a linear Kernel
    svc = svm.SVC(kernel = 'linear', C=C).fit(x_train, y_train)
    start = timer()
    predicted = svc.predict(x_test)
    end =timer()
    predicted = svc.predict(x_test)
    expected = y_test
    # Compare results
    lsvmaccuracy = metrics.accuracy_score(expected,predicted)
    lsvmprecision=metrics.precision_score(expected, predicted)
    lsvmrecall=metrics.recall_score(expected, predicted)
    
    print(lsvmaccuracy)
    print(lsvmprecision)
    print(lsvmrecall)
    print(end-start)
    
    from pyspark.ml import Pipeline

    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.feature import QuantileDiscretizer

    from pyspark.ml.classification import LinearSVC
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    spark = SparkSession \
        .builder \
        .appName("Spark ML example on data ") \
        .getOrCreate()

    datatrain= "/Users/krsingh/Desktop/datasets/newtraintdata7.csv"
    dftr = spark.read.csv(datatrain,header = 'True',inferSchema='True')
    datatest= "/Users/krsingh/Desktop/datasets/testset3.csv"
    dfts = spark.read.csv(datatest,header = 'True',inferSchema='True')

    '''indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dftr) for column in ["month","day"]]
    pipeline = Pipeline(stages=indexers)
    dfdr = pipeline.fit(dftr).transform(dftr)

    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dfts) for column in ["month","day"]]
    pipeline = Pipeline(stages=indexers)
    dfds = pipeline.fit(dfts).transform(dfts)

    dftr = dftr.drop("Month","Day")

    dfts = dfts.drop("Month","Day")'''


    featuretr = VectorAssembler(
        inputCols=[x for x in dftr.columns],
        outputCol='features')
    feature_vector_tr= featuretr.transform(dftr)
    featurets = VectorAssembler(
        inputCols=[x for x in dfts.columns],
        outputCol='features')
    feature_vector_ts= featurets.transform(dfts)
    trainingData=feature_vector_tr
    testData=feature_vector_ts
    from pyspark.ml.classification import LinearSVC
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    svm = LinearSVC(labelCol="Fire", featuresCol="features")
    svm_model = svm.fit(trainingData)
    start1=timer()
    svm_prediction = svm_model.transform(testData)
    end1=timer()
    

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='Fire', metricName='accuracy')
    psvmaccuracy=evaluator.evaluate(svm_prediction)
    
    test=svm_prediction.drop("features","rawPrediction")
    print(psvmaccuracy)
    sns.set_palette("husl")
    firevsmonth = sns.lineplot(x="Month",y="Fire",data=svm_prediction.toPandas()).set_title('Fire')
    plt.savefig('/Users/krsingh/fyp_project/FYP/assets/output.png')
    plt.close()
    sns.set_palette("PuBuGn_d")
    predvsmonth = sns.lineplot(x="Month",y="prediction",data=svm_prediction.toPandas()).set_title('Predictions')
    plt.savefig('/Users/krsingh/fyp_project/FYP/assets/output1.png')
    plt.close()
    sns.set_palette("hls")
    intvsmonth = sns.lineplot(x="Month",y="Intensity",data=test.toPandas()).set_title('Fire Intenisty')
    plt.savefig('/Users/krsingh/fyp_project/FYP/assets/output2.png')
    plt.close()
    svm_pred=svm_prediction.toPandas()
    l=svm_pred.Fire
    y_pred=svm_pred.prediction
    from sklearn.metrics import classification_report, confusion_matrix
    print(metrics.precision_score(l, y_pred))
    print(metrics.recall_score(l, y_pred))
    print(end1-start1)
    #print(classification_report(l,y_pred))
    pd.set_option('colheader_justify', 'center')   # FOR TABLE <th>
    def message (row):
        if row['Intensity']==2:
            return "SMALL FIRE ALERT !"
        elif row['Intensity']==3:
            return "WILD FIRE ALERT !!!"
        elif row['Intensity']==4:
            return "WILD FIRE ALERT !!!"
        else :
            return "----------"

    test=svm_prediction.drop("Year","Temp","RH","Wind","Rain","features","rawPrediction")
    test=test.filter(svm_prediction.prediction==1)
    test=test.filter(svm_prediction.Intensity>=2)
    test1=test.toPandas()
    test1['MESSAGE'] = test1.apply (lambda row: message(row), axis=1)

    from bs4 import BeautifulSoup
    with open("/Users/krsingh/fyp_project/FYP/FYP/template/pg2.html", "r") as f:
        contents = f.read()
        soup = BeautifulSoup(contents,'html.parser')
        ptag2 = soup.find("div",id='table_pd')
        if(ptag2):        
            ptag2.decompose()        
    f.close()
    if(soup):
        with open("/Users/krsingh/fyp_project/FYP/FYP/template/pg2.html", "w") as f:
            f.write(soup.prettify())
    f.close()
    html_string = '''
    <div id='table_pd'>
        {table}
    </div>   
    '''

    # OUTPUT AN HTML FILE
    with open('/Users/krsingh/fyp_project/FYP/FYP/template/pg2.html', 'a') as f:
        f.write(html_string.format(table=test1.to_html(classes='mystyle')))

    with open("/Users/krsingh/fyp_project/FYP/FYP/template/pg4.html", "r") as f:
        contents = f.read()
        soup = BeautifulSoup(contents,'html.parser')
        ptag2 = soup.find("div",id='table_pd')
        if(ptag2):        
            ptag2.decompose()        
    f.close()
    if(soup):
        with open("/Users/krsingh/fyp_project/FYP/FYP/template/pg4.html", "w") as f:
            f.write(soup.prettify())
    f.close()
    html_string = '''
    <div id='table_pd'>
        {table}
    </div>   
    '''

    # OUTPUT AN HTML FILE
    with open('/Users/krsingh/fyp_project/FYP/FYP/template/pg4.html', 'a') as f:
        f.write(html_string.format(table=test1.to_html(classes='mystyle')))

    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import smtplib
    fromaddr = "ForestfireAlerts.gmail.com"
    toaddr = ["kars16cs@cmrit.ac.in","kmad16cs@cmrit.ac.in","kpne16cs@cmrit.ac.in","hata16cs@cmrit.ac.in","pushpa.m@cmrit.ac.in"]
    for i in range(len(toaddr)):
        html = open("/Users/krsingh/fyp_project/FYP/FYP/template/pg4.html")
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr[i]
        msg['Subject'] = "Fire Alerts Report"
        part2 = MIMEText(html.read(), 'html')
        msg.attach(part2)
        debug = False
        if debug:
            print(msg.as_string())
        else:
            server = smtplib.SMTP('smtp.gmail.com',587)
            server.starttls()
            server.login("forestfire.alerts@gmail.com", "hmnk30241812")
            text = msg.as_string()
            server.sendmail(fromaddr, toaddr[i], text)
            server.quit()
    
    
main()