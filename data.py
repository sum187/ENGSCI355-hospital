import csv
from multiprocessing import Event
from tokenize import Double
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from fitter import Fitter
import os

# get patient_data
data = []
with open("patient_data.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        data.append(row)
header[0]='ID'
data=np.array(data)
df = pd.DataFrame(data,columns=header)

# get list of different patient ids and events
id_list=[]
for line in data:
    id=line[0]
    if id not in id_list:
        id_list.append(id)
events=[]
for event in data[:,2]:
    if event not in events:
        events.append(event)

# get orderly_data
orderly_data = []
with open("orderly_data.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        orderly_data.append(row)
header[0]='Entity'        
orderly_data=np.array(orderly_data)
orderly_df = pd.DataFrame(orderly_data,columns=header)

def to_int_hour(dt_time):
    try:
        time=24*dt_time.days + dt_time.seconds/3600
    except:
        time=24*dt_time.day + dt_time.second/3600
    return time

################################################################################################################################################
# Ward
################################################################################################################################################
ward_event=['Wards.start-obs', 'Wards.start-stay', 'Wards.leave', 'Wards.start-discharge',
 'Wards.start-wait-admission', 'Wards.start-admission', 'Wards.start-wait-test', 'Wards.start-test']
if True:
    if not os.path.exists('time_data/Admission_time.csv'):
        Admission_time=[]
        WardStayTime=[]
        DischargeTime=[]
        TestTime=[]
        WardObservationTime=[]
        for id in id_list:
            list=df.loc[df['ID'] == id]
            time1=list.loc[list['Event'] == 'Wards.start-admission']['Time'].to_numpy()
            time2=list.loc[list['Event'] == 'Wards.start-stay']['Time'].to_numpy()
            time3=list.loc[list['Event'] == 'Wards.start-discharge']['Time'].to_numpy()
            time4=list.loc[list['Event'] == 'Wards.leave']['Time'].to_numpy()
            time5=list.loc[list['Event'] == 'Wards.start-wait-test']['Time'].to_numpy()
            time6=list.loc[list['Event'] == 'Wards.start-obs']['Time'].to_numpy()
            
            if np.size(time2)>0 and np.size(time6)>0:
                n=1
                if np.size(time2)==np.size(time6):
                    n=0
                for i in range(np.size(time6)):
                    try:
                        t1=datetime.strptime(time2[i+n], '%d/%m/%Y %H:%M:%S')
                        t2=datetime.strptime(time6[i], '%d/%m/%Y %H:%M:%S')
                        time=to_int_hour(t1-t2)
                    except:
                        1    
                    if time<0:
                        n=1
                        try:
                            t1=datetime.strptime(time2[i+n], '%d/%m/%Y %H:%M:%S')
                            t2=datetime.strptime(time6[i], '%d/%m/%Y %H:%M:%S')
                            time=to_int_hour(t1-t2)
                        except:
                            1
                    if time>=0:
                        WardObservationTime.append(time*60)        

            if np.size(time1)>0:
                t1=datetime.strptime(time1[0], '%d/%m/%Y %H:%M:%S')
                t2=datetime.strptime(time2[0], '%d/%m/%Y %H:%M:%S')
                Admission_time.append(to_int_hour(t2-t1)*60)

            if np.size(time3)>0 and np.size(time2)>0:
                t3=datetime.strptime(time3[0], '%d/%m/%Y %H:%M:%S')
                t2=datetime.strptime(time2[0], '%d/%m/%Y %H:%M:%S')
                WardStayTime.append(to_int_hour(t3-t2)) 

            if np.size(time3)>0 and np.size(time4)>0:
                t3=datetime.strptime(time3[0], '%d/%m/%Y %H:%M:%S')
                t4=datetime.strptime(time4[-1], '%d/%m/%Y %H:%M:%S')
                DischargeTime.append(to_int_hour(t4-t3)*60)    
            
            if np.size(time4)>0 and np.size(time5)>0:
                t5=datetime.strptime(time5[0], '%d/%m/%Y %H:%M:%S')
                t4=datetime.strptime(time4[0], '%d/%m/%Y %H:%M:%S')
                TestTime.append(to_int_hour(t4-t5)*60)  

        if not os.path.exists('time_data/'):
            os.makedirs('time_data/')         
        with open('time_data/Admission_time.csv', 'w') as my_file:
            np.savetxt(my_file,Admission_time)
        with open('time_data/WardStayTime.csv', 'w') as my_file:
            np.savetxt(my_file,WardStayTime)
        with open('time_data/DischargeTime.csv', 'w') as my_file:
            np.savetxt(my_file,DischargeTime)
        with open('time_data/TestTime.csv', 'w') as my_file:
            np.savetxt(my_file,TestTime)
        with open('time_data/WardObservationTime.csv', 'w') as my_file:
            np.savetxt(my_file,WardObservationTime)    
    else:
        Admission_time = np.loadtxt('time_data/Admission_time.csv')
        WardStayTime = np.loadtxt('time_data/WardStayTime.csv')
        DischargeTime = np.loadtxt('time_data/DischargeTime.csv')
        TestTime = np.loadtxt('time_data/TestTime.csv')
        WardObservationTime = np.loadtxt('time_data/WardObservationTime.csv')

    plt.figure()
    f = Fitter(WardObservationTime,distributions=["norm"])
    f.fit()
    f.summary()
    print(f.fitted_param["norm"])
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for WardObservationTime')
    print('##############################################################################')
    print('WardObservationTime: Normal Distribution')
    print('Mean={}, StandardDeviation={}'.format(*f.fitted_param["norm"]))
    print('##############################################################################')

    plt.figure()
    f = Fitter(Admission_time,distributions=["norm"])
    f.fit()
    f.summary()
    print(f.fitted_param["norm"])
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for Admission_time')
    print('##############################################################################')
    print('Admission_time: Normal Distribution')
    print('Mean={}, StandardDeviation={}'.format(*f.fitted_param["norm"]))
    print('##############################################################################')

    plt.figure()
    f = Fitter(WardStayTime,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in hour)')
    plt.ylabel('probability')
    plt.title('Distribution fit for WardStayTime')
    print('##############################################################################')
    print('WardStayTime: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')

    plt.figure()
    f = Fitter(DischargeTime,distributions=["norm"])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for DischargeTime')
    print('##############################################################################')
    print('DischargeTime: Normal Distribution')
    print('Mean={}, StandardDeviation={}'.format(*f.fitted_param["norm"]))
    print('##############################################################################')

    plt.figure()
    plt.hist(TestTime,bins=20) #lognormal
    f = Fitter(TestTime,distributions=['lognorm'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TestTime')
    print('##############################################################################')
    print('TestTime: Exponential Distribution')
    print('NormalStandardDeviation={}, NormalMean={}, Scale={}'.format(*f.fitted_param["lognorm"]))
    print('##############################################################################')

##############################################################################################################################################
# PT
##############################################################################################################################################
PT_event=['PT.wait-for-assignment-start', 'PT.orderly-to-patient-start', 'PT.dropoff-start', 'PT.leave', 'PT.pickup-start', 'PT.travel-start']
PickupTime=[]
DropoffTime=[]
TravelTime=[]
DropWaitTime=[]

if False:
    if not os.path.exists('time_data/PickupTime.csv'):
        for id in id_list:
            list=df.loc[df['ID'] == id]
            time1=list.loc[list['Event'] == 'PT.pickup-start']['Time'].to_numpy()
            time2=list.loc[list['Event'] == 'PT.travel-start']['Time'].to_numpy()
            time3=list.loc[list['Event'] == 'PT.dropoff-start']['Time'].to_numpy()
            
            # get PickupTime
            if np.size(time1)>0 and np.size(time2)>0:
                for i in range(len(time2)):
                    t1=datetime.strptime(time1[i], '%d/%m/%Y %H:%M:%S')
                    t2=datetime.strptime(time2[i], '%d/%m/%Y %H:%M:%S')
                    PickupTime.append(to_int_hour(t2-t1)*60)

            ###########################################################################
            ######################## Wrong ##########################################
            '''
            # get TravelTime
            if np.size(time2)>0 and np.size(time3)>0:
                for i in range(len(time3)):
                    t2=datetime.strptime(time2[i], '%d/%m/%Y %H:%M:%S')
                    t3=datetime.strptime(time3[i], '%d/%m/%Y %H:%M:%S')
                    TravelTime.append(to_int_hour(t3-t2))
            '''        
            ############################################################################
        orderly_Entity=['Orderly7', 'Orderly9', 'Orderly1', 'Orderly4', 'Orderly5', 'Orderly6', 'Orderly2', 'Orderly8', 'Orderly3']
        orderly_event=['PatientTransit.orderly-to-patient-start', 'PatientTransit.dropoff-start', 'PatientTransit.start-base-travel',
        'PatientTransit.wait-dropoff-start', 'PatientTransit.start-wait-task', 'PatientTransit.pickup-start', 'PatientTransit.travel-start']

        for ent in orderly_Entity:
            list=orderly_df.loc[orderly_df['Entity'] == ent]
            time1=list.loc[list['Event'] == 'PatientTransit.dropoff-start']['Time'].to_numpy()
            time2=list.loc[list['Event'] == 'PatientTransit.wait-dropoff-start']['Time'].to_numpy()
            time3=list.loc[list['Event'] == 'PatientTransit.start-base-travel']['Time'].to_numpy()
            
            # get DropoffTime
            for i in range(len(time2)):
                t1=datetime.strptime(time1[i], '%d/%m/%Y %H:%M:%S')
                t2=datetime.strptime(time2[i], '%d/%m/%Y %H:%M:%S')
                DropoffTime.append(to_int_hour(t2-t1)*60)
            
            # get DropWaitTime
            for i in range(len(list)):
                if list.iloc[i]['Event']=='PatientTransit.wait-dropoff-start' and i<len(list)-1:
                    if list.iloc[i+1]['Event']=='PatientTransit.start-base-travel':
                        t1=datetime.strptime(list.iloc[i]['Time'], '%d/%m/%Y %H:%M:%S')
                        t2=datetime.strptime(list.iloc[i+1]['Time'], '%d/%m/%Y %H:%M:%S')
                        DropWaitTime.append(to_int_hour(t2-t1)*60)

        if not os.path.exists('time_data/'):
            os.makedirs('time_data/')
        #with open('time_data/TravelTime.csv', 'w') as my_file:
        #    np.savetxt(my_file,TravelTime)
        with open('time_data/PickupTime.csv', 'w') as my_file:
            np.savetxt(my_file,PickupTime)
        with open('time_data/DropoffTime.csv', 'w') as my_file:
            np.savetxt(my_file,DropoffTime)
        with open('time_data/DropWaitTime.csv', 'w') as my_file:
            np.savetxt(my_file,DropWaitTime)

    else:
        PickupTime = np.loadtxt('time_data/PickupTime.csv')
        #TravelTime = np.loadtxt('time_data/TravelTime.csv')
        DropoffTime = np.loadtxt('time_data/DropoffTime.csv')
        DropWaitTime = np.loadtxt('time_data/DropWaitTime.csv')
    
    plt.figure(figsize=(5, 4))
    f = Fitter(PickupTime,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for PickupTime')
    print('##############################################################################')
    print('PickupTime: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')
    ######################################################################################
    '''
    plt.figure(figsize=(5, 4))
    f = Fitter(TravelTime,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TravelTime')
    print('##############################################################################')
    print('TravelTime: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')
    '''
    #####################################################################################
    plt.figure(figsize=(5, 4))
    f = Fitter(DropoffTime,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for DropoffTime')
    print('##############################################################################')
    print('DropoffTime: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    plt.hist(DropWaitTime,bins=30)
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for DropWaitTime')
    print('DropWaitTime: Constant time')
    print('ConstantTime=20')
    print('##############################################################################')

    # get TravelTime
    orderly_Entity=['Orderly7', 'Orderly9', 'Orderly1', 'Orderly4', 'Orderly5', 'Orderly6', 'Orderly2', 'Orderly8', 'Orderly3']
    orderly_event=['PatientTransit.orderly-to-patient-start', 'PatientTransit.dropoff-start', 'PatientTransit.start-base-travel',
            'PatientTransit.wait-dropoff-start', 'PatientTransit.start-wait-task', 'PatientTransit.pickup-start', 'PatientTransit.travel-start']

    a=orderly_df.loc[orderly_df['Entity'] == 'Orderly7']
    def print_full(x):
        pd.set_option('display.max_rows', len(x))
        print(x)
        pd.reset_option('display.max_rows')

    TravelTime_B_to_Ward=[]
    TravelTime_B_to_ED=[]
    TravelTime_Ward_to_ED=[]

    def add_TravelTime_from_to(start,end,time,TravelTime_B_to_Ward,TravelTime_B_to_ED,TravelTime_Ward_to_ED):
        if (start=='OrderlyBase' and end=='ED_Submodel') or (start=='ED_Submodel' and end=='OrderlyBase'):
            TravelTime_B_to_ED.append(time*60)
        elif (start=='OrderlyBase' and end=='Wards_Submodel') or (start=='Wards_Submodel' and end=='OrderlyBase'):    
            TravelTime_B_to_Ward.append(time*60)
        elif (start=='ED_Submodel' and end=='Wards_Submodel') or (start=='Wards_Submodel' and end=='ED_Submodel'):
            TravelTime_Ward_to_ED.append(time*60)    

    for ent in orderly_Entity:
        list=orderly_df.loc[orderly_df['Entity'] == ent]
        for i in range(len(list)-1):
            value=list.iloc[i]
            if value['Event']=='PatientTransit.orderly-to-patient-start' or value['Event']=='PatientTransit.travel-start' or value['Event']=='PatientTransit.start-base-travel':
                start=value['OrdStartLoc']

                if value['Event']=='PatientTransit.orderly-to-patient-start':
                    end=value['PatStartLoc']
                else:
                    end=value['Dest']

                time0=value['Time']
                time1=list.iloc[i+1]['Time']
                t1=datetime.strptime(time0, '%d/%m/%Y %H:%M:%S')
                t2=datetime.strptime(time1, '%d/%m/%Y %H:%M:%S')
                add_TravelTime_from_to(start,end,to_int_hour(t2-t1),TravelTime_B_to_Ward,TravelTime_B_to_ED,TravelTime_Ward_to_ED)


    plt.figure(figsize=(5, 4))
    f = Fitter(TravelTime_B_to_Ward,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TravelTime_B_to_Ward')
    print('##############################################################################')
    print('TravelTime_B_to_Ward: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    f = Fitter(TravelTime_B_to_ED,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TravelTime_B_to_ED')
    print('##############################################################################')
    print('TravelTime_B_to_ED: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    f = Fitter(TravelTime_Ward_to_ED,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TravelTime_Ward_to_ED')
    print('##############################################################################')
    print('TravelTime_Ward_to_ED: Exponential Distribution')
    print('MinValue={}, Mean={}'.format(*f.fitted_param["expon"]))
    print('##############################################################################')

###############################################################################################################################################
# ED
###############################################################################################################################################
ED_event=['ED.second-consultation-start', 'ED.wait-second-consultation-start', 'ED.leave-tests', 
'ED.wait-registration-start', 'ED.observation-start', 'ED.wait-test-start', 'ED.register-start', 'ED.leave-ed', 
'ED.wait-consultation-start', 'ED.consultation-start', 'ED.wait-triage-start', 'ED.triage-start']

RegistrationTime=[]
TriageTime=[]
ConsultationTime=[]
ObservationTime=[]
TestResultsTime=[]
if False:
    # check if timedata file exist
    if not os.path.exists('time_data/RegistrationTime.csv'):
        for id in id_list:
            list=df.loc[df['ID'] == id]
            time1=list.loc[list['Event'] == 'ED.register-start']['Time'].to_numpy()
            time11=list.loc[list['Event'] == 'ED.wait-triage-start']['Time'].to_numpy()
            time2=list.loc[list['Event'] == 'ED.triage-start']['Time'].to_numpy()
            time21=list.loc[list['Event'] == 'ED.wait-consultation-start']['Time'].to_numpy()     
            time3=list.loc[list['Event'] == 'ED.wait-test-start']['Time'].to_numpy()
            time4=list.loc[list['Event'] == 'ED.wait-second-consultation-start']['Time'].to_numpy()   
            
            # get RegistrationTime
            if np.size(time1)>0 and np.size(time11)>0:
                t1=datetime.strptime(time1[0], '%d/%m/%Y %H:%M:%S')
                t11=datetime.strptime(time11[0], '%d/%m/%Y %H:%M:%S')
                RegistrationTime.append(to_int_hour(t11-t1)*60)
            
            # get TriageTime
            if np.size(time2)>0 and np.size(time21)>0:
                t2=datetime.strptime(time2[0], '%d/%m/%Y %H:%M:%S')
                t21=datetime.strptime(time21[0], '%d/%m/%Y %H:%M:%S')
                TriageTime.append(to_int_hour(t21-t2)*60)

            # get ConsultationTime
            for i in range(len(list)):
                if (list.iloc[i]['Event']=='ED.consultation-start' or list.iloc[i]['Event']=='ED.second-consultation-start') and i<len(list)-1:
                    on=True
                    num=1
                    while on:
                        if (list.iloc[i+num]['Event']=='ED.leave-tests' or list.iloc[i+num]['Event']=='ED.leave-ed'):
                            t1=datetime.strptime(list.iloc[i]['Time'], '%d/%m/%Y %H:%M:%S')
                            t2=datetime.strptime(list.iloc[i+num]['Time'], '%d/%m/%Y %H:%M:%S')
                            Time=to_int_hour(t2-t1)
                            ConsultationTime.append(to_int_hour(t2-t1)*60)
                            on=False
                        num+=1
                        if on==True:
                            on=i<len(list)-num

            # get ObservationTime
            for i in range(len(list)):
                if list.iloc[i]['Event']=='ED.observation-start' and i<len(list)-1:
                    if (list.iloc[i+1]['Event']=='ED.wait-consultation-start' or list.iloc[i+1]['Event']=='ED.wait-second-consultation-start' or
                        list.iloc[i+1]['Event']=='ED.wait-test-start' or list.iloc[i]['Time']!=list.iloc[i+1]['Time']):
                        t1=datetime.strptime(list.iloc[i]['Time'], '%d/%m/%Y %H:%M:%S')
                        t2=datetime.strptime(list.iloc[i+1]['Time'], '%d/%m/%Y %H:%M:%S')
                        Time=to_int_hour(t2-t1)
                        ObservationTime.append(to_int_hour(t2-t1)*60)

            # get TestResultsTime
            if np.size(time3)>0 and np.size(time4)>0:
                t3=datetime.strptime(time3[0], '%d/%m/%Y %H:%M:%S')
                t4=datetime.strptime(time4[0], '%d/%m/%Y %H:%M:%S')
                TestResultsTime.append(to_int_hour(t4-t3)*60)  
                # remove 0 TestResultsTime           
                if TestResultsTime[-1]==0:
                    TestResultsTime.pop(-1)
        
        if not os.path.exists('time_data/'):
            os.makedirs('time_data/')
        with open('time_data/RegistrationTime.csv', 'w') as my_file:
            np.savetxt(my_file,RegistrationTime)
        with open('time_data/TriageTime.csv', 'w') as my_file:
            np.savetxt(my_file,TriageTime)
        with open('time_data/ConsultationTime.csv', 'w') as my_file:
            np.savetxt(my_file,ConsultationTime)    
        with open('time_data/ObservationTime.csv', 'w') as my_file:
            np.savetxt(my_file,ObservationTime)    
        with open('time_data/TestResultsTime.csv', 'w') as my_file:
            np.savetxt(my_file,TestResultsTime)
    else:
        RegistrationTime = np.loadtxt('time_data/RegistrationTime.csv')
        TriageTime = np.loadtxt('time_data/TriageTime.csv')
        ConsultationTime = np.loadtxt('time_data/ConsultationTime.csv')
        ObservationTime = np.loadtxt('time_data/ObservationTime.csv')
        TestResultsTime = np.loadtxt('time_data/TestResultsTime.csv')

    plt.figure(figsize=(5, 4))
    f = Fitter(RegistrationTime,distributions=['uniform'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for RegistrationTime')
    print('##############################################################################')
    print('RegistrationTime: Uniform Distribution')
    print('Min={}, Max={}'.format(min(RegistrationTime),max(RegistrationTime)))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    f = Fitter(TriageTime,distributions=["lognorm"])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TriageTime')
    print('##############################################################################')
    print('TriageTime: LogNormal Distribution')
    print('NormalStandardDeviation={}, NormalMean={}, Scale={}'.format(*f.fitted_param["lognorm"]))
    print('##############################################################################')
    #(shape, loc, scale)
    #(loc is µ, shape is σ and scale is α):
    ###########################################################################################
    # check lognormal distribution
    '''
    param=f.fitted_param["lognorm"]
    mu, sigma,scale = param[1],param[0],param[2]
    x = np.linspace(min(TriageTime), max(TriageTime), 10000)
    pdf = (1/((x-mu)*np.sqrt(2*np.pi*sigma**2))*np.exp(-np.log((x-mu)/scale)**2/(2*sigma**2)))
    plt.plot(x, pdf, 'r--',linewidth=0.3)
    plt.axis('tight')
    '''
    ###########################################################################################

    plt.figure(figsize=(5, 4))
    f = Fitter(ConsultationTime,distributions=["lognorm"])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for ConsultationTime')
    print('##############################################################################')
    print('ConsultationTime: LogNormal Distribution')
    print('NormalStandardDeviation={}, NormalMean={}, Scale={}'.format(*f.fitted_param["lognorm"]))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    f = Fitter(ObservationTime,distributions=['norm'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for ObservationTime')
    print('##############################################################################')
    print('ObservationTime: Normal Distribution')
    print('Mean={}, StandardDeviation={}'.format(*f.fitted_param["norm"]))
    print('##############################################################################')

    plt.figure(figsize=(5, 4))
    f = Fitter(TestResultsTime,distributions=['norm'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in minute)')
    plt.ylabel('probability')
    plt.title('Distribution fit for TestResultsTime')
    print('##############################################################################')
    print('TestResultsTime: Normal Distribution')
    print('Mean={}, StandardDeviation={}'.format(*f.fitted_param["norm"]))
    print('##############################################################################')
    ####################################################################################
    # Working out Patient InterArrivalTime
    ####################################################################################
    Arrival_time=[]
    ids=[int(i) for i in id_list]
    ids.sort()
    for id in ids:
        patient=df.loc[df['ID']==str(id)]
        patient_event=patient['Event'].iloc[0:3].to_numpy()
        just_arrived='patient-arrive' in patient_event and ('ED.wait-consultation-start' in patient_event or 'ED.wait-registration-start' in patient_event or
                                                            'ED.wait-triage-start' in patient_event)                                                 
        if just_arrived:
            time=datetime.strptime(patient['Time'].iloc[0], '%d/%m/%Y %H:%M:%S')
            Arrival_time.append(time)

    Arrival_time.sort()
    Arrival_time=np.array(Arrival_time)        
    InterArrivalTime=Arrival_time[1:]-Arrival_time[0:-1]
    InterArrivalTime=[to_int_hour(i) for i in InterArrivalTime]

    plt.figure(figsize=(5, 4))
    f = Fitter(InterArrivalTime,distributions=['expon'])
    f.fit()
    f.summary()
    plt.xlabel('Time (in hour)')
    plt.ylabel('probability')
    plt.title('Distribution fit for patient InterArrivalTime')
    print('##############################################################################')
    print('InterArrivalTime: Exponential Distribution')
    print('Mean={}'.format(f.fitted_param["expon"][1]))
    print('##############################################################################')
    ####################################################################################
    # working out discrete distribution of priority patient
    ####################################################################################
    ID_list=[]
    priority_list=[]
    for line in data:
        id=line[0]
        if id not in ID_list:
            ID_list.append(id)
            priority_list.append(line[1])
    priority_list=np.array(priority_list)
    total=len(priority_list)   
    [i for i in range(10)]
    print('Patient Priority: discrete distribution')
    print('ValueList=(1,2,3,4,5), ProbabilityList=({}, {}, {}, {}, {})'.format(*[np.sum(priority_list==str(i))/total for i in range(1,6)]))
    print('##############################################################################')
    ####################################################################################
    # working out walkin and ambulance ratio for priority patient 1 and 2
    ####################################################################################
    AmbulanceNum=0
    WalkinNum=0
    for id in id_list:
        patient=df.loc[df['ID']==id]
        patient_event=patient['Event'].iloc[0:3].to_numpy()                                           
        if 'patient-arrive' in patient_event and 'ED.wait-consultation-start' in patient_event:
            AmbulanceNum+=1
        elif 'patient-arrive' in patient_event and 'ED.wait-triage-start' in patient_event:
            WalkinNum+=1
    print('walkin and ambulance ratio for priority patient 1 and 2')
    print('(Walkin, Ambulance)=({}, {})'.format(WalkinNum/(WalkinNum+AmbulanceNum),AmbulanceNum/(WalkinNum+AmbulanceNum)))
    print('##############################################################################')

plt.show()
