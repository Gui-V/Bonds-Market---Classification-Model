#%%
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import pickle
import datetime as dt

# %%
os.listdir()

# %% Lê o ficheiro pickle com a DB Trades
trades_ds=pd.read_pickle("./trades_comp.pkl",compression='gzip')
trades_ds

# %%
#Apaga colunas sem interesse
trades_ds.drop('DEAL_ID', axis=1, inplace=True)
trades_ds.drop('PrimSec', axis=1, inplace=True)
trades_ds.drop('SYS_ID', axis=1, inplace=True)

#del trades_ds['DEAL_ID']
#%% Escolhe apenas Obrigações do Tesouro (OTs) -> ['SecType'] == 1

trades_ds=trades_ds.loc[(trades_ds['CTP_ID'] > 4) & (trades_ds['SecType'] == 1),:].reset_index(drop=True)
trades_ds.drop('CTP_ID', axis=1, inplace=True)
trades_ds.drop('SecType', axis=1, inplace=True)

#Converts Buy in Sell and Sell in Buy
trades_ds['Quantity']=np.where(trades_ds['TransType'] =='B', trades_ds['Quantity']*-1,trades_ds['Quantity'])
#Del TransType column
trades_ds.drop('TransType', axis=1, inplace=True)

#%% New columns with DAILY DEMAND (in €) for each bond---
#Main pivot1 - Bonds (Name and €)
pivot1=trades_ds.pivot_table(index='TradeDate',columns=['Security',],values='Quantity',aggfunc=sum)

#Drop day 26/12/2016 - day without transactions
pivot1.drop(pd.to_datetime('2016-12-26'), inplace=True)

#Converte Nan em 0
pivot1.fillna(0,inplace=True)

#1st, 2nd and 3rd max demand (bond name)
pivot1_aux=pd.DataFrame(pivot1.columns.values[np.argsort(-pivot1.values, axis=1)],index=pivot1.index,columns = [(counter+1) for counter, value in enumerate(pivot1.columns)]  )
pivot1['MaxDemandBond']=pivot1_aux.iloc[:,0]
pivot1['MaxDemandBond2']=pivot1_aux.iloc[:,1]

#1st, 2nd and 3rd max demand (€)
pivot1_aux2=pd.DataFrame(-np.sort(-pivot1.iloc[:,:-(len(pivot1.columns)-23)].values)[:,:3],index=pivot1.index, columns=['1st-largest','2nd-largest','3rd-largest'])
pivot1['MaxDemandBond_Eur']=pivot1_aux2.iloc[:,0]
pivot1['MaxDemand_Bond2_Eur']=pivot1_aux2.iloc[:,1]

#%% Códigos úteis
#Aula de dúvidas Prof Nuno António - 24/03/2020
pivot1['MaxDemandBond'].value_counts()
pivot1['MaxDemandBond2'].value_counts()

#Código útil por condição
df_new = df.drop(df[(df['col_1'] == 1.0) & (df['col_2'] == 0.0)].index)

#Código útil - Ordena o ranking de bonds mais procuradas
aaa=pd.DataFrame(pivot1_aux.columns[np.argsort(pivot1_aux.values)], pivot1_aux.index, np.unique(pivot1_aux.values))

#Código útil - Cria uma matriz de encoding apenas para a bond mais procurada
pd.get_dummies(pivot1['MaxDemandBond'])

#Código útil - Varre toda a DF e elimina da 3a + procurada em diante
aaa.applymap(lambda x: 100 if x>2 else x)

#Codigo útil
[int(s) for s in trades_ds.columns[5].split() if s.isdigit()]
trades_ds.columns[5].findall('\d+', s)
trades_ds.columns[5].extract('(\d+)')

#Codigo útil - filtrar por condiçoes
trades_ds.loc[(trades_ds.Maturity_years=='BondPT_10y')&(trades_ds.TradeDate=='2020-01-30'), 'Quantity'].sum()

#Código útil - Somar por linha (:28 é para somar excluindo as últimas 28)
pivot1.iloc[:,:-28].sum(axis=1)

#Código útil - Scaling
sel_feature = ['P/E','Debt','Revenue'] # Select features
X1 = df1[sel_feature].values
Y1 = df1['math score'].values
Y1 = Y1.flatten()
X_scale1 = scale(X1)

#Código útil -Score features - Scikit-Learn’s SelectKBest 
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X1,Y1)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print 5 best features

#%% New columns with COUNTRY DEMAND for each bond---
#Main pivot 2 - Countrys (Name and Freq)
pivot2=trades_ds.pivot_table(index=['TradeDate'],columns=['COUNTRY_ID','Security'],values='Quantity', aggfunc='sum', fill_value=0)

#1st, 2nd and 3rd max demand (Country name)
pivot2_aux2=pd.DataFrame(pivot2.columns.values[np.argsort(-pivot2.values, axis=1)],index=pivot2.index)

pivot1['Country_w/_Fst_most_demand_(Name)']=pivot2_aux2.iloc[:,0]

pivot1['Country_w/_Fst_most_demand_(Name)'] = pd.DataFrame(pivot1['Country_w/_Fst_most_demand_(Name)'].tolist(), index=pivot1.index)

#Analysis
#First - Country
pivot1['Country_w/_Fst_most_demand_(Name)'].value_counts()
pivot1['Country_w/_Fst_most_demand_(Name)'].value_counts().plot(kind='bar')

#%% Maturities to pivot1

trades_aux=trades_ds.groupby(['TradeDate','Security','Maturity'], as_index=False)['Quantity'].sum()

maturity=[]
for day in range(len(pivot1)):
    maturity.append(trades_aux[(pivot1.index[day]==trades_aux['TradeDate'])&(pivot1.MaxDemandBond[day]==trades_aux['Security'])].Maturity.values)
pivot1['Maturity']=pd.DataFrame(maturity,index=pivot1.index)

#pivot1.Maturity.value_counts()


#%% Create Bond Categories
import math
trades_ds['Maturity_years']=pd.DataFrame(np.zeros((len(trades_ds),1)))
trades_ds.loc[:,'Maturity_years']=trades_ds.apply(lambda x: 'BondPT_'+str(math.ceil(x.Maturity))+'y', axis=1)

#check_ruido=pd.DataFrame(-np.sort(-cat_bonds.iloc[:,:(len(cat_bonds.columns))].values),index=cat_bonds.index)

#1st, 2nd and 3rd max demand (bond name) for each maturity
cat_bonds=trades_ds.pivot_table(index='TradeDate',columns=['Maturity_years',],values='Quantity',aggfunc=sum)
pivot_cat_bonds=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.values, axis=1)],index=cat_bonds.index,columns = [(counter+1) for counter, value in enumerate(cat_bonds.columns)]  )
cat_bonds['MaxDemandBond']=pivot_cat_bonds.iloc[:,0]

#Create avg daily maturity
""" cat_bonds.loc[:,'somapositiva']=cat_bonds.iloc[:,:-1].where(cat_bonds.iloc[:,:-1]>0).sum(axis=1)
bb=cat_bonds.iloc[:,:].where(cat_bonds.iloc[:,:]>0)
bb=pd.DataFrame(-np.sort(-bb.values),index=bb.index).fillna(0)
bb=bb.div(cat_bonds.loc[:,'somapositiva'], axis=0)
aa=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.iloc[:,:].where(cat_bonds.iloc[:,:]>0).values, axis=1)],index=bb.index)
aaa=aa.apply(lambda x: x.str.extract('(\d+)', expand = False).astype(int))

pivot1['MatMed']=pd.DataFrame(np.sum(aaa.values*bb.values,axis=1), index=bb.index).set_index(bb.index)
pivot1['MatMed']=np.ceil(pivot1['MatMed'])

pivot1['MatMed'].value_counts() """

#%% Category Merge (0-1y; 5-6y; 7-8y; 11-30y)

cat_bonds['BondPT_0_1y']=cat_bonds['BondPT_0y'].add(cat_bonds['BondPT_1y'], fill_value=0)
cat_bonds.drop('BondPT_0y', axis=1, inplace=True)
cat_bonds.drop('BondPT_1y', axis=1, inplace=True)

cat_bonds['BondPT_5_6y']=cat_bonds['BondPT_5y'].add(cat_bonds['BondPT_6y'], fill_value=0)
cat_bonds.drop('BondPT_5y', axis=1, inplace=True)
cat_bonds.drop('BondPT_6y', axis=1, inplace=True)

cat_bonds['BondPT_7_8y']=cat_bonds['BondPT_7y'].add(cat_bonds['BondPT_8y'], fill_value=0)
cat_bonds.drop('BondPT_7y', axis=1, inplace=True)
cat_bonds.drop('BondPT_8y', axis=1, inplace=True)


cat_bonds['BondPT_11_31']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(11,32):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_11_31']=cat_bonds['BondPT_11_31'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)
        #cat_bonds['BondPT_'+str(17)+'y']+cat_bonds['BondPT_11_30']

cat_bonds.drop('MaxDemandBond', axis=1, inplace=True)

pivot_cat_bonds=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.values, axis=1)],index=cat_bonds.index,)

cat_bonds['Average']=cat_bonds.mean(axis=1)
cat_bonds['Max']=cat_bonds.max(axis=1)

cat_bonds['MaxDemandBond']=pivot_cat_bonds.iloc[:,0]

#%% Category Merge (6-14y; 15-30y)

cat_bonds['BondPT_0_14y']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(0,15):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_0_14y']=cat_bonds['BondPT_0_14y'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)

cat_bonds['BondPT_15_31y']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(15,32):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_15_31y']=cat_bonds['BondPT_15_31y'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)
cat_bonds.drop('MaxDemandBond', axis=1, inplace=True)

pivot_cat_bonds=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.values, axis=1)],index=cat_bonds.index,)
cat_bonds['MaxDemandBond']=pivot_cat_bonds.iloc[:,0]

#%% 3 Categorias (0-5; 6-14; 15-31)

cat_bonds['BondPT_0_5y']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(0,6):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_0_5y']=cat_bonds['BondPT_0_5y'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)

cat_bonds['BondPT_6_14y']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(6,15):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_6_14y']=cat_bonds['BondPT_6_14y'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)

cat_bonds['BondPT_15_31y']=pd.DataFrame(np.zeros((len(cat_bonds),1)))
for col in range(15,32):
    if ('BondPT_'+str(col)+'y') in cat_bonds:
        print(col)
        cat_bonds['BondPT_15_31y']=cat_bonds['BondPT_15_31y'].add(cat_bonds['BondPT_'+str(col)+'y'], fill_value=0)
        cat_bonds.drop(('BondPT_'+str(col)+'y'), axis=1, inplace=True)
cat_bonds.drop('MaxDemandBond', axis=1, inplace=True)

pivot_cat_bonds=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.values, axis=1)],index=cat_bonds.index,)
cat_bonds['MaxDemandBond']=pivot_cat_bonds.iloc[:,0]

#%%
from datetime import datetime, date, timedelta
from pandas.tseries.offsets import BDay

cat_bonds.reset_index(drop=False, inplace=True)

#Atention to number 3
asas=cat_bonds.iloc[:,:-1].melt(id_vars=['TradeDate'],var_name="Mat",value_name="Value")
asas=asas.sort_values(by='TradeDate').reset_index(drop=True)

today=asas["TradeDate"].min()+BDay(120)

#Buil for 1st day 30-11-2013
#(asas["TradeDate"].max()-asas["TradeDate"].min())-BDay(60)
count120=[]
count60=[]
count30=[]
count7=[]
count1=[]
count0=[]

#Attention. Don't run - It will take 10 min to run. Read Pickle file below instead.
asas['TradeDate']=pd.to_datetime(asas['TradeDate'], format ='%Y-%m-%d')
for day in range(len(asas["TradeDate"].dt.normalize().unique())):


    while  today not in set(asas.TradeDate):
            today=today+BDay(1)
            print('while')        

    #Last 120 days - Counts for each maturity
    day120=today-BDay(120)
    
    for ele in range(3):
        
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day120),'MaxDemandBond'].value_counts():
            count120.append(cat_bonds.loc[(cat_bonds.TradeDate<today)&((cat_bonds.TradeDate>=day120)),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count120.append(0)
    asas.loc[asas.loc[asas.TradeDate==today].index, 'count120'] = count120
    count120=[]



    #Last 60 days - Counts for each maturity
    day60=today-BDay(60)
    
    for ele in range(3):
        
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day60),'MaxDemandBond'].value_counts():
            count60.append(cat_bonds.loc[(cat_bonds.TradeDate<today)&((cat_bonds.TradeDate>=day60)),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count60.append(0)
    asas.loc[asas.loc[asas.TradeDate==today].index, 'count60'] = count60
    count60=[]


    #Last 30 days - Counts for each maturity - 

    day30=today-BDay(30)

    for ele in range(3):
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day30),'MaxDemandBond'].value_counts():
            count30.append(cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day30),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count30.append(0)

    asas.loc[asas.loc[asas.TradeDate==today].index, 'count30'] = count30
    count30=[]

    #Last 7 days - Counts for each maturity - 

    day7=today-BDay(7)

    for ele in range(3):
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day7),'MaxDemandBond'].value_counts():
            count7.append(cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day7),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count7.append(0)

    asas.loc[asas.loc[asas.TradeDate==today].index, 'count7'] = count7
    count7=[]

    #Last days - Counts for each maturity - 

    day1=today-BDay(1)

    for ele in range(3):
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day1),'MaxDemandBond'].value_counts():
            count1.append(cat_bonds.loc[(cat_bonds.TradeDate<today)&(cat_bonds.TradeDate>=day1),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count1.append(0)
    asas.loc[asas.loc[asas.TradeDate==today].index, 'count1'] = count1
    count1=[]

#Today - Counts for each maturity - 



    for ele in range(3):
        if (asas.loc[asas.TradeDate==today,'Mat'].reset_index(drop=True)[ele]) in \
            cat_bonds.loc[(cat_bonds.TradeDate==today),'MaxDemandBond'].value_counts():
            count0.append(cat_bonds.loc[(cat_bonds.TradeDate==today),'MaxDemandBond'].value_counts()[asas.loc[asas.TradeDate==\
                today,'Mat'].reset_index(drop=True)[ele]])
        else:
            count0.append(0)
    asas.loc[asas.loc[asas.TradeDate==today].index, 'count0'] = count0
    count0=[]
    
    print(today)
    today=today+BDay(1)
    if today >asas["TradeDate"].max():
        break

#Picke file
#Save to pickle inputs table
#asas.to_pickle("./asas3cat_comp.pkl",compression='gzip')

#Read pickle table with inputs
asas=pd.read_pickle("./asas3cat_comp.pkl",compression='gzip')

#asas.loc[:,['TradeDate','Mat','Value','count120','count60','count30','count7','count1','count0']].to_excel("Counts_3bucks.xlsx",sheet_name='paraCarol')
#cat_bonds.loc[:,['TradeDate','MaxDemandBond']].to_excel("3cats.xlsx",sheet_name='paraCarol')


asas['Mat'] = asas['Mat'].map({'BondPT_0_5y': 0,'BondPT_6_14y': 1, 'BondPT_15_31y': 2})
asas['Mat'] = asas['Mat'].map({0:'BondPT_0_5y', 1:'BondPT_6_14y', 2:'BondPT_15_31y'})
asas.Value.fillna(0, inplace=True)

asas['Mat0_5']=pd.DataFrame(np.zeros((len(asas),1)))
asas['Mat6_14']=pd.DataFrame(np.zeros((len(asas),1)))
asas['Mat15_31']=pd.DataFrame(np.zeros((len(asas),1)))
asas['Day']=asas['TradeDate'].dt.day
asas['Month']=asas['TradeDate'].dt.month

for item in asas.TradeDate.unique():
    for i in range(2):
        
        if asas.loc[asas.loc[asas.TradeDate==item].index[i],'Mat0_14']==1:
            sum1=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count60','count30','count7',\
                'count1']].sum()
            lag160=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count60']]
            lag130=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count30']]
            lag17=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count7']]
            lag11=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count1']]

        elif asas.loc[asas.loc[asas.TradeDate==item].index[i],'Mat0_14']==0:
            sum0=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count60','count30','count7',\
                'count1']].sum()
            lag060=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count60']]
            lag030=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count30']]
            lag07=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count7']]
            lag01=asas.loc[asas.loc[asas.TradeDate==item].index[i],['count1']]

    dif=sum1-sum0
    dif_lag60=lag160-lag060
    dif_lag30=lag130-lag030
    dif_lag7=lag17-lag07
    dif_lag1=lag11-lag01

    if (math.isnan(dif_lag60) and math.isnan(dif_lag30) and math.isnan(dif_lag7) and math.isnan(dif_lag1))==False:
        asas.loc[asas.loc[asas.TradeDate==item].index, 'count60_dif'] = int(dif_lag60)
        asas.loc[asas.loc[asas.TradeDate==item].index, 'count30_dif'] = int(dif_lag30)
        asas.loc[asas.loc[asas.TradeDate==item].index, 'count7_dif'] = int(dif_lag7)
        asas.loc[asas.loc[asas.TradeDate==item].index, 'count1_dif'] = int(dif_lag1)

    asas.loc[asas.loc[asas.TradeDate==item].index, 'Dif'] = dif
    """ if dif>0:
        asas.loc[asas.loc[asas.TradeDate==item].index, 'Dif'] = 1
    elif dif<0:
        asas.loc[asas.loc[asas.TradeDate==item].index, 'Dif'] = -1
    elif dif==0:
        asas.loc[asas.loc[asas.TradeDate==item].index, 'Dif'] = 0 """

asas_log = asas.dropna().copy(deep=True)


#asas_log.drop(['count60','count30','count7','count1','Mat15_31'], axis=1, inplace=True)

#asas_log=asas_log.loc[asas_log['TradeDate'].duplicated()]

#X.to_excel("Dataset_12Abril.xlsx",sheet_name='paraCarol')


for y in range(len(asas)):
    if asas.Mat[y]=='BondPT_0_14y':
        asas['Mat0_14'][y]=1
    elif asas.Mat[y]=='BondPT_15_31y':
        asas['Mat15_31'][y]=1



#%% Previous Analysis
cat_bonds['MaxDemandBond'].value_counts()
cat_bonds['MaxDemandBond'].value_counts().plot(kind='bar')
plt.xlabel('Maturidades', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title('Target - Todas as Classes de Maturidades das Bonds',fontsize=18)
plt.show()

city=['Delhi','Beijing','Washington','Tokyo','Moscow']
pos = np.arange(len(city))
Happiness_Index=[60,40,70,65,85]
 
plt.bar(cat_bonds['MaxDemandBond'].value_counts()[0],cat_bonds['MaxDemandBond'].value_counts(),color='blue',edgecolor='black')
plt.xticks(pos, city)
plt.xlabel('City', fontsize=16)
plt.ylabel('Happiness_Index', fontsize=16)
plt.title('Barchart - Happiness index across cities',fontsize=20)
plt.show()














#Piechart
labels = cat_bonds['MaxDemandBond'].value_counts().index
sizes = cat_bonds['MaxDemandBond'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#%%
#Memory clean

del ax1;del col;del day;del fig1;del labels; del maturity;
del pivot2; del pivot2_aux2; del pivot_cat_bonds; del trades_aux; del sizes

#%% CATEGORIZAR VARIAVEIS CATEGORICAS


#%% #Import DB with inputs

#First remove non essencial columns from pivot1 (output table)
pivot1.drop(pivot1.columns[0:23],axis=1, inplace=True)

#Pivot1 new max demand bond column from categories created in cat_bonds
pivot1.loc[:,'MaxDemandBond_nova']=cat_bonds.loc[:,'MaxDemandBond']

#Read excel table with inputs
#inputs = pd.read_excel('input2.xlsx')

#Save to pickle inputs table
#inputs.to_pickle("./dbtx_comp.pkl",compression='gzip')

#Read pickle table with inputs
inputs=pd.read_pickle("./dbtx_comp.pkl",compression='gzip')

#Drop empty column: Unnamed: 12 column
inputs.drop('Unnamed: 12',axis=1, inplace=True)

#Agreggate data to pivot1 DF
for col in inputs.columns:
    print(col)
    if col != 'Datas':
        pivot1[col]=inputs.set_index(['Datas']).loc[:,col]



#Procura por MTS

pivot1.iloc[:,34:-4].sum(axis=1)


cat_mts=pivot.pivot_table(index='TradeDate',columns=['Maturity_years',],values='Quantity',aggfunc=sum)
pivot_cat_bonds=pd.DataFrame(cat_bonds.columns.values[np.argsort(-cat_bonds.values, axis=1)],index=cat_bonds.index,columns = [(counter+1) for counter, value in enumerate(cat_bonds.columns)]  )
cat_bonds['MaxDemandBond']=pivot_cat_bonds.iloc[:,0]















































#Delete holidays and empty rows
pivot1.drop(pd.to_datetime(['2014-04-25','2014-05-01','2014-12-24','2014-12-31','2015-02-01',\
    '2015-05-25','2015-07-06','2015-12-31','2017-05-01','2018-07-08','2018-07-22'\
        ,'2019-05-01','2019-07-14','2019-07-20','2019-12-31']), inplace=True)

plt.plot(cat_bonds['2013'].BondPT_2y)


cat_bonds.BondPT_0y.diff()

cat_bonds['MaxDemand2y']=(pivot1['MaxDemandBond_nova']=='BondPT_2y').astype(int)

cat_bonds['Corr']=cat_bonds.BondPT_10y['2017'].corr(cat_bonds['MaxDemand10y']['2017'])

cat_bonds.MaxDemandBond.value_counts()

cat_bonds.loc[(cat_bonds.MaxDemand10y==1), 'BondPT_10y'].plot()
cat_bonds.loc[(cat_bonds.MaxDemand9y==0), 'BondPT_9y'].plot()

sns.boxplot(cat_bonds.loc[(cat_bonds.MaxDemand10y==1), 'BondPT_10y'])

#Remover outliers acima de 2.0
sns.boxplot(cat_bonds.loc[(cat_bonds.MaxDemand10y==1)&(cat_bonds.BondPT_10y<2E8), 'BondPT_10y'])
(cat_bonds.loc[(cat_bonds.MaxDemand10y==1), 'BondPT_10y']['2017']).plot()
(cat_bonds.loc[(cat_bonds.MaxDemand10y==1)&(cat_bonds.BondPT_10y<2E8), 'BondPT_10y']).mean()

#To the first bucket

cat_bonds['MaxDemand1']=(pivot1['MaxDemandBond_nova']=='BondPT_0_5y').astype(int)

from sklearn import preprocessing

# separate the data and target attributes
X = pivot1[['Var SnP']]
# standardize the data attributes
pivot1['Var SnP'] = preprocessing.scale(X)

pivot1['MaxDemand1']=(pivot1['MaxDemandBond_nova']=='BondPT_0_5y').astype(int)

taxas=pd.read_excel('Taxas absolutas.xlsx')


#Taxas
pivot1['taxa2y']=taxas.set_index('Datas').loc[:,'PT_2y']
pivot1['procura2y']=cat_bonds.loc[:,'BondPT_2y']
pivot1.taxa2y.corr(pivot1['txSpread vs GE_2y'])

cat_bonds.loc[:,'BondPT_5y'].corr(pivot1['taxa5y'])
cat_bonds.loc[(cat_bonds.MaxDemand1==0), 'BondPT_0_5y'].mean()

pivot1.loc[(pivot1.MaxDemand1==1), 'taxa5y'].plot()
cat_bonds.loc[cat_bonds.MaxDemand1==1, 'BondPT_5y'].plot()

#Chart - Taxa vs Vol
import matplotlib as mpl
plt.style.use('ggplot')

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(15,8))
ax[0].plot(pivot1['2014'].index, pivot1['2014'].taxa2y)
ax[1].bar(pivot1['2014'].index, pivot1['2014'].procura2y, width=1)
###


procura2y.PT_2y.plot()
taxas.set_index('Datas')['2013'].PT_5y.plot()
pivot1['2013'].VIX.plot()

pivot1['2013'].VIX.diff().corr(taxas.set_index('Datas')['2013'].PT_5y.diff())

cat_bonds.loc[:,'BondPT_10y']['2019'].hist(bins=50)
cat_bonds.loc[:,'BondPT_10y'].sum()
cat_bonds.loc[:,'BondPT_10y'].mean()
cat_bonds.loc[:,'MaxDemandBond'].value_counts()

pivot1[variaveis_plot].hist(bins=20, figsize=(20, 30), layout=(6, 6), xlabelsize=8, ylabelsize=8);



pivot1.loc[(pivot1.MaxDemand1==1), 'Mês'].plot()
pivot1.loc[(pivot1.MaxDemand1==0), 'MaxDemand1'].plot()


corr = pivot1[['MaxDemandBond_Eur','Trimestre','Spread vs GE_2y-30y','txSpread vs GE_2y-30y',\
    'txSpread vs GE_2y-30y_bi', 'Spead2y10y','txSpead2y10y', 'txSpead2y10y_bi', \
        'PSI20', 'Var SnP','Var PSI20', 'Ouro', 'OIL', 'PIB', 'Rating', 'Desemprego',\
            'txSpread vs GE_2y', 'txSpread vs GE_5y', 'txSpread vs GE_10y',\
                'txSpread vs GE_30y', 'txSpread2y5y', 'txSpread2y10y', 'txSpead5y10y','Pressao mercado']].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

f, ax = plt.subplots(figsize=(30, 30))
heatmap = sns.heatmap(corr,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 6})

#add the column names as labels
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns, rotation = 45)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

#%% New MTS columns (MTS1) with total demand for each maturity

#Maturity: 0-5 year
pivot1['MTS1']=pivot1['MTS_0-1y'].add(pivot1['MTS_2y'], fill_value=0).add(pivot1['MTS_3y'], \
    fill_value=0).add(pivot1['MTS_4y'], fill_value=0).add(pivot1['MTS_5y'], fill_value=0)


#Maturity: 6-10 year - alterar para 14 no excel
pivot1['MTS2']=inputs.set_index('Datas')['MTS_6y'].add(inputs['MTS_7y'],fill_value=0).add(inputs['MTS_8y'],fill_value=0).add(inputs['MTS_9y'],\
     fill_value=0).add(inputs['MTS_10y'], fill_value=0)

#Maturity: 11-30 year - alterar para 15 a 30 no excel
pivot1['MTS3']=inputs.set_index('Datas')['MTS_11-30y']

#passo 2 deletar os dias em q 0a5 n tem procura
pivot1['BondPT_0_5y']=cat_bonds['BondPT_0_5y']


piii=pivot1.drop(pivot1[pivot1['Maturity'] > 6 ].index)




plt.plot(piii['MTS1']);

piii['BondPT_0_5y'].corr(piii['MTS1'])


corr = piii[['BondPT_0_5y','MTS1']].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

f, ax = plt.subplots(figsize=(11, 15))
heatmap = sns.heatmap(corr,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 16})

# Transform a list of columns to categorical
cols = ['txSpread vs GE_2y-30y_bi', 'txSpead2y10y_bi', 'Rating','txSpread vs GE_2y','txSpread vs GE_5y',\
    'txSpread vs GE_10y','txSpread vs GE_30y','txSpread2y5y','txSpread2y10y','txSpead5y10y','Unnamed: 42',\
         'Pressao mercado','MaxDemandBond_nova']
pivot1[cols] = pivot1[cols].apply(lambda x:x.astype('category'))

# Check for missing values
print(pivot1.isnull().sum())

#Temporaily remove 4 first columns from pivot1 (output table)
pivot1 = pivot1.iloc[4:]

# Ckeck the top counts of all categorical variables
categorical=pivot1.select_dtypes(exclude=["number","bool_","object_"]).columns.tolist()
categorical
for var in pivot1[categorical]:
    print(var,":\n",pivot1[var].value_counts(), sep="")


#Analyze data distribution to fill the gaps

# Histograms on all numeric variables
numerical=pivot1.select_dtypes(include=[np.number]).columns.tolist()
pivot1[numerical].hist(bins=30, figsize=(10, 10), layout=(2, 2), xlabelsize=8, ylabelsize=8);

# Density Plot and Histogram
sns.distplot(pivot1['VIX'], hist=True, kde=True, rug=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})



#pivot1.to_pickle("./pivot1_comp.pkl",compression='gzip')
#pivot1.to_pickle("./pivot1_2classes_comp.pkl",compression='gzip')

pivot1=pd.read_pickle("./pivot1_comp.pkl",compression='gzip')
#pivot1=pd.read_pickle("./pivot1_2classes_comp.pkl",compression='gzip')
pivot1.info()

Bond4anos=pivot1.loc[(pivot1.MaxDemandBond_nova=='BondPT_4y'), :]
Bond11_30anos=pivot1.loc[(pivot1.MaxDemandBond_nova=='BondPT_11_30'), :]

#pip install plotly
#import plotly.express as px
#fig = px.scatter(Bond4anos, x="Mês", y="MaxDemandBond_nova")
#fig.show()

# Bar Plots . Mês
tb1 = pd.crosstab(index=pivot1['Mês'], columns=cat_bonds['MaxDemandBond'])
tb1.plot(kind="bar", figsize=(20,12), stacked=True)
plt.legend(bbox_to_anchor=(1, 1))
# Bar Plots . Trimestre
tb1 = pd.crosstab(index=pivot1['Trimestre'], columns=pivot1['MaxDemandBond_nova'])
tb1.plot(kind="bar", figsize=(18,8), stacked=True)

# Bar Plots . Rating
tb1 = pd.crosstab(index=pivot1['Rating'], columns=cat_bonds['MaxDemandBond'])
tb1.plot(kind="bar", figsize=(20,12), stacked=True)
plt.legend(bbox_to_anchor=(1, 1))

tb1 = pd.crosstab(index=pivot1['Rating'], columns=pivot1['MaxDemandBond_nova'])
tb1.plot(kind="bar", figsize=(20,8), stacked=True)
plt.legend(bbox_to_anchor=(1, 1))

#Scatter Plots
sns.set(style="whitegrid")
sns.stripplot(x=Bond11_30anos["PSI20"])
sns.stripplot(x="Ouro", y="MaxDemandBond_nova", data=pivot1)


sns.pairplot(pivot1, x_vars=['Ouro', 'Rating'], y_vars=['MaxDemandBond_nova'])
sns.pairplot(df, x_vars=cols[4:], y_vars=['avgvol'])

pivot1[['Spread vs GE_2y', 'MaxDemandBond_nova']].hist()
plt.show()


pivot2019['MaxDemandBond_nova'] = pivot2019['MaxDemandBond_nova'].map({'BondPT_0_7y': 0,'BondPT_8_12y': 1, 'BondPT_13_30y': 2})
pivot1['MaxDemandBond_nova'] = pivot1['MaxDemandBond_nova'].map({'BondPT_0_5y': 0,'BondPT_6_14y': 1, 'BondPT_15_30y': 2})
pivot1['MaxDemandBond_nova'] = pivot1['MaxDemandBond_nova'].map({'BondPT_6_14y': 0,'BondPT_15_30y': 1})

pivot1=pivot1.dropna()
pivot120=pivot1['2013']
sns.pairplot(asas[['Mat','count60','count30','count7','count1','count0']],hue='count0')

#Data understanding - MODELO1 - Final Dataset

#modelo = pd.read_excel('Modelo1.xlsx', sheet_name='Modelo6a')
# Load scikit's random forest classifier library
modelo1 = modelo.dropna().copy(deep=True)

#Info
modelo1[['PT_2Y', 'PT_5Y','PT_10Y', 'PT_30Y', 'PTGE_2Y',\
    'PTGE_5Y', 'PTGE_5Y.1', 'PTGE_5Y.2','PTGE_D_2Y', 'PTGE_D_5Y','PTGE_D_10Y', 'PTGE_D_30Y', 'SnP',\
        'VIX','PSI20', 'Ouro', 'OIL', 'EURUSD', 'PIB', 'Rating', 'Desemprego','Count60_C1-C2',\
            'Count60_C2-C3','Count63_C2-C3', 'Count30_C1-C3','Count60_C2-C3.1','Euribor_1w', 'Euribor_1m',\
                'Euribor_3m','Euribor_6m', 'Euribor_12m', 'Pressao', 'Inf_5Y_US', 'Inf_5Y_EU','SnP_vol',\
                    'NDQ_vol', 'SPX INDEX_H-L', 'NDQ INDEX_H-L','EUORDEPO INDEX', 'EUBCI INDEX',\
                        'GRWCCURR INDEX', 'GRECBCLI INDEX','GRECBEXP INDEX', 'GRWCBEXP INDEX', 'EUAR84FI INDEX',\
                            'EURR002W INDEX','ECBLDEPO INDEX', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y',\
                                '10Y', '15Y', 'Longo', 'Muito longo']].info().tail(15)

#Summary statistics for all variables
pd.set_option('display.float_format', lambda x: '%.3f' % x)
summary=modelo1[['VIX','SPX INDEX_H-L','Pressao','Inf_5Y_EU','PTGE_D_10Y','NDQ_vol']].describe()
summary=summary.transpose()
summary.head(len(summary))

#Histograms
#Hist
variaveis_plot=modelo1.columns.tolist()
#pivot1[numerical].hist(bins=20, figsize=(10, 10), layout=(40, 40), xlabelsize=8, ylabelsize=8);
modelo1[variaveis_plot].hist(bins=20, figsize=(20, 30), layout=(8, 8), xlabelsize=8, ylabelsize=8);

#Correlation
corr = modelo1[['PT_10Y', 'PT_30Y', 'PTGE_2Y',\
    'PTGE_5Y', 'PTGE_5Y.1','PTGE_D_2Y', 'PTGE_D_5Y','PTGE_D_10Y', 'PTGE_D_30Y', 'SnP',\
        'VIX','PSI20', 'Ouro', 'OIL', 'EURUSD', 'PIB', 'Rating', 'Desemprego','Count60_C1-C2',\
            'Count60_C2-C3','Euribor_6m','Pressao','Inf_5Y_US', 'Inf_5Y_EU','SnP_vol','NDQ_vol',\
                'SPX INDEX_H-L', 'NDQ INDEX_H-L','EUORDEPO INDEX','EUBCI INDEX','GRWCCURR INDEX',\
                    'GRECBCLI INDEX','GRECBEXP INDEX', 'GRWCBEXP INDEX', 'EUAR84FI INDEX',\
                            'EURR002W INDEX','ECBLDEPO INDEX','10Y', '15Y', 'Muito longo']].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

f, ax = plt.subplots(figsize=(30, 30))
heatmap = sns.heatmap(corr,
                      mask = mask,
                      square = True,
                      linewidths = .5,
                      cmap = 'coolwarm',
                      cbar_kws = {'shrink': .4,
                                'ticks' : [-1, -.5, 0, 0.5, 1]},
                      vmin = -1,
                      vmax = 1,
                      annot = True,
                      annot_kws = {'size': 6})

#add the column names as labels
ax.set_yticklabels(corr.columns, rotation = 0)
ax.set_xticklabels(corr.columns, rotation = 45)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

#Pairplots
modelo12=modelo1.loc[modelo1.Datas.dt.year==2018,:]
sns.pairplot(modelo12[['OIL','NDQ_vol','MaxDemand']],hue='MaxDemand')


#%%

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb  
from xgboost import  XGBClassifier  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

#modelo1 = pd.read_excel('Modelo1.xlsx', sheet_name='Modelo6a')

# Load scikit's random forest classifier library
X = modelo1.dropna().copy(deep=True)

#X=X[['MaxDemandBond_nova','PTGE_5Y_N','PTGE_10Y_N','PTGE_30Y_N','PTGE_D_5Y_S100_ln','PTGE_D_10Y_clean','PTGE_D_30Y_clean',\
    #'PT_D_2Y5Y_clean','PT_D_2Y10Y_VA1d_N','PT_5Y10Y_S100_ln','PT_5Y30Y_S100_ln','PT_10Y30Y_N','Mês','Ano']]
#X['MaxDemandBond_nova'] = X['MaxDemandBond_nova'].map({'BondPT_0_7y': 1, 'BondPT_8_30y': 0})
#X=X.drop(['Datas'],1)

#Delete dates with less than 7 Bonds demanded

X.drop(X[(X['Datas'] == '2014-06-17') & (X['Datas'] == '2014-08-15')&(X['Datas'] == '2016-08-15')&\
    (X['Datas'] == '2016-08-15')].index, inplace=True)

#X_train=X_train[['Into column name here']]


y = X['MaxDemand']#.map({'BondPT_0_5y': 0, 'BondPT_6_14y': 1, 'BondPT_15_31y': 2})

#Move MaxDemand (output) to first column position
cols = list(modelo1)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('MaxDemand')))
modelo1 = modelo1.loc[:, cols]

modelo1[['Rating']] = modelo1[['Rating']].apply(lambda x:x.astype('category'))


#TIME SERIES - Y and X, Test and Train - TRAIN AND TEST SIMPLE SPLIT
X = modelo1.loc[:, 'Datas':]
y = modelo1.loc[:, :'MaxDemand']

train_size = int(len(modelo1) * 0.83)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(X)]

X_train=X_train.drop(['Datas'],1)
X_test=X_test.drop(['Datas'],1)
#########

#Check y balance
import collections
#Y total
print(collections.Counter(y.MaxDemand))
sns.countplot(x="y", data=pd.DataFrame(data={'y':y.MaxDemand}))
#Check y balance
#y_train
print(collections.Counter(y_train.MaxDemand))
sns.countplot(x="y", data=pd.DataFrame(data={'y':y_train.MaxDemand}))



#TIME SERIES - Y and X, Test and Train - CVS
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#cv = 10
cv =TimeSeriesSplit(n_splits=3)
estimator = rf_model

plot_learning_curve(estimator, "Learning curve", X_train, y_train, cv=cv, n_jobs=4,
                  train_sizes=np.linspace(0.1, 1.0, 5))

#Scrip to validate the cross validation splits
splits = BlockingTimeSeriesSplit(n_splits=2)
index = 1
for train_index, test_index in splits.split(X):
	train = X.iloc[train_index]
	test = X.iloc[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))
	index += 1


#STRATIFIED TRAIN AND TEST SPLIT
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=123)

# Remove the Target from the training
X_train = X_train.drop(['MaxDemand'],1)
X_test = X_test.drop(['MaxDemand'],1)

""" tempDF = X.copy(deep=True)
tempDF.drop(columns='MaxDemand', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
tempDF_scaled = scaler.fit_transform(tempDF)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) """


#Start Func Def
#1 - Function to create a baseline approach
def baseline_f (input_model):
    #train and fit regression model
    model=input_model()
    model.fit(X_train, y_train)

    # predict
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # evaluate RF
    # Train and Test Accuracy
    train_acc=metrics.accuracy_score(y_train, train_preds)
    test_acc=metrics.accuracy_score(y_test, test_preds)
    print("------------------------")
    print(str(input_model))
    print(f"Training Accuracy: {(train_acc * 100):.4}%")
    print(f"Test Accuracy:     {(test_acc * 100):.4}%")

    # store accuracy in a new dataframe
    model_score = [input_model, train_acc, test_acc]
    models = pd.DataFrame([model_score])


# Do cross-validation with multiple algorithms and randomized search to select which to use
np.random.seed(123)
#cv =TimeSeriesSplit(n_splits=i+1)
#BlockingTimeSeriesSplit(n_splits=i)
#cv=5
for i in range(1,12):
    cv = BlockingTimeSeriesSplit(n_splits=i)
    algorithms = [
                    
                    {"classifier": [RandomForestClassifier()],
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth":[3,4],
                    "classifier__min_samples_leaf":[2,5,10],
                    "classifier__random_state":[123]},
                    ]
    pipe = Pipeline([("classifier", RandomForestClassifier())])
    gridSearch = GridSearchCV(pipe, algorithms, cv=cv, verbose=1, n_jobs=-1)
    best_model = gridSearch.fit(X_train, y_train)
    print("The mean accuracy of the model is:",best_model.score(X_train, y_train))
    print("The best parameters for the model are:",best_model.best_params_)
    #plot_learning_curve(best_model, "Learning curve", X_train, y_train, cv=cv, n_jobs=4,
                    #train_sizes=np.linspace(0.1, 1.0, 5))

#AUC
classModel = best_model.best_estimator_[0]
classModel.fit(X_train, y_train)

y_pred_train = classModel.predict(X_train) 
y_pred_test = classModel.predict(X_test) 
classModel.predict_proba(X_test)
probs = classModel.predict_proba(X_test)
FP,TP,thresholds = metrics.roc_curve(y_test,probs[:,1])
plt.plot(FP,TP,label="ROC")
plt.xlabel("False Positive Rate  \n(probability of false alarm)")
plt.ylabel("True Positive Rate/Sensitivity/Recall \n(probability of detection)")
cutoff=np.argmax(np.abs(TP-FP)) 
optimal_threshold = thresholds[cutoff]
plt.show()
print("AUC:{0:.3f}".format(metrics.auc(FP, TP)))
print("Optimal threshold:{0:.3f}".format(optimal_threshold))




#OU
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            #Definir a % para validação (80% neste caso)
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
model = rf_model
btscv = BlockingTimeSeriesSplit(n_splits=6)
plot_learning_curve(model, "Learning curve", X_train, y_train, cv=btscv, n_jobs=4,
                  train_sizes=np.linspace(0.1, 1.0, 5))




#OU





# Do cross-validation with multiple algorithms to select which to use
""" algorithms = [
    RandomForestClassifier(n_estimators=200, max_depth=4, random_state=123),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=123),
]
CV = TimeSeriesSplit(n_splits=5)
#cv_df = pd.DataFrame(index=range(CV * len(algorithms)))
entries = []
for algorithm in algorithms:
  algorithm_name = algorithm.__class__.__name__
  accuracies = cross_val_score(algorithm, X_train, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((algorithm_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['algorithm_name', 'fold_idx', 'accuracy'])
cv_df

fig = plt.figure(figsize=(6,4))
cv_df.groupby('algorithm_name').accuracy.mean().plot.bar(ylim=0)
plt.show() """







#2 -Grid Search
def grid_search_f(classifier, param_grid):
    #1st -
    print('Parameters currently in use:\n')
    print(classifier.get_params())
          
    param_grid=param_grid
    # instantiate the tuned random forest
    grid_search = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)

    # train the tuned random forest
    grid_search.fit(X_train, y_train)

    # print best estimator parameters found during the grid search
    print("------------------------")
    print((str(classifier) + 'Best Parameters'))
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)
    return grid_search.best_params_


#3 - Function to train and evaluate a  model with tunned parameters
#it also prints the metrics scores 
def run_model(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    # predict
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # evaluate - ROC AUC??
    #train_accuracy = roc_auc_score(y_train, train_preds)
    #test_accuracy = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    #Print Confusion Matrix and reports of the model accuracy
    print('Confusion Matrix')
    print(confusion_matrix(y_test,test_preds))
    print('Model Scores')
    print("------------------------")
    print(classification_report(y_test,test_preds))
    print("------------------------------------------------------")
    print('Classification Report : \n', report)
#End Function Def

#1- list of all classifiers to run for baseline models 
models=[LogisticRegression,RandomForestClassifier,XGBClassifier]

#running each model and print accuracy scores
for model in models:
    baseline_f (model)
""" 
Choose best 2 models from the report above to carry on.
Best train model: XGBoost
Best test model: Random Forest
 """

#2 - Grid Search

#Hyperparameter
#Grid Search -> Random Forest Classifier
param_grid_rf = {'bootstrap': [True],'n_estimators': [10, 80, 100,200,300],
                  'criterion': ['gini', 'entropy'],         
                  'max_depth': [2,5,10,15,20,None], 
                  'min_samples_leaf': [1, 10, 20, 50, 100],
                  'min_samples_split': [2, 30, 40],
                 }
#GridSearch
rf_params=grid_search_f(RandomForestClassifier(),param_grid_rf)

# Running RandomForestClassifier with best parameters
rf_model=RandomForestClassifier(bootstrap=True,n_estimators=100, 
                                  criterion= 'gini', 
                                  max_depth= 4,
                                  min_samples_leaf=2, 
                                  min_samples_split= 2)
                               
                               
run_model(rf_model, X_train, y_train,X_test, y_test)

#Hyperparameter
#Grid Search -> XGB Classifier
param_grid_xg = {'n_estimators': [100, 200, 300],
              'learning_rate': [0.05, 0.1], 
              'max_depth': [3, 5, 10],
              'colsample_bytree': [0.7, 1],
              'gamma': [0.0, 0.1, 0.2],
              'random_state':[123]
                }
grid_search_f(XGBClassifier(), param_grid_xg)

# Running XGBClassifier with best parameters
xgb_model=XGBClassifier(colsample_bytree= 1, 
                        n_estimators= 100,
                        gamma= 0.1,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=123
                        )
                                                                 
run_model(xgb_model, X_train, y_train,X_test, y_test)





#__________



#4 Feature Importance
# plot the important features - based on Random Forest
#(Methods that use ensembles of decision trees 
# can compute the relative importance of each attribute. )
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False).plot(kind='barh', figsize=(15,7))
#feat_importances.nlargest(10).sort_values().plot(kind='barh', figsize=(10,5))
#ax.set_ylabel('feature', size = 18);
plt.xlabel('Relative Feature Importance for Random Forest');
plt.title('Feature Importance Order', size = 16);


#Alternative with Feature Permutation
from sklearn import inspection
#pip install mlxtend  
import mlxtend
from mlxtend.evaluate import feature_importance_permutation

importance_vals=rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
indices = np.argsort(importance_vals)[::-1]

feature_list = list(X_train.columns)
features = list(feature_list)

ranked_index = [feature_list[i] for i in indices]
#range(X.shape[1])

# Plot the feature importances of the forest
plt.figure(figsize=(12,8))
plt.title("Random Forest feature importance")
plt.bar(range(X_train.shape[1]), importance_vals[indices],yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
plt.xticks(range(X_train.shape[1]), (ranked_index), rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.ylim([0, 0.3])
plt.show()

#feature_list = list(df_normalized_w_target.columns)[7:-1]

# Permutation Importance 

imp_vals, imp_all = feature_importance_permutation(predict_method=rf_model.predict,X=X_train.values,y=y_train.values,metric='accuracy',num_rounds=10,seed=42)
#imp_vals, imp_all

std = np.std(imp_all, axis=1)
indices = np.argsort(imp_vals)[::-1]
ranked_index_2 = [feature_list[i] for i in indices]


plt.figure(figsize=(12,8))
plt.title("Random Forest feature importance via permutation importance w. std. dev.")
plt.bar(range(X_train.shape[1]), imp_vals[indices],
        yerr=std[indices])
#plt.xticks(range(X.shape[1]), indices)
plt.xticks(range(X_train.shape[1]), (ranked_index_2), rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()


#Forward feature selection
#http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(rf_model, 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(X_train, y_train)





#Incluir plot do num de árvores na RF
#Por alterar
""" accuracy_rate = []
training_acc = []

for i in range(1,21):   
    rfc_plot = RandomForestClassifier(bootstrap=True,
                       max_depth=None, 
                       min_samples_leaf=1, min_samples_split=2,
                       n_estimators=i)
    rfc_plot.fit(X_train, y_train) 
    accuracy_rate.append(rfc_plot.score(X_test, y_test))
    training_acc.append(rfc_plot.score(X_train, y_train))
    
plt.figure(figsize=(12,8))
plt.plot(np.arange(1,21), accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.plot(np.arange(1,21), training_acc,color='red', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)

#training > testing
plt.title('Accuracy Rate vs. N-Estimators')
plt.xlabel('N-Estimators')
plt.ylabel('Accuracy Rate')
 """
#_______





#3- Random Forest Performance
#Multi Class Confusion Matrix
rfc_pred = rfc.predict(X_test)
#Print Matrix
print(confusion_matrix(y_test,rfc_pred))

#Print Metrics
print(classification_report(y_test,rfc_pred))

#INCLUIR CONFUSION MATRIX DO PROF
y_pred_train = classModel.predict(X_train) 
y_pred_test = classModel.predict(X_test) 
# Plot consusion matrix of the test set results
product_id_df = X[['product', 'product_id']].drop_duplicates().sort_values('product_id')
conf_mat = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(8,6))
ax = sns.heatmap(conf_mat, annot=True, fmt='d',
    xticklabels=product_id_df['product'].values, yticklabels=product_id_df['product'].values)
bottom, top = ax.get_ylim() # These two lines were added due to bug on current Seaborn version
ax.set_ylim(bottom + 0.5, top - 0.5) #
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
#AULA8
#XGBoost Performance




#Emsemble voting: https://scikit-learn.org/stable/modules/ensemble.html




























algorithms = [
    RandomForestClassifier(n_estimators=300, max_depth=5, random_state=123),
    
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(algorithms)))
entries = []
for algorithm in algorithms:
  algorithm_name = algorithm.__class__.__name__
  accuracies = cross_val_score(algorithm, X_train_scaled,  y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((algorithm_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['algorithm_name', 'fold_idx', 'accuracy'])
cv_df
#classModel = LogisticRegression(multi_class='multinomial',solver='newton-cg',fit_intercept=True,random_state=123)
#classModel = LogisticRegression(random_state=123)
classModel =RandomForestClassifier(n_estimators=300, max_depth=5, random_state=123)
classModel.fit(X_train_scaled, y_train)

y_pred_train = classModel.predict(X_train_scaled) 
y_pred_test = classModel.predict(X_test_scaled) 

product_id_df = X['MaxDemandBond_nova'].drop_duplicates().sort_values('MaxDemandBond_nova')
conf_mat = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(8,6))
ax = sns.heatmap(conf_mat, annot=True, fmt='d')
bottom, top = ax.get_ylim() # These two lines were added due to bug on current Seaborn version
ax.set_ylim(bottom + 0.5, top - 0.5) #
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# save the model to disk
filename = 'modelo_3cat_randfor_13-4.sav'
pickle.dump(classModel, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test_scaled, y_test)
print(result)



def performanceMetricsDF(metricsObj, yTrain, yPredTrain, yTest, yPredTest):
  measures_list = ['ACCURACY','PRECISION', 'RECALL']
  train_results = [metricsObj.accuracy_score(yTrain, yPredTrain),
                metricsObj.precision_score(yTrain, yPredTrain),
                metricsObj.recall_score(yTrain, yPredTrain)]
  test_results = [metricsObj.accuracy_score(yTest, yPredTest),
               metricsObj.precision_score(yTest, yPredTest),
               metricsObj.recall_score(yTest, yPredTest)]
  resultsDF = pd.DataFrame({'Measure': measures_list, 'Train': train_results, 'Test':test_results})
  return(resultsDF)

y_pred_train = classModel.predict(X_train_scaled) 
y_pred_test = classModel.predict(X_test_scaled) 

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels, fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    bottom, top = ax.get_ylim() # These two lines were added due to bug on current Seaborn version
    ax.set_ylim(bottom + 0.5, top - 0.5) #

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

cf = metrics.confusion_matrix(y_test,y_pred_test)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['0', '1']
make_confusion_matrix(cf, 
                      group_names=labels,
                      categories=categories, 
                      cmap='Blues')

resultsDF = performanceMetricsDF(metrics, y_train, y_pred_train, y_test, y_pred_test)
resultsDF

#%%
from sklearn.model_selection import learning_curve
from sklearn import tree

X = asas_log.dropna().copy(deep=True)

X=X[['Dif','count60_dif','count30_dif','count7_dif','count1_dif','count0','Mat0_14']]
#X['MaxDemandBond_nova'] = X['MaxDemandBond_nova'].map({'BondPT_0_7y': 1, 'BondPT_8_30y': 0})

y = X['count0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=123)

# Remove the Target from the training
X_train = X_train.drop(['count0'],1)
X_test = X_test.drop(['count0'],1)

tempDF = X.copy(deep=True)
tempDF.drop(columns='count0', inplace=True)




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



cv = 10
estimator = tree.DecisionTreeRegressor(random_state=123, max_depth=2, min_samples_leaf=3)
plot_learning_curve(estimator, "Learning curve", X_train, y_train, ylim=(-5, 5), cv=cv, n_jobs=4,
                  train_sizes=np.linspace(0.1, 1.0, 5))

dt_regr = tree.DecisionTreeRegressor(random_state=123, max_depth=2, min_samples_leaf=3)
dt_regr.fit(X_train, y_train)



formattedList = [float(format(member,'.6f')) for member in dt_regr.feature_importances_]
formattedList2 = [abs(float(format(member,'.6f'))) for member in dt_regr.feature_importances_]
data_tuples = list(zip(X.columns,formattedList,formattedList2))
coeff_df = pd.DataFrame(data=data_tuples, columns=['Feature','Coefficient','AbsCoefficient'])
coeff_df.reset_index(drop=True, inplace=True)
coeff_df.sort_values(by=['AbsCoefficient'], inplace=True, ascending=False)
coeff_df


y_pred_train = dt_regr.predict(X_train) 
y_pred_test = dt_regr.predict(X_test) 


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def performanceMetricsDF(metricsObj, yTrain, yPredTrain, yTest, yPredTest):
  measures_list = ['MAE','RMSE', 'R^2','MAPE (%)','MAX Error']
  train_results = [metricsObj.mean_absolute_error(yTrain, yPredTrain),
                np.sqrt(metricsObj.mean_squared_error(yTrain, yPredTrain)),
                metricsObj.r2_score(yTrain, yPredTrain),
                mean_absolute_percentage_error(yTrain, yPredTrain),
                metricsObj.max_error(yTrain, yPredTrain)]
  test_results = [metricsObj.mean_absolute_error(yTest, yPredTest),
                np.sqrt(metricsObj.mean_squared_error(yTest, yPredTest)),
                metricsObj.r2_score(yTest, yPredTest),
                  mean_absolute_percentage_error(yTest, yPredTest),
                metricsObj.max_error(yTest, yPredTest)]
  resultsDF = pd.DataFrame({'Measure': measures_list, 'Train': train_results, 'Test':test_results})
  return(resultsDF)

resultsDF = performanceMetricsDF(metrics, y_train, y_pred_train, y_test, y_pred_test)
resultsDF


