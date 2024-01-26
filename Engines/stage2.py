import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
## Decision tree cannot deal with Categoric varables 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import pickle


#read the dataframe
from pandas import read_excel
import numpy as np
sheetname='June'
path='C:\Git\hackday\Engines\input.xlsx'
data=read_excel(path, sheet_name = sheetname)
data.head(10)

#some insights
data.dtypes
data.isnull().sum()
data.dropna(subset=["CATEGORY"],axis=0, inplace=True)
data.dropna(subset=["DEPARTMENT"],axis=0, inplace=True)
data.dropna(subset=["LOCATION"],axis=0, inplace=True)
data.dropna(subset=["RESOLUTION_SLA"],axis=0, inplace=True)
data.isnull().sum()
data.dtypes


##################
#EDA analysis
##################
data.dtypes

type(data)
data.columns

###Columns in focus to generate new columns RESPONSE_SLA_TIME, RESOLUTION_SLA_TIME
# REGISTRATION_TIME
# RESPONSE_DEADLINE
# RESOLUTION_DEADLINE

#print(data['RESPONSE_DEADLINE'] - data['REGISTRATION_TIME'])
toprecords= data.head(10)

#SLA time
data['RESPONSE_SLA_TIME']= data['RESPONSE_DEADLINE'] - data['REGISTRATION_TIME']
data['RESOLUTION_SLA_TIME']= data['RESOLUTION_DEADLINE'] - data['REGISTRATION_TIME']

data['RESPONSE_SLA_TIME'] = data['RESPONSE_SLA_TIME'] / np.timedelta64(1, 'm')
data['RESOLUTION_SLA_TIME'] = data['RESOLUTION_SLA_TIME'] / np.timedelta64(1, 'm')

#Actual time / Elapsed time
data['RESPONSE ELAPSED TIME'] = data['RESPONSE_TIME'] - data['REGISTRATION_TIME']
data['RESOLUTION ELAPSED TIME'] = data['RESOLUTION_TIME'] - data['REGISTRATION_TIME']

data['RESPONSE ELAPSED TIME'] = data['RESPONSE ELAPSED TIME'] / np.timedelta64(1, 'm')
data['RESOLUTION ELAPSED TIME'] = data['RESOLUTION ELAPSED TIME'] / np.timedelta64(1, 'm')

data.dtypes

data.head(5)
#print(toprecords['CATEGORY'], toprecords['RESOLUTION_TIME'] - toprecords['REGISTRATION_TIME'])

data.dropna(subset=["RESPONSE ELAPSED TIME"],axis=0, inplace=True)
data.dropna(subset=["RESOLUTION ELAPSED TIME"],axis=0, inplace=True)

data['RESPONSE_ELAPSED_TIME'] = data['RESPONSE ELAPSED TIME'].astype('int')
data['RESOLUTION_ELAPSED_TIME'] = data['RESOLUTION ELAPSED TIME'].astype('int')


data['RESPONSE_SLA_TIME'] = data['RESPONSE_SLA_TIME'].astype('int')
data['RESOLUTION_SLA_TIME'] = data['RESOLUTION_SLA_TIME'].astype('int')

#Columns already which we have
#  RESPONSE ELAPSED TIME
#  RESOLUTION ELAPSED TIME

data.columns

data['REGISTRATION_DATE'] = data["REGISTRATION_TIME"].values.astype('datetime64[D]')

subdata=data[['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA_TIME','RESOLUTION_ELAPSED_TIME','RESOLUTION_SLA']]
print("subdata") 

subdata = subdata.groupby(['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA'], as_index=False).sum()

subdata

subdata2 = data[['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA','TICKET_NO']]

subdata2 = subdata2.groupby(['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA'], as_index=False).count()



resourcedf =  data[['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY', 'ASSIGNED_ENGINEER','RESOLUTION_SLA']]

resourcedf = resourcedf.groupby(['REGISTRATION_DATE','LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA'], as_index=False).ASSIGNED_ENGINEER.nunique().to_frame()

resourcedf

mergedf = pd.concat([subdata, subdata2,resourcedf], axis=1)

#mergedf.to_csv('aggredateddata.csv', index=False)


#Modifed dataset
modifedDataset = mergedf.rename(columns = {"TICKET_NO": "NO_OF_TICKETS",
                                  "ASSIGNED_ENGINEER":"TOTAL_NO_OF_ASSIGNED_ENGINEERS"
                                  })

#modifedDataset = mergedf.rename(index = {"TICKET_NO": "NO_OF_TICKETS",
                                 # "ASSIGNED_ENGINEER":"TOTAL_NO_OF_ASSIGNED_ENGINEERS"
                                 # }, 
                                 #inplace = True)


#Overallresource=99
#Perdayresourcedeployment given by resource

modifedDataset = modifedDataset.loc[:,~modifedDataset.columns.duplicated()]

modifedDataset['OVERALL_NO_OF_ASSIGNED_ENGINEERS']=99



modifedDataset.head(5)

#data.dtypes
#Input variables
#REGISTRATION_DATE --Not required
#LOCATION
#DEPARTMENT
#CATEGORY
#CRITICALITY
#No.OFTICKETS
#No.OFRESOURCES
#RESOLUTION_SLA_TIME
#RESOLUTION ELAPSED TIME

#Target variable 
#RESOLUTION_SLA

#Trim spaces
modifedDataset['LOCATIONS'] = modifedDataset['LOCATIONS'].str.replace(' ', '')
modifedDataset['DEPARTMENT'] = modifedDataset['DEPARTMENT'].str.replace(' ', '')
modifedDataset['CATEGORY'] = modifedDataset['CATEGORY'].str.replace(' ', '')
modifedDataset['CRITICALITY'] = modifedDataset['CRITICALITY'].str.replace(' ', '')
modifedDataset['RESOLUTION_SLA'] = modifedDataset['RESOLUTION_SLA'].str.replace(' ', '')

#input variables
X = modifedDataset[['LOCATIONS','DEPARTMENT','CATEGORY','CRITICALITY','RESOLUTION_SLA_TIME', 'RESOLUTION_ELAPSED_TIME','NO_OF_TICKETS','TOTAL_NO_OF_ASSIGNED_ENGINEERS','OVERALL_NO_OF_ASSIGNED_ENGINEERS']].values

#target variable
y = modifedDataset[['RESOLUTION_SLA']].values

print(type(X))
#data.CATEGORY.unique()
#data.DEPARTMENT.unique()

#['GGN', 'Technology', 'OperatingSystem', 'Minor', 3798, 3101, 1, 1, 99]
modifedDataset[(modifedDataset['LOCATIONS']=='GGN')  & (modifedDataset['DEPARTMENT']=='Technology') & (modifedDataset['CATEGORY']=='OperatingSystem') & (modifedDataset['CRITICALITY']=='Minor')]


#encode model doesnot understand english

le_LOCATIONS = preprocessing.LabelEncoder()
le_LOCATIONS.fit(['BLR', 'GGN', 'KOC',
       'PNQ', 'HYD'])

X[:,0] = le_LOCATIONS.transform(X[:,0]) 


le_DEPARTMENT = preprocessing.LabelEncoder()
le_DEPARTMENT.fit(['Audit-Business', 'Audit-Management', 'CapabilityHubs-RPOBusiness',
       'FRM', 'GRCS-DEL', 'Technology', 'CF-HR', 'K-CRC-RBSPPI-Permanent',
       'K-CRC-RBSPPI-Retainers', 'RC-RA-InternalAudit&EnterpriseRisk',
       'Administration', 'CapabilityHubs-KM-Knowledge',
       'CapabilityHubs-CorporateMGT',
       'CapabilityHubs-Research-LegalSupport', 'DACore-TS-FA/FDD',
       'T&RCore-CF-VALS', 'MC-PMO', 'T&RCore-InfraGILT',
       'T&RCore-Strategy', 'T&RCore-TS-FA/FDD', 'T&RCore-AAS',
       'T&RCore-CF-M&A', 'T&RCore-InfraModeling', 'GDC-Audit-Business',
       'Tax-CorpTax', 'Tax-Common', 'Tax-GMS', 'Tax-ProjectSalt',
       'MC-PMOPoolingIES', 'T&RCore-Modeling', 'GDC2-Tax-Valuation',
       'AdvisoryInnovation', 'T&RCore-SPIInsights',
       'T&RCoreIntegration/Separation', 'T&RCore-TSINFRA', 'MC-P3',
       'RC-RA-MajorProjects&ContractAdvisory',
       'CapabilityHubs-Research-DAHub', 'T&RCore-Infrastructure',
       'T&RCore-RES-Modelling', 'C&O-ProductOperations&Procurement',
       'T&RCore-CF-CAG', 'MC-Global-ProjectDelivery-RM',
       'RCCore-Modeling', 'MC-ITAdvisory-EIM', 'IT', 'T&R-Management',
       'GDCAdv-RC-RS&C-Forensic', 'MC-CH-Data&Analytics',
       'T&RCore-Benchmarking', 'MC-SharedServices&Outsourcing',
       'CF-Facilities&Admin', 'T&RCore-ITDevelopment',
       'CapabilityHubs-UKResearch&Benchmarking',
       'CapabilityHubs-AMS-Marketing', 'CapabilityHubs-Research-DA-MGT',
       'T&RCore-ITDueDiligence', 'CapabilityHubs-BusinessSupportGroup',
       'GDC2-T&RCore-FDD', 'CapabilityHubs-RPOUKPeopleCenterteam',
       'T&RCore-StrategyPooling', 'T&RCore-ManagementStrategy', 'CF-CSR',
       'MC-ITAdvisory-Testing,QA', 'TE-ES-Workday',
       'GDCAdv-RC-RA-ITAudit&Assurance', 'TE-ES-EnterpriseAnalytics',
       'MC-TechnicalSupportTeam', 'KGS-Adv-Ops-DedicatedSupport',
       'Tax-IES', 'Tax-TPL', 'Tax-SBA', 'Tax-PrivateTaxCompliance',
       'CapabilityHubs-Research-MD-PRA', 'MC-PMOAuditCore',
       'CapabilityHubs-AMS-KBSSalesPipeline',
       'CapabilityHubs-Research-MD-SECRES',
       'CapabilityHubs-Pursuits-DAHub', 'CapabilityHubs-Research-GC&K',
       'CapabilityHubs-Research-MD-CI', 'CapabilityHubs-KM-Audit',
       'RAK-Research', 'GRCS-BLR', 'RC-RS&C-RiskAnalytics',
       'RC-RS&C-Spectrum', 'RC-TR-Cyber', 'RC-RA-ITAudit&Assurance',
       'MC-GDN-Workday', 'Tax-Management',
       'TE-AI2-RoboticsProcessAutomation', 'KPMGBusinessSchool',
       'MC-GDN-D&A', 'CapabilityHubs-Research-Tax', 'RC-TR-GRCTechnology',
        'KTech-ES', 'TE-ES-OracleFinancials', 'MC-PMOESS',
       'GDC2-T&RCore-Benchmarking', 'MC-ITAdvisory-CIOAdvisory',
       'CF-HRGDC', 'MC-TaxTechnologyHub',
       'MC-Global-FunctionalSupport-RM', 'GDCAudit-Management',
       'MC-ServiceNow', 'MC-GDN-ServiceNow', 'TE-AI2-ProcessAutomation',
       'CapabilityHubs-AMS-MD-ACT', 'CapabilityHubs-PursuitsCentral',
       'MC-FinancialManagement', 'T&RCore-TVP',
       'MC-Global-Communities-RM', 'CF-HRKRC', 'MC-KLA',
       'RC-FORENSIC-ASTRUS', 'RC-Management',
       'CapabilityHubs-KM-Platform-Content', 'CapabilityHubs-QRM-AML',
       'Tax-CorptaxManagement', 'CapabilityHubs-Benchmarking-ABB',
       'MC-ITAdvisory-SAP', 'CapabilityHubs-MS-Graphics',
       'CF-Risk-&-Legal', 'CF-HRMC', 'T&RHub-CreativeServices',
       'CapabilityHubs-MS-CreativeCentral', 'Tax-Pooling-TP',
       'RC-Digital', 'TE-ES-OracleHCM', 'MC-Management', 'MS-Marvel',
       'CapabilityHubs-Research-GCM', 'RC-RS&C-Forensic',
       'GDCAudit-ProductIntegrity', 'KTech-RDC',
       'GDCAudit-InformationTechnologyTechnicalLeads', 'CF-HRRC',
       'Audit-RC', 'MC-People&Change',
       'CapabilityHubs-Pursuits-MD-MPC-BGL',
       'MC-Quality/ProcessExcellence', 'MC-PMOManagement',
       'C&O-CustomerSolutions', 'MC-Pooling', 'SGI-D&A',
       'MCCore-Modeling', 'MC-Global-Methods-RM', 'GDC2-T&RCore-AAS',
       'MC-PMO-DACore', 'C&O-ServiceOperations-HCLS', 'MC-FS', 'CF-SPMO',
       'GDCAudit-IntelligentAutomation', 'MC-Global-Insights-RM',
       'LH-HCLS-OB', 'T&RCore-ITDevelopmentTesting',
       'MC-GDN-ProductOperations&Procurement', 'MC-DMS',
       'MC-ITAdvisory-EPM', 'MC-PMOESSGDC2', 'MC-AppSupport',
       'MC-CH-GlobalD&A', 'MC-TaxTechnologyD&A',
       'CapabilityHubs-MS-DAHub', 'MS', 'Forensic-Inv', 'Tax-R&D',
       'CapabilityHubs-KM-GCM-GTM', 'GDCAdv-RC-RS&C-RiskAnalytics',
       'RC-Secondments', 'Tax-RMTCorpTax-SEZ', 'Digital',
       'TE-ES-Microsoft', 'CapabilityHubs-KM-DigitalMarketing',
       'MS-Rubicon', 'CapabilityHubs-AMS-GCMPlatinumSupport',
       'CapabilityHubs-ResourceManagement', 'GPOS-Management',
       'MC-HyperionFT', 'MC-PMOGCMS', 'ITSupport', 'Tax-Assurance',
       'RAK-Research-LegalSupport', 'CS-Graphics',
       'CapabilityHubs-AMS-FS', 'Tax-M&A', 'MC-PMOTCoE',
       'CapabilityHubs-Pursuits-RC', 'Tax-PMOManagement', 'HR', 'BPS',
       'RC-Forensic-ComplianceandMonitoring', 'GPOS-GLMS',
       'CapabilityHubs-Research-SN', 'RC-RS&C-Operations&ComplianceRisk',
       'KGSMNationalLeadership', 'CapabilityHubs-Research-MGT',
       'AdvisoryOperations', 'KGS-Adv-RPA-TIGER', 'Audit-DEL', 'RC-AAS',
       'CapabilityHubs-AMS-KBSAccountSupport', 'MC-Global-Tech&Infra-RM',
       'Tax-PE', 'MC-PoweredOperations',
       'CapabilityHubs-QRM-ClientAcceptance', 'CapabilityHubs-KM-Hub',
       'RAK-CRM', 'CapabilityHubs-MS-DAHub-ProofReader',
       'CapabilityHubs-AMS-GCM-Reporting&Solutions', 'GSS-Mar-Comm',
       'GPOS-SPS', 'KGS-LeadsDedicatedOps', 'RAK-RESE-SUPP-REG-SGI',
       'T&RCore-Restructuring/DDL', 'CapabilityHubs-MS-Creative-MC',
       'CapabilityHubs-KM-Portal', 'KGS-Adv-Ops-CentralizedReporting',
       'RC-Queen', 'T&RCore-Analytics', 'MC-Global-StrategySupport-RM',
       'CapabilityHubs-KM-MD-CM', 'CF-IT', 'CapabilityHubs-KM-Strategy',
       'CapabilityHubs-KM-Platform-Search', 'CapabilityHubs-AMS-CRM',
       'CapabilityHubs-KM-GCM-Knowledge',
       'KGSCapabilityHubs-GCM-Knowledge', 'CapabilityHubs-MS-MD-MSS',
       'CapabilityHubs-KM-SOC-MEDIA', 'Finance',
       'CapabilityHubs-Research-MD-MA', 'Tax-Support', 'Audit-DEL-D5',
       'MC-CIOA-ServiceMgmt',
       'CapabilityHubs-AMS-GCMAccountandSectorSupport', 'CF-COOOffice',
       'RAK-ABB-BM', 'CapabilityHubs-MS-Creative-SN',
       'GDC2–T&RCoreCF–TaxValuations',
       'CapabilityHubs-KM-MD-DigitalMarketing', 'CapabilityHubs-KM-M&A',
       'CapabilityHubs-Research-MD-ClientValue',
       'CapabilityHubs-Benchmarking-MC', 'RAK-Marketing-AMS',
       'RAK-ResourceManagement', 'CapabilityHubs-Research-RC',
       'RAK-KM-Knowledge', 'GDCAdv-RC-RA-InternalAudit&EnterpriseRisk',
       'AAS-BLR', 'RAK-KM-Audit'])
X[:,1] = le_DEPARTMENT.transform(X[:,1])


le_CATEGORY = preprocessing.LabelEncoder()
le_CATEGORY.fit(['LostAccessories', 'OperatingSystem', 'HardwareIssues',
       'NetworkIssues', 'HostedApplications', 'Loginissues',
       'ApplicationIssues', 'IPPhone', 'DisplayConfigurationIssues',
       'ReconciliationActivity', 'SecurityIncident', 'UATorTesting',
       'VideoConference'])

X[:,2] = le_CATEGORY.transform(X[:,2]) 



le_CRITICALITY = preprocessing.LabelEncoder()
le_CRITICALITY.fit(['Minor', 'Moderate'])

X[:,3] = le_CRITICALITY.transform(X[:,3]) 



X[0:5]

unique, count =np.unique(y,return_counts=True)
print(unique, count)


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


decTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

decTree.fit(X_train,y_train)

#save the model
decisionTreeClassifier_filename = 'decisionTreeClassifier.pkl'
pickle.dump(decTree, open(decisionTreeClassifier_filename, 'wb'))


predTree = decTree.predict(X_test)



print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, predTree, labels=['MET','MISSED']))



newData = [['GGN', 'Technology', 'OperatingSystem', 'Minor', 3798, 3101, 1, 1, 99]]

newData = np.array(newData)

le_LOCATIONS = preprocessing.LabelEncoder()
le_LOCATIONS.fit(['BLR', 'GGN', 'KOC',
       'PNQ', 'HYD'])

newData[:,0] = le_LOCATIONS.transform(newData[:,0]) 


le_DEPARTMENT = preprocessing.LabelEncoder()
le_DEPARTMENT.fit(['Audit-Business', 'Audit-Management', 'CapabilityHubs-RPOBusiness',
       'FRM', 'GRCS-DEL', 'Technology', 'CF-HR', 'K-CRC-RBSPPI-Permanent',
       'K-CRC-RBSPPI-Retainers', 'RC-RA-InternalAudit&EnterpriseRisk',
       'Administration', 'CapabilityHubs-KM-Knowledge',
       'CapabilityHubs-CorporateMGT',
       'CapabilityHubs-Research-LegalSupport', 'DACore-TS-FA/FDD',
       'T&RCore-CF-VALS', 'MC-PMO', 'T&RCore-InfraGILT',
       'T&RCore-Strategy', 'T&RCore-TS-FA/FDD', 'T&RCore-AAS',
       'T&RCore-CF-M&A', 'T&RCore-InfraModeling', 'GDC-Audit-Business',
       'Tax-CorpTax', 'Tax-Common', 'Tax-GMS', 'Tax-ProjectSalt',
       'MC-PMOPoolingIES', 'T&RCore-Modeling', 'GDC2-Tax-Valuation',
       'AdvisoryInnovation', 'T&RCore-SPIInsights',
       'T&RCoreIntegration/Separation', 'T&RCore-TSINFRA', 'MC-P3',
       'RC-RA-MajorProjects&ContractAdvisory',
       'CapabilityHubs-Research-DAHub', 'T&RCore-Infrastructure',
       'T&RCore-RES-Modelling', 'C&O-ProductOperations&Procurement',
       'T&RCore-CF-CAG', 'MC-Global-ProjectDelivery-RM',
       'RCCore-Modeling', 'MC-ITAdvisory-EIM', 'IT', 'T&R-Management',
       'GDCAdv-RC-RS&C-Forensic', 'MC-CH-Data&Analytics',
       'T&RCore-Benchmarking', 'MC-SharedServices&Outsourcing',
       'CF-Facilities&Admin', 'T&RCore-ITDevelopment',
       'CapabilityHubs-UKResearch&Benchmarking',
       'CapabilityHubs-AMS-Marketing', 'CapabilityHubs-Research-DA-MGT',
       'T&RCore-ITDueDiligence', 'CapabilityHubs-BusinessSupportGroup',
       'GDC2-T&RCore-FDD', 'CapabilityHubs-RPOUKPeopleCenterteam',
       'T&RCore-StrategyPooling', 'T&RCore-ManagementStrategy', 'CF-CSR',
       'MC-ITAdvisory-Testing,QA', 'TE-ES-Workday',
       'GDCAdv-RC-RA-ITAudit&Assurance', 'TE-ES-EnterpriseAnalytics',
       'MC-TechnicalSupportTeam', 'KGS-Adv-Ops-DedicatedSupport',
       'Tax-IES', 'Tax-TPL', 'Tax-SBA', 'Tax-PrivateTaxCompliance',
       'CapabilityHubs-Research-MD-PRA', 'MC-PMOAuditCore',
       'CapabilityHubs-AMS-KBSSalesPipeline',
       'CapabilityHubs-Research-MD-SECRES',
       'CapabilityHubs-Pursuits-DAHub', 'CapabilityHubs-Research-GC&K',
       'CapabilityHubs-Research-MD-CI', 'CapabilityHubs-KM-Audit',
       'RAK-Research', 'GRCS-BLR', 'RC-RS&C-RiskAnalytics',
       'RC-RS&C-Spectrum', 'RC-TR-Cyber', 'RC-RA-ITAudit&Assurance',
       'MC-GDN-Workday', 'Tax-Management',
       'TE-AI2-RoboticsProcessAutomation', 'KPMGBusinessSchool',
       'MC-GDN-D&A', 'CapabilityHubs-Research-Tax', 'RC-TR-GRCTechnology',
        'KTech-ES', 'TE-ES-OracleFinancials', 'MC-PMOESS',
       'GDC2-T&RCore-Benchmarking', 'MC-ITAdvisory-CIOAdvisory',
       'CF-HRGDC', 'MC-TaxTechnologyHub',
       'MC-Global-FunctionalSupport-RM', 'GDCAudit-Management',
       'MC-ServiceNow', 'MC-GDN-ServiceNow', 'TE-AI2-ProcessAutomation',
       'CapabilityHubs-AMS-MD-ACT', 'CapabilityHubs-PursuitsCentral',
       'MC-FinancialManagement', 'T&RCore-TVP',
       'MC-Global-Communities-RM', 'CF-HRKRC', 'MC-KLA',
       'RC-FORENSIC-ASTRUS', 'RC-Management',
       'CapabilityHubs-KM-Platform-Content', 'CapabilityHubs-QRM-AML',
       'Tax-CorptaxManagement', 'CapabilityHubs-Benchmarking-ABB',
       'MC-ITAdvisory-SAP', 'CapabilityHubs-MS-Graphics',
       'CF-Risk-&-Legal', 'CF-HRMC', 'T&RHub-CreativeServices',
       'CapabilityHubs-MS-CreativeCentral', 'Tax-Pooling-TP',
       'RC-Digital', 'TE-ES-OracleHCM', 'MC-Management', 'MS-Marvel',
       'CapabilityHubs-Research-GCM', 'RC-RS&C-Forensic',
       'GDCAudit-ProductIntegrity', 'KTech-RDC',
       'GDCAudit-InformationTechnologyTechnicalLeads', 'CF-HRRC',
       'Audit-RC', 'MC-People&Change',
       'CapabilityHubs-Pursuits-MD-MPC-BGL',
       'MC-Quality/ProcessExcellence', 'MC-PMOManagement',
       'C&O-CustomerSolutions', 'MC-Pooling', 'SGI-D&A',
       'MCCore-Modeling', 'MC-Global-Methods-RM', 'GDC2-T&RCore-AAS',
       'MC-PMO-DACore', 'C&O-ServiceOperations-HCLS', 'MC-FS', 'CF-SPMO',
       'GDCAudit-IntelligentAutomation', 'MC-Global-Insights-RM',
       'LH-HCLS-OB', 'T&RCore-ITDevelopmentTesting',
       'MC-GDN-ProductOperations&Procurement', 'MC-DMS',
       'MC-ITAdvisory-EPM', 'MC-PMOESSGDC2', 'MC-AppSupport',
       'MC-CH-GlobalD&A', 'MC-TaxTechnologyD&A',
       'CapabilityHubs-MS-DAHub', 'MS', 'Forensic-Inv', 'Tax-R&D',
       'CapabilityHubs-KM-GCM-GTM', 'GDCAdv-RC-RS&C-RiskAnalytics',
       'RC-Secondments', 'Tax-RMTCorpTax-SEZ', 'Digital',
       'TE-ES-Microsoft', 'CapabilityHubs-KM-DigitalMarketing',
       'MS-Rubicon', 'CapabilityHubs-AMS-GCMPlatinumSupport',
       'CapabilityHubs-ResourceManagement', 'GPOS-Management',
       'MC-HyperionFT', 'MC-PMOGCMS', 'ITSupport', 'Tax-Assurance',
       'RAK-Research-LegalSupport', 'CS-Graphics',
       'CapabilityHubs-AMS-FS', 'Tax-M&A', 'MC-PMOTCoE',
       'CapabilityHubs-Pursuits-RC', 'Tax-PMOManagement', 'HR', 'BPS',
       'RC-Forensic-ComplianceandMonitoring', 'GPOS-GLMS',
       'CapabilityHubs-Research-SN', 'RC-RS&C-Operations&ComplianceRisk',
       'KGSMNationalLeadership', 'CapabilityHubs-Research-MGT',
       'AdvisoryOperations', 'KGS-Adv-RPA-TIGER', 'Audit-DEL', 'RC-AAS',
       'CapabilityHubs-AMS-KBSAccountSupport', 'MC-Global-Tech&Infra-RM',
       'Tax-PE', 'MC-PoweredOperations',
       'CapabilityHubs-QRM-ClientAcceptance', 'CapabilityHubs-KM-Hub',
       'RAK-CRM', 'CapabilityHubs-MS-DAHub-ProofReader',
       'CapabilityHubs-AMS-GCM-Reporting&Solutions', 'GSS-Mar-Comm',
       'GPOS-SPS', 'KGS-LeadsDedicatedOps', 'RAK-RESE-SUPP-REG-SGI',
       'T&RCore-Restructuring/DDL', 'CapabilityHubs-MS-Creative-MC',
       'CapabilityHubs-KM-Portal', 'KGS-Adv-Ops-CentralizedReporting',
       'RC-Queen', 'T&RCore-Analytics', 'MC-Global-StrategySupport-RM',
       'CapabilityHubs-KM-MD-CM', 'CF-IT', 'CapabilityHubs-KM-Strategy',
       'CapabilityHubs-KM-Platform-Search', 'CapabilityHubs-AMS-CRM',
       'CapabilityHubs-KM-GCM-Knowledge',
       'KGSCapabilityHubs-GCM-Knowledge', 'CapabilityHubs-MS-MD-MSS',
       'CapabilityHubs-KM-SOC-MEDIA', 'Finance',
       'CapabilityHubs-Research-MD-MA', 'Tax-Support', 'Audit-DEL-D5',
       'MC-CIOA-ServiceMgmt',
       'CapabilityHubs-AMS-GCMAccountandSectorSupport', 'CF-COOOffice',
       'RAK-ABB-BM', 'CapabilityHubs-MS-Creative-SN',
       'GDC2–T&RCoreCF–TaxValuations',
       'CapabilityHubs-KM-MD-DigitalMarketing', 'CapabilityHubs-KM-M&A',
       'CapabilityHubs-Research-MD-ClientValue',
       'CapabilityHubs-Benchmarking-MC', 'RAK-Marketing-AMS',
       'RAK-ResourceManagement', 'CapabilityHubs-Research-RC',
       'RAK-KM-Knowledge', 'GDCAdv-RC-RA-InternalAudit&EnterpriseRisk',
       'AAS-BLR', 'RAK-KM-Audit'])
newData[:,1] = le_DEPARTMENT.transform(newData[:,1])


le_CATEGORY = preprocessing.LabelEncoder()
le_CATEGORY.fit(['LostAccessories', 'OperatingSystem', 'HardwareIssues',
       'NetworkIssues', 'HostedApplications', 'Loginissues',
       'ApplicationIssues', 'IPPhone', 'DisplayConfigurationIssues',
       'ReconciliationActivity', 'SecurityIncident', 'UATorTesting',
       'VideoConference'])

newData[:,2] = le_CATEGORY.transform(newData[:,2]) 



le_CRITICALITY = preprocessing.LabelEncoder()
le_CRITICALITY.fit(['Minor', 'Moderate'])

newData[:,3] = le_CRITICALITY.transform(newData[:,3]) 



newData[0:5]

predTree = decTree.predict(newData)
print("te rsult" +predTree)