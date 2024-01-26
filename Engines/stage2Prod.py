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

def encodeNewData(data):

    newData = np.array(data)
    
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

    return newData
    

def runStage2(newData):
    #load the model
    decisionTreeClassifier_filename = 'C:\Git\hackday\Engines\decisionTreeClassifier.pkl'
    loaded_model = pickle.load(open(decisionTreeClassifier_filename, 'rb'))
    
    
    
    #predict 
    #newData = [['GGN', 'Technology', 'OperatingSystem', 'Minor', 3798,3101, 1, 1, 99]]
    data = [newData]
    data = encodeNewData(data)
    predTree = loaded_model.predict(data)
    suggestedSLA=0
    if(predTree=='MISSED'):
        Ks = 10
        for n in range(1,Ks):
            predTreelocal=""			
            SLA_Time= float(data[0][4])
            SLA_Time= SLA_Time + (SLA_Time*0.1)
            data[0][4] = str(SLA_Time)
            #Train Model and Predict
            print(data)
            predTreelocal = loaded_model.predict(data)
            print(n, predTreelocal,data[0][4])
            if(predTreelocal=='MET'):
                suggestedSLA=data[0][4]
                break
    return predTree,suggestedSLA
