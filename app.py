from flask import Flask, render_template, flash, redirect, url_for, session, request, Response, Markup
import json
import sys
import  Engines.prodStage1 as stage1
import Engines.stage2Prod as stage2
import time

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/user", methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        f = request.form
        issue = f.get('logTicket')
        location = f.get('selectLocation')
        department = f.get('selectDepartment')
        criticality = f.get('criticality')
        

        #process json
        with open('tickets.json') as json_file:
            ticketsData = json.load(json_file)
        
        tickets = ticketsData["tickets"]
        i = len(tickets)
        '''
        {
            "TICKET_No":,
            "LOCATIONS": "",
            "DEPARTMENT": "",
            "CATEGORY": "",
            "CRITICALITY": "",
            "RESOLUTION_SLA_TIME": "",
            "RESOLUTION_ELAPSED_TIME":"",
            "NO_OF_TICKETS": "",
            "TOTAL_NO_OF_ASSIGNED_ENGINEERS": "",
            "OVERALL_NO_OF_ASSIGNED_ENGINEERS": "",
			"SUGGESTED_SLA",""

        }
        '''

        #new ticket = i+1
        i +=1
        #if i is 1 means no ticket so no need to check
        ticket = {}
        ticket["TICKET_No"] = i
        

        #get category
        CATEGORY = stage1.getCategory(str(issue))

        ticket["CATEGORY"] = CATEGORY
        ticket["LOCATIONS"] = location
        ticket["DEPARTMENT"] = department
        ticket["CRITICALITY"] = criticality

        tickets.append(ticket)
        ticketsData["tickets"] = tickets

        with open('tickets.json', 'w') as f:
            json.dump(ticketsData, f)
        print(CATEGORY)
        flash('Your Ticket No. '+" "+str(i)+'has been logged under category '+str(CATEGORY))
        return redirect(url_for('user'))
        
        

    return render_template('user.html')

@app.route("/admin", methods=['GET', 'POST'])
def admin():
    #read config

    with open('adminConfig.json') as json_file:
        configData = json.load(json_file)
    
    
    #now create big json to feed to model
    
    #first read tickets json
    with open('tickets.json') as json_file:
            ticketsData = json.load(json_file)
    
    #create a new json that has also data from adminConfig.json
    tickets = ticketsData["tickets"]

    NO_OF_TICKETS = len(tickets)

    for ticket in tickets:
        ticket["RESOLUTION_ELAPSED_TIME"] = configData["RESOLUTION_ELAPSED_TIME"]
        RESOLUTION_SLA_TIME = 0
        sla_category = ticket["CATEGORY"]
        for each in configData["RESOLUTION_SLA_TIME"]:
            if each["id"] == sla_category:
                RESOLUTION_SLA_TIME = each["value"]

        ticket["RESOLUTION_SLA_TIME"] = RESOLUTION_SLA_TIME
        ticket["TOTAL_NO_OF_ASSIGNED_ENGINEERS"] = configData["TOTAL_NO_OF_ASSIGNED_ENGINEERS"]
        ticket["OVERALL_NO_OF_ASSIGNED_ENGINEERS"] = configData["OVERALL_NO_OF_ASSIGNED_ENGINEERS"]
        ticket["NO_OF_TICKETS"] = NO_OF_TICKETS
    
    print(tickets)
    

    #run model for each ticket
    for ticket in tickets:
         dataToPredict = [ticket["LOCATIONS"], ticket["DEPARTMENT"], str(ticket["CATEGORY"]).replace(" ",""), ticket["CRITICALITY"], ticket["RESOLUTION_SLA_TIME"], ticket["RESOLUTION_ELAPSED_TIME"], ticket["NO_OF_TICKETS"], ticket["TOTAL_NO_OF_ASSIGNED_ENGINEERS"], ticket["OVERALL_NO_OF_ASSIGNED_ENGINEERS"]]
         print(dataToPredict)
         outputVar,outputVar1 = stage2.runStage2(dataToPredict)
         print(type(outputVar))
         print(outputVar)
         ticket["RESOLUTION_SLA"] = outputVar[0]
         type(outputVar1)
         ticket["SUGGESTED_SLA"] = outputVar1

    if request.method == 'POST':
        f = request.form
        #update config
        with open('adminConfig.json') as json_file:
            configDataUpdate = json.load(json_file)
        
        assignedEngineers = f.get('assignedEngineers')
        resolution_elapsed_time = f.get('resolution_elapsed_time')
        selectSlaTime = f.get('selectSlaTime')
        resolution_sla_time = f.get('resolution_sla_time')
        total_no_of_engineers = f.get('total_no_of_engineers')

        configDataUpdate["RESOLUTION_ELAPSED_TIME"] = resolution_elapsed_time
        configDataUpdate["TOTAL_NO_OF_ASSIGNED_ENGINEERS"] = total_no_of_engineers
        configDataUpdate["OVERALL_NO_OF_ASSIGNED_ENGINEERS"] = assignedEngineers

        for each in configDataUpdate["RESOLUTION_SLA_TIME"]:
            if each["id"] == selectSlaTime:
                each["value"] = resolution_sla_time
        
        with open('adminConfig.json', 'w') as f:
            json.dump(configDataUpdate, f)
        
        configData=configDataUpdate

        with open('tickets.json') as json_file:
            ticketsData = json.load(json_file)
    
        #create a new json that has also data from adminConfig.json
        tickets = ticketsData["tickets"]
    
        NO_OF_TICKETS = len(tickets)
    
        for ticket in tickets:
            ticket["RESOLUTION_ELAPSED_TIME"] = configData["RESOLUTION_ELAPSED_TIME"]
            RESOLUTION_SLA_TIME = 0
            sla_category = ticket["CATEGORY"]
            for each in configData["RESOLUTION_SLA_TIME"]:
                if each["id"] == sla_category:
                    RESOLUTION_SLA_TIME = each["value"]
    
            ticket["RESOLUTION_SLA_TIME"] = RESOLUTION_SLA_TIME
            ticket["TOTAL_NO_OF_ASSIGNED_ENGINEERS"] = configData["TOTAL_NO_OF_ASSIGNED_ENGINEERS"]
            ticket["OVERALL_NO_OF_ASSIGNED_ENGINEERS"] = configData["OVERALL_NO_OF_ASSIGNED_ENGINEERS"]
            ticket["NO_OF_TICKETS"] = NO_OF_TICKETS
        
        print(tickets)
        
    
        #run model for each ticket
        for ticket in tickets:
            dataToPredict = [ticket["LOCATIONS"], ticket["DEPARTMENT"], str(ticket["CATEGORY"]).replace(" ",""), ticket["CRITICALITY"], ticket["RESOLUTION_SLA_TIME"], ticket["RESOLUTION_ELAPSED_TIME"], ticket["NO_OF_TICKETS"], ticket["TOTAL_NO_OF_ASSIGNED_ENGINEERS"], ticket["OVERALL_NO_OF_ASSIGNED_ENGINEERS"]]
            print(dataToPredict)
            outputVar,outputVar1 = stage2.runStage2(dataToPredict)
            print(type(outputVar))
            print(outputVar)
            ticket["RESOLUTION_SLA"] = outputVar[0]
            type(outputVar1)
            ticket["SUGGESTED_SLA"] = outputVar1
        
    return render_template('admin.html', configData=configData, tickets=tickets)

if __name__ == "__main__":
    # app.run()
    app.run(debug=True, host='0.0.0.0', port=5001)
