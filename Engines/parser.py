#remove 
import re

with open('Book1.txt', 'r') as file:
    data = file.read()


data = re.sub(r'Application Issues.*?,','Application Issues,', data, flags=re.DOTALL)
data = re.sub(r'Security Incident.*?,','Security Incident,', data, flags=re.DOTALL)
data = re.sub(r'Hosted Applications.*?,','Application Issues,', data, flags=re.DOTALL)
data = re.sub(r'Hardware Issues.*?,','Hardware Issues,', data, flags=re.DOTALL)
data = re.sub(r'Operating System.*?,','Hardware Issues,', data, flags=re.DOTALL)
data = re.sub(r'Lost Accessories.*?,','Application Issues,', data, flags=re.DOTALL)
data = re.sub(r'Network Issues.*?,','Application Issues,', data, flags=re.DOTALL)
data = re.sub(r'Login issues.*?,','Login issues,', data, flags=re.DOTALL)
data = re.sub(r'IP Phone.*?,','Login issues,', data, flags=re.DOTALL)

with open('book1comp.csv', 'w') as file:
    file.write(data)

