# Author: Mohammed Alsoughayer
# Descritpion: Gather Data and save it in a csv file

# Import necessary packages
# HTTP request packages 
from urllib.request import urlopen
import requests

# Data Packages
import csv
import numpy as np
import pandas as pd
import math
import seaborn as sns

# Parsing packages
import re
from bs4 import BeautifulSoup


# Practice
URL = "https://en.wikipedia.org/wiki/List_of_weight_training_exercises"
# using standard library packages
page = urlopen(URL)
html = page.read().decode("utf-8")

# using request
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}
r = requests.get(url=URL)

soup = BeautifulSoup(r.content, 'html5lib')

# write html to text file
# f = open("web_info3.txt", "w")
# f.write(soup.prettify())
# f.close()

workouts=[]  # a list to store quotes
   
table = soup.find('table', attrs = {'class':"wikitable"}) 
# get columns 
columns = []
first_row = table.find('tr')
for col in first_row.find_all('th'):
    txt = col.text
    columns.append(txt.replace('\n','').replace('-',''))
# print(columns)

# fill all entries, entry encoding ('':0, 'Yes':1, 'Some':2)
counter = 1
for row in table.findAll('tr'):
    col_index = 1
    # skip first column 
    if counter == 1:
        counter = 0
        continue
    workout = {}
    workout[columns[0]] = row.find('th').text.replace('\n','')
    for col in row.findAll('td'):
        inp = col.text.replace('\n','')
        if inp == 'Yes':
            workout[columns[col_index]] = 1
        elif inp == 'Some':
            workout[columns[col_index]] = 2
        else:
            workout[columns[col_index]] = 0
        col_index = col_index + 1
    workouts.append(workout)

# write data into csv file 
filename = 'strength_workouts.csv'
with open(filename, 'w', newline='') as f:
    w = csv.DictWriter(f,columns)
    w.writeheader()
    for row in workouts:
        w.writerow(row)


# for row in table.findAll('div',
#                          attrs = {'class':'col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30px-top'}):
#     quote = {}
#     quote['theme'] = row.h5.text
#     quote['url'] = row.a['href']
#     quote['img'] = row.img['src']
#     quote['lines'] = row.img['alt'].split(" #")[0]
#     quote['author'] = row.img['alt'].split(" #")[1]
#     quotes.append(quote)
   
# filename = 'inspirational_quotes.csv'
# with open(filename, 'w', newline='') as f:
#     w = csv.DictWriter(f,['theme','url','img','lines','author'])
#     w.writeheader()
#     for quote in quotes:
#         w.writerow(quote)


# pattern = "<li.*?>.*?</li.*?>"
# match_results = re.findall(pattern, html, re.IGNORECASE)
# for paragraph in match_results:
#     paragraph = re.sub("<.*?>", "", paragraph) # Remove HTML tags
#     print(paragraph)
# paragraphs=[]  # a list to store quotes
   
# table = soup.findAll('a', attrs = {'class':"content-link css-5r4717"})
   
# for row in table:
#     paragraph = {}
#     paragraph['workout'] = row.text
#     # paragraph['decription'] = row.text
#     paragraphs.append(paragraph)

# print(paragraphs)
# filename = 'inspirational_quotes.csv'
# with open(filename, 'w', newline='') as f:
#     w = csv.DictWriter(f,['theme','url','img','lines','author'])
#     w.writeheader()
#     for quote in quotes:
#         w.writerow(quote)

# pattern = "<title.*?>.*?</title.*?>"
# match_results = re.search(pattern, html, re.IGNORECASE)
# title = match_results.group()
# title = re.sub("<.*?>", "", title) # Remove HTML tags

# print(title)

# soup = BeautifulSoup(html, "html.parser")
# print(soup.title.string)