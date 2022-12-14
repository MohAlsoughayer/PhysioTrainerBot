# Author: Mohammed Alsoughayer
# Descritpion: Gather Data and save it in a csv file 

# Import necessary packages
import requests
import numpy as np
import pandas as pd
import math
import seaborn as sns
import re
import mechanicalsoup
from bs4 import BeautifulSoup
import html
import urllib.parse
# Create Class 
class HealthProfile():
    def __init__(self, age, symptoms, location, description):
        self.age = age
        self.symptoms = symptoms
        self.location = location
        self.description = description

#examples 
p1 = HealthProfile(16, ['unstable knee', 'swelling'], 'knee', 'while playing football, I made a sharp turn and my knee popped')
p2 = HealthProfile(24, ['swelling', 'pain'], 'wrist', 'while playing football, I got hit by the ball')

# web-crawling 
base_url = 'https://www.physio-pedia.com/'

r = requests.get(url=base_url)

soup = BeautifulSoup(r.content, 'html5lib')
print(soup.prettify())
# Connect to Qwant
# browser = mechanicalsoup.StatefulBrowser(user_agent='MechanicalSoup')
# browser.open(base_url)

# # Fill-in the search form
# browser.select_form('form[action="/search"]')
# browser["stq"] = p1.location
# browser.submit_selected()

# # Display the results
# for link in browser.page.select('.result a'):
#     # Qwant shows redirection links, not the actual URL, so extract
#     # the actual URL from the redirect link:
#     href = link.attrs['href']
#     m = re.match(r"^/redirect/[^/]*/(.*)$", href)
#     if m:
#         href = urllib.parse.unquote(m.group(1))
#     print(link.text, '->', href)
