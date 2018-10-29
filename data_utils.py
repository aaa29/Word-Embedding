#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:19:35 2018

@author: amine
"""


import numpy as np
import pyarabic.araby as araby
import re

from nltk.tokenize import word_tokenize


arabic_letters = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإةؤئى'
sent_end = '؟.،,؛!'


def clean_str(string):
 
    string2 = re.sub(r"[^ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإةؤئى ]", " ", string)
  
    
    string2 = string2.split()
    
    if string2 != []:
        return string2
    else:
        return string.split()
    

def clean_str2(strings):
    
    strings = strings.split('.')
    
    return [clean_str(string) for string in strings if len(clean_str(string))>3]
    





        







