import os
import pandas as pd
import numpy as np
import pyspark.sql.functions as fn
from pyspark.sql.functions import desc
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id, rand, col
from textblob import TextBlob
import nltk
import datetime
nltk.download('punkt')
import html
import re
from string import punctuation
import contractions


def encodeDecode(s):
    return str(s.encode('ascii', 'ignore').decode('utf-8'))
  
def removeHashtag(s):
    return(' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", s).split()))

def replaceHTMLCode(s):
    return html.unescape(s)
  
def lowerText(s):
    return s.lower()

def contractionsExpansion(s):
    return contractions.fix(s)

def htmlRemover(s):
    return ' '.join(re.sub("(\w+:\/\/\S+)", " ", s).split())

def removeExtraWhiteSpace(s):
    return re.sub(' +', ' ', s)
  
def removeExtraNoise(s):
    return s.translate(s.maketrans("\n\t\r", "   "))
  
def removeRT(s):
    return re.sub('RT ', '', s)
  
def removePunctuation(s):
    return ''.join(c for c in s if c not in punctuation)
  
def replaceSmiley(s):
    return " ".join([SMILEYS_dict[word] if word in SMILEYS_dict else word for word in s.split()])
  
def replaceSlangs(s):
    return " ".join([tweet_slang_dict[word] if word in tweet_slang_dict else word for word in s.split()])

def replaceMeaning(s):
      return re.sub('Meaning ', '', s)

def clean_text(t):
    # Args: t is a column of text in a data frame
    return (t.map(lowerText)\
             .map(contractionsExpansion)\
             .map(removeHashtag)\
             .map(removeExtraWhiteSpace)\
             .map(removeExtraNoise)\
             .map(removePunctuation))