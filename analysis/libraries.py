# libraries
import os
import getpass
import glob
import numpy as np
import pandas as pd
import random
import networkx as nx
import requests
import re

## formatting number to appear comma separated and with two digits after decimal: e.g, 1000 shown as 1,000.00
pd.set_option('float_format', "{:,.2f}".format)

import matplotlib.pyplot as plt
#%matplotlib inline
#from matplotlib.legend import Legend

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 200)

import itertools 
import ast # for converting outputs to a list
import networkx as nx
from pyvis.network import Network

# edsl-related stuff
import copy
import random

from edsl.questions import QuestionCheckBox, QuestionFreeText
from edsl import Scenario, Model
from edsl.questions import QuestionMultipleChoice
from itertools import combinations
from edsl.questions.derived.QuestionLinearScale import QuestionLinearScale
from textwrap import dedent

Model.available()
m4 = Model('gpt-4o', temprature = 0.25)
#m4 = Model('gemini-pro')