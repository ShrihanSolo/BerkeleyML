'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'


feature_set = {'00', '000', '01', '02', '03', '04', '05',
 '08',
 '09',
 '10',
 '11',
 '12',
 '2000',
 '2001',
 '30',
 '99',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'as',
 'at',
 'attached',
 'be',
 'been',
 'but',
 'by',
 'can',
 'cc',
 'com',
 'company',
 'contract',
 'corp',
 'daren',
 'day',
 'deal',
 'do',
 'ect',
 'enron',
 'farmer',
 'for',
 'forwarded',
 'from',
 'gas',
 'get',
 'has',
 'have',
 'here',
 'hou',
 'hpl',
 'http',
 'if',
 'in',
 'information',
 'into',
 'is',
 'it',
 'know',
 'let',
 'may',
 'me',
 'message',
 'meter',
 'mmbtu',
 'month',
 'more',
 'my',
 'need',
 'new',
 'no',
 'nom',
 'not',
 'of',
 'on',
 'only',
 'or',
 'our',
 'out',
 'please',
 'pm',
 'price',
 'questions',
 're',
 'robert',
 'see',
 'should',
 'sitara',
 'so',
 'subject',
 'texas',
 'thanks',
 'that',
 'the',
 'there',
 'these',
 'they',
 'this',
 'time',
 'to',
 'up',
 'us',
 'volume',
 'volumes',
 'was',
 'we',
 'what',
 'will',
 'with',
 'would',
 'xls',
 'you',
 'your'}

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def freq_website_feature(text, freq):
    return float(freq['website'])

def freq_unsubscribe_feature(text, freq):
    return float(freq['unsubscribe'])

def freq_click_feature(text, freq):
    return float(freq['click'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_thank_feature(text, freq):
    return float(freq['thank'])

def freq_thanks_feature(text, freq):
    return float(freq['thanks'])

def freq_forwarded_feature(text, freq):
    return float(freq['forwarded'])

def freq_rich_feature(text, freq):
    return float(freq['rich'])

def freq_dash_feature(text, freq):
    return text.count('-')

def freq_percent_feature(text, freq):
    return text.count('%')

def freq_at_feature(text, freq):
    return text.count('@')

def freq_dot_feature(text, freq):
    return text.count('.')

def freq_slash_feature(text, freq):
    return text.count('.')

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    
    feature.append(freq_website_feature(text, freq))
    feature.append(freq_unsubscribe_feature(text, freq))
    feature.append(freq_click_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_rich_feature(text, freq))
    feature.append(freq_dash_feature(text, freq))
    feature.append(freq_percent_feature(text, freq))
    feature.append(freq_at_feature(text, freq))
    feature.append(freq_dot_feature(text, freq))
    feature.append(freq_slash_feature(text, freq))
    
    for i in feature_set:
        feature.append(float(freq[i]))
    
    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1))

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)
