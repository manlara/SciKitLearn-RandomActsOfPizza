import json
#import matplotlib.pyplot as plt
#import pylab as pl
import itertools
import operator
import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import BernoulliRBM

def get_data(myfile, params):
  
  x = []
  y_target = []
  
  with open(myfile) as json_data:
    d = json.load(json_data)
    for items in d:
      Up = 0
      Sub = 0
      UpMinusDown = 0
      UpPlusDown = 0
      Comments = 0
      Account = 0
      Edit = 0
      text = ''
      if items.has_key('requester_upvotes_minus_downvotes_at_retrieval'):
        UpMinusDown = items['requester_upvotes_minus_downvotes_at_retrieval']
      if items.has_key('requester_upvotes_plus_downvotes_at_request'):
        UpPlusDown = items['requester_upvotes_plus_downvotes_at_request']
      if items.has_key('request_number_of_comments_at_retrieval'):
        Comments = items['request_number_of_comments_at_retrieval']
      if items.has_key('request_title'):
        text = items['request_title']
      if items.has_key('request_text'):
        text = text + " " + items['request_text']
      if items.has_key('requester_account_age_in_days_at_retrieval'):
        Account = items['requester_account_age_in_days_at_retrieval']
      if items.has_key('number_of_upvotes_of_request_at_retrieval'):
        Up = items['number_of_upvotes_of_request_at_retrieval']
      if items.has_key('requester_number_of_subreddits_at_request'):
        Sub = items['requester_number_of_subreddits_at_request']
      if items.has_key('requester_days_since_first_post_on_raop_at_retrieval'):
        SincePost = items['requester_days_since_first_post_on_raop_at_retrieval']
      if items.has_key('post_was_edited'):
        if items['post_was_edited']==True:
          Edit = 1
      
      count_help = 0
      count_family = 0
      count_please = 0
      text_array = re.split('\s+', text)
      for item in text_array:
        if 'help' in item.lower():
          count_help = count_help + 1
        if 'family' in item.lower():
          count_family = count_family + 1
        if 'please' in item.lower():
          count_please = count_please + 1
      
      classifier = []
      for param in params:
        if param=="Edit": classifier.append(Edit)
        elif param=="Account": classifier.append(Account)
        elif param=="Up": classifier.append(Up)
        elif param=="Comments": classifier.append(Comments)
        elif param=="count_help": classifier.append(count_help)
        elif param=="count_family": classifier.append(count_family)
        elif param=="count_please": classifier.append(count_please)
        elif param=="UpMinusDown": classifier.append(UpMinusDown)
        elif param=="UpPlusDown": classifier.append(UpPlusDown)
        
      x.append(classifier)
      
      if items['requester_received_pizza']==True:
        y_target.append(1)
      else:
        y_target.append(0)
        
  np_x = np.array(x)
  np_y = np.array(y_target)
  
  return (np_x,np_y)




def do_training(ml_type, params):

  #print "Getting training data"
  np_x, np_y = get_data('train.json', params)

  #print np_x
  #print np_y

  if (ml_type=="GaussianNB"):
    #print "Using a Gaussian Naive Bayes Model to fit the training data with"
    clf = GaussianNB()
  elif (ml_type=="MultinomialNB"):
    #print "Using a Multinomial Naive Bayes Model to fit the training data with"
    clf = MultinomialNB()
  elif (ml_type=="RandomForestClassifier"):
    #print "Using a Random Forest Classifier Model to fit the training data with"
    clf = RandomForestClassifier()
  elif (ml_type=="GradientBoostingClassifier"):
    #print "Using a Gradient Boosting Classifier Model to fit the training data with"
    clf = GradientBoostingClassifier()
  elif (ml_type=="BernoulliRBM"):
    #print "Using a BernoulliRBM Classifier Model to fit the training data with"
    clf = BernoulliRBM()
  clf.fit(np_x, np_y)


  #print "Getting test data"
  np_test_x, np_test_y = get_data('test.json', params)

  total = len(np_test_x)
  correct = 0
  for i in range(0,total):
    predictor = clf.predict([np_test_x[i]])
    value = predictor[0]
    truth = np_test_y[i]
    if predictor[0] == truth:
      correct = correct + 1

  perc = float(correct)/float(total)*100.0
  #print "Stats\n Number of correct predictions / total: "+str(correct)+" / "+str(total)+" = "+str(perc)+" %"
  return perc

ml_types = ["GaussianNB","RandomForestClassifier","GradientBoostingClassifier"]
#params = ["Edit","Account","Up","Comments","UpMinusDown","UpPlusDown","count_help","count_family","count_please"]
params = ["Edit","Account","Up","Comments","UpMinusDown","UpPlusDown"]

maxClassifier = ()
maxVal = 0

allCombos = []
for L in range(3, len(params)+1):
  for subset in itertools.combinations(params, L):
    #print subset
    allCombos.append(subset)

allCombos = list(set(allCombos))
print len(allCombos)
results = {}
counter = 1
for item in allCombos:
  for ml in ml_types:
    if counter%10==0: print str(counter)+" / "+str(len(allCombos)*len(ml_types))
    val = do_training(ml,item)
    results[ml+". " + ""''.join(item)] = val
    counter = counter + 1;


sorted_results = sorted(results.items(),key=operator.itemgetter(1), reverse=True)
print sorted_results

#print(clf.predict([[0, -20, 0]]))

#plt.subplot(2,1,1)
#plt.hist(x0, bins=5000)
#plt.title("Pizza Request Failed")
#plt.xlabel("Upvotes - Downvotes")
#
#plt.subplot(2,1,2)
#plt.hist(x1, bins=5000)
#plt.title("Pizza Request Passed")
#plt.xlabel("Upvotes - Downvotes")
#
#plt.show()
#