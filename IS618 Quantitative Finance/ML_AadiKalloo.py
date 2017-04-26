#By Aadi Kalloo
#This script introduces an ad-hoc joint neural network / randomForests algorithm
# using two stocks: Priceline and Berkshire Hathaway. While I wouldn't necessarily
# use this practice in a real trade setting, I thought it would be more 
# educational to manually implement the randomForests algorithm. 
# This code is meant for use in Quantopian and produces alpha=0.41, sharpe=1.29 during backtesting




from random import seed
from random import randrange
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from collections import deque
import numpy as np

def initialize(context):
    #define two stocks -- priceline and berkshire hathaway:
    context.stocks = {'PL': sid(19917), 'BH': sid(1091)}
    #set a moving average window of 10 days
    context.window_length = 10
    #define logistic regression, neural network, and randomForests algorithms
    #along with pipeline for classifier
    log_reg = linear_model.LogisticRegression()
    n_net = BernoulliRBM(n_components=512, learning_rate=0.1, batch_size=10, n_iter=10, verbose=False, random_state=0)
    context.classifier = Pipeline(steps=[('n_net', n_net), ('log_reg', log_reg)])     
    #declare dependent and independent variable arrays
    context.X = []
    context.Y = []
    context.w = 30
    #set blocks of pricing data
    context.recent_prices = deque(maxlen=context.window_length+2)
    #since there are two stocks, everything below is defined twice -- one for each stock
    #this includes input data, output, and prediction holding variables
    context.X = deque(maxlen=1000)
    context.Y = deque(maxlen=1000)
    context.prediction = 0
    context.X2 = deque(maxlen=1000)
    context.Y2 = deque(maxlen=1000)
    context.prediction2 = 0
    #run algorithm every day at market open and 5 hours after market open
    schedule_function(scheduled_fun, date_rules.every_day(), time_rules.market_open())
    schedule_function(scheduled_fun, date_rules.every_day(), time_rules.market_open(minutes=300))

def scheduled_fun(context, data):
    #closing price change data for both stocks
    changes = np.diff(data.history((context.stocks)['PL'], 'close', context.w, '1d')) > 0
    context.X.append(changes[:-1])
    context.Y.append(changes[-1])
    changes2 = np.diff(data.history((context.stocks)['BH'], 'close', context.w, '1d')) > 0
    context.X2.append(changes2[:-1])
    context.Y2.append(changes2[-1])
    if len(context.Y) >= 100:
        #fit data and predict
        context.classifier.fit(np.array(context.X), np.array(context.Y))
        context.prediction = context.classifier.predict(changes[1:])
        #buy based on prediction %
        order_target_percent((context.stocks)['PL'], context.prediction)
        #short if prediction is particularly bad
        if context.prediction < 0.1:
            order_target_percent((context.stocks)['PL'], -context.prediction)
        record(priceline=int(context.prediction))
    #repeat above steps for second stock
    if len(context.Y2) >= 100:
        #use manual random forests implementation
        context.prediction2 = random_forest(np.array(context.X2), np.array(context.Y2))
        context.prediction2 = int(context.prediction2[0])
        order_target_percent((context.stocks)['BH'], context.prediction2*0.5)
        if context.prediction2 < 0.1:
            order_target_percent((context.stocks)['PL'], -context.prediction2)
        record(berkhathaway=int(context.prediction2))

# Split a data based on an attribute and an attribute value
def split_pred(index, value, data):
    left_branch, right_branch = list(), list()
    for record in data:
        if record[index] < value:
            left_branch.append(record)
        else:
            right_branch.append(record)
    return left_branch, right_branch
 
# Calculate the Gini index for a split data
def get_gn_idx(groups, all_classes):
    gini = 0.0
    for single_class in all_classes:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [record[-1] for record in group].count(single_class) / float(size)
            gini = gini + (proportion * (1.0 - proportion))
    return gini

# Select the best split point for a data
def determine_split(data, n_features):
    all_classes = list(set(record[-1] for record in data))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(data[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for record in data:
            groups = split_pred(index, record[index], data)
            gini = get_gn_idx(groups, all_classes)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, record[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def create_term_node(group):
    outcomes = [record[-1] for record in group]
    return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left_branch, right_branch = node['groups']
    del(node['groups'])
    # check for a no split
    if not right_branch or not left_branch:
        node['left_branch'] = node['right_branch'] = create_term_node(left_branch + right_branch)
        return
    # check for max depth
    if depth >= max_depth:
        node['left_branch'], node['right_branch'] = create_term_node(left_branch), create_term_node(right_branch)
        return
    if len(right_branch) <= min_size:
        node['right_branch'] = create_term_node(right_branch)
    else:
        node['right_branch'] = determine_split(right_branch, n_features)
        split(node['right_branch'], max_depth, min_size, n_features, depth+1)
    # process left_branch child
    if len(left_branch) <= min_size:
        node['left_branch'] = create_term_node(left_branch)
    else:
        node['left_branch'] = determine_split(left_branch, n_features)
        split(node['left_branch'], max_depth, min_size, n_features, depth+1)

# Build decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = determine_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root
 
# Make a prediction using tree
def predict(node, record):
    if record < node['value']:
        if isinstance(node['left_branch'], dict):
            return predict(node['left_branch'], record)
        else:
            return node['left_branch']
    else:
        if isinstance(node['right_branch'], dict):
            return predict(node['right_branch'], record)
        else:
            return node['right_branch']
 
# Create a random subsample from the data with replacement
def subsample(data, ratio):
    sample = list()
    n_sample = round(len(data) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index])
    return sample
 
# Make a prediction with a list of bagged trees
def bagging_predict(trees, record):
    predictions = [predict(tree, record) for tree in trees]
    return max(set(predictions), key=predictions.count)
 
# Random Forest Algorithm
def random_forest(train, test, max_depth=10, min_size=1, sample_size=1, n_trees=1, n_features=20):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, record) for record in test]
    return(predictions)
 
