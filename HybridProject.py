# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:36:10 2017

A hybrid approach including content-based and item-item CF.

Some standard offline evaluation metrics are included to evaluate
and improve different systems
@author: undeadstone
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math

# In[General Comments on the two systems]
'''
Goal is to create two types of systems, one is content-based system,
which include a user preference profiles and a item features with 
TFIDF included as weights. TF is term frequncy, which can be collected
through users' rating history and items' features. IDF is inverse 
document frequency, which can be obtained through all the items' 
features.

The other one is an item-item collabrotive filtering, which include 
an item-item similarity matrix where w[i,j] represents the similarity
between item i and j based on all users' rating on both item i and j.
'''
'''Both systems need a user rating history, which can be created using 
a dict (ex: ur_dic = {userid: movieid: rating} ) or a movie dict 
(ex: mr_dic = {movieid : userid: rating } )
'''
'''
In this case, there is no time-dependent data, 
one user got rate one movie only once. The reason why we do not
use a user-item matrix to store all the data is that it is a sparse
matrix, it is more memory efficient to not store unrated information.
'''   

'''
The original data is too big, lets try divide the data into several
folds so that we can do a cross-validation later. Try to divide data
into 10 folds. Let's train 10% first.
'''
'''
ur_dict = {}
for index in train_list:
    userid = user_rating.iloc[index, 0]
    movieid = user_rating.iloc[index, 1]
    rating = user_rating.iloc[index, 2]
    if userid not in ur_dic:
        ur_dic[userid] = {}
    if rating >= 0:
        ur_dic[userid][movieid] = rating 
'''
'''    
ur_mv_matrix = user_ratingsmall.pivot_table(index = 'user_id', \
                                            columns ='anime_id',\
                                            values = 'rating')

'''
def moviedict(user_rating, anime):
    '''
    Generate a movie dict contains userid and rating.
    Auguments:
        user_rating ------- DataFrame contains movieid, userid and rating
    Returns:
        mv_dict --- {movieid : userid: rating, ... }
    '''
    mv_dic = {}
    
    for index in range(user_rating.shape[0]):
        userid = user_rating.iloc[index, 0]
        movieid = user_rating.iloc[index, 1]
        rating = user_rating.iloc[index, 2]
        if movieid not in mv_dic:
            mv_dic[movieid] = {}
        if rating >= 0:
            mv_dic[movieid][userid] = rating
    
    for index in range(anime.shape[0]):
        movieid = anime.iloc[index, 0]
        if movieid not in mv_dic:
            mv_dic[movieid] = {}
    
    return mv_dic


def Nfoldmoviedict(mv_dic, n):
    '''
    Split user_rating into n folds, and generate training and
    testing dicts.
    Auguments:
        user_rating ---- DataFrame contains movieid, userid and rating
    Returns:
        mvtrain_dict --- movieid : userid: rating for training
        mvtest_dict ---  movieid : userid: rating for testing
    '''
    mvtrain_dic = {}
    mvtest_dic = {}
    
    for movie in mv_dic:
        np.random.seed(10)
        size = len(mv_dic[movie])
        test_size = int(size / n)
        mvtrain_dic[movie] = {}
        mvtest_dic[movie] = {}
        np.random.seed(0)
        if len(mv_dic[movie]):
            test_list = list(np.random.randint(0, len(mv_dic[movie]), test_size)) #contains index of user
        else:
            test_list = []
        userlist = list(mv_dic[movie].keys())
        for i in test_list:
            user = userlist[i]
            mvtest_dic[movie][user] = mv_dic[movie][user]
        for user in userlist:
            if userlist.index(user) not in test_list:
                mvtrain_dic[movie][user] = mv_dic[movie][user]
                        
    return mvtrain_dic, mvtest_dic


'''
Some thoughts on propressing the data on hadoop ecosystem?
'''
# In[Content-based system]
'''
In CB system, a user preference list is created simply based on 
anime's genre and type. In the future, episodes and general rating
can be included to improve the performance.
For now, a movie features dict 
(ex: mf = {movieid:featureid: TFIDf}) is created.
And a user preference dict 
(ex: up = {userid:featureid: weights})

In this dataset, only genre and type are considered as 
feature information, however, in this way, the term frequnce
in each movie will be the same, relative to the number of users 
who viewed the movie.  
'''
'''
A simple method to calculate all these, but memory not efficient
'''
def normalize(mf):
    '''
    Euclidean norm of the movie features vector.
    Auguments:
        mf -- a dictionary contains a dict with key = feature
              and value = tfidf     
    '''
    for i in mf:
        sq = 0
        for j in mf[i]:
            sq += mf[i][j]**2
        sq = math.sqrt(sq) 
        if sq:
            for j in mf[i]:
                mf[i][j] = mf[i][j] / sq
    return mf

def MovieFeatureProfile(anime, mv_dic):  
    '''
    Generate a movie feature profile
    Auguments:
        anime ------ DataFrame with movieid and feature information in each column
        mv_dict --- {movieid : userid: rating }
    Return:
        mf = {movieid:featureid: TFIDf}
    '''
    movie_size = anime.shape[0]
    N = movie_size
    mf = {}
    df = {}
    for i in range(movie_size):
        movieid = anime.iloc[i,0]
        n_user = len(mv_dic[movieid])
        if movieid not in mf:
            mf[movieid] = {}
        genre = anime.iloc[i,2]
        flist = []
        if genre and not pd.isnull(genre):
            flist = genre.split(',')    
        anitype = anime.iloc[i,3]
        if not pd.isnull(anitype):
            flist.append(anitype)
        for feature in flist:
            feature = feature.strip()
            mf[movieid][feature] = n_user
            if feature not in df:
                df[feature] = 1
            else:
                df[feature] += 1
    
    tfidf = [math.log(N / df[item]) for item in df]
    for i in mf:
        index = 0
        for j in mf[i]:        
            mf[i][j] = mf[i][j]*tfidf[index]
            index += 1
    
    mf_norm = normalize(mf)
    
    return mf_norm
            
# user profile
def UserFeatureProfile(anime, mv_dic):
    
    '''
    Generate a user feature profile
    Auguments:
        anime ------ DataFrame with movieid and feature information in each column
        mv_dict --- {movieid : userid: rating }
    Return:
        up = {userid:featureid: weights}
    '''
    up = {}
    threshold = 5
    movie_size = anime.shape[0]
    for i in range(movie_size):
        movieid = anime.iloc[i,0]
        for userid in mv_dic[movieid]:
            if userid not in up:
                up[userid] = {}
            if mv_dic[movieid][userid] > threshold:
                genre = anime.iloc[i,2]
                flist = []
                if genre and not pd.isnull(genre):
                    flist = genre.split(',')    
                    anitype = anime.iloc[i,3]
                    if not pd.isnull(anitype):
                        flist.append(anitype)
                        for feature in flist:
                            feature = feature.strip()
                            if feature not in up[userid]:
                                up[userid][feature] = 1
                            else:
                                up[userid][feature] += 1
    return up
'''
This user profile is based on the info from the movie that user
has rated. Thus for user who has not rated any movies, this system
cann't recommend anything. In the future, this profile can be improved
by looking for other sources from user such as comments, tags, 
blogs. 

The weights to the feature are based on ratings, if rate > 5, we 
will add 1 to the feature in the profile.
'''
# In[Import Data and Analysis]
anime = pd.read_csv("anime.csv")
'''
Attributes of anime:
anieme_id, name, genre, type, episodes, rating, members
'''
user_rating = pd.read_csv("rating.csv")
'''
Attributes of user_rating:
    user_id, anime_id, rating
'''               
# In[Metrics to evaluate Content-based system]
def predict(mf, up):
    '''
    Predict mvpredict_dic based on input
    Auguments:
        mf ------ movie feature {movieid:featureid: normalized (TFIDf) }
        up ------ user profile {userid:featureid: weights}
    Returns:
        mvpredict ----- {movieid : userid: rating } for predicted results
    '''
    mv = {}
    for movieid in mf:
        mv[movieid] = {}
        for userid in up:
            for feature in mf[movieid]:
                if feature in up[userid]:
                    if userid not in mv[movieid]:
                        mv[movieid][userid] = mf[movieid][feature]*up[userid][feature]
                    mv[movieid][userid] += mf[movieid][feature]*up[userid][feature]
                    
    return mv

def evaluate(mvtrain, mvtest):
    '''
    Calcluate the overall RMSE error between training and test dataset
    Arguments:
        mvtrain ----------- predicted results
        mvtest ------------ original results
    Return:
        rms --------------- rms error
    '''
    num = 0
    err = 0
    for movie in mvtest:
        for user in mvtest:
            if movie in mvtrain and user in mvtrain[movie]:
                err += (mvtrain[movie][user] - mvtest[movie][user])**2
                num += 1
    rms = math.sqrt(err/num)
    
    return rms    
np.random.seed(1)       
size = int(user_rating.shape[0] / 100)
test_list = list(np.random.randint(0, user_rating.shape[0], size))
user_ratingsmall = user_rating.loc[test_list] 
mv_dic = moviedict(user_ratingsmall, anime)
n = 5
mvtrain_dic, mvtest_dic = Nfoldmoviedict(mv_dic, n)
mf_norm = MovieFeatureProfile(anime, mvtrain_dic)
up = UserFeatureProfile(anime, mvtrain_dic)
mpre = predict(mf_norm, up)
rms = evaluate(mvtrain_dic, mpre)

# In[Collaborative filtering]
'''

'''








