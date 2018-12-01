
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import datetime


# # Raw Data Preparation


# ### listings


# In[2]:


listings = pd.read_csv('listings.csv')


# In[3]:


listings.columns


# ### community districts


# In[5]:


# with open('community_district_data.csv') as fp:
#     type_list = list()
#     for line in fp:
#         print(line)
    #data = fp.read


# ### reviews

# In[6]:


reviews = pd.read_csv('reviews.csv')


# In[7]:


reviews


# # Data preprocessing

# In[5]:


df_geo = pd.DataFrame(columns=['listing_id','latitude','longitude',
                           'zipcode','borough'])

df_geo['listing_id'] = listings['id']
df_geo['latitude'] = listings['latitude'].astype('float',copy=False)
df_geo['longitude'] = listings['longitude'].astype('float',copy=False)
#df_geo['zipcode'] = listings['zipcode'].astype('float',copy=False)


# In[6]:


df_geo.columns


# In[8]:


# check values of each col
print(df_geo['longitude'].isnull().sum())
print(df_geo['latitude'].isnull().sum())

# zip code cannot be used because of too many missing values & wrong values
print(df_geo['zipcode'].isnull().sum())

# each listing id is an unique primary key
print(len(df_geo['listing_id'].unique()))


# In[51]:


get_ipython().system(' pip install uszipcode')


# In[9]:


# remap zip code for each row
from uszipcode import SearchEngine
from uszipcode import Zipcode
search = SearchEngine(simple_zipcode=True)

for i in range(len(df_geo)):
#for i in range(5):
    lat = df_geo.loc[i,'latitude']
    lng = df_geo.loc[i,'longitude']
    result = search.by_coordinates(lat, lng, radius=30, returns=1)
    df_geo.loc[i,'zipcode'] = result[0].zipcode

# save zip code data
df_geo.to_csv('dataframe_geography.csv')


# In[ ]:


# load and process zipcode-suborough-borough mapping dataframe
df_bor = pd.read_excel('borough.xlsx')
df_bor['zip_list'] = df_bor['zipcode'].astype(str,copy=False).str.split(',')

df_area = pd.DataFrame(columns = ['zipcode','sub_borough','borough'])
for index in range(len(df_bor)):
    a = df_bor.loc[index,'borough']
    b = df_bor.loc[index,'sub_borough']
    for i in range(len(df_bor.loc[index,'zip_list'])):
        c = df_bor.loc[index,'zip_list'][i]
        df_area = df_area.append({'zipcode':c,'sub_borough':b,'borough':a},ignore_index=True)

# save geo_map df
df_area.to_csv('geo_map.csv')


# In[31]:


# join geo dataframe with map dataframe
df_geo = pd.read_csv('dataframe_geography.csv',index_col = ['Unnamed: 0'])
df_map = pd.read_csv('geo_map.csv',index_col = ['Unnamed: 0'])
df_geo.drop('borough',axis=1,inplace=True)
df_geo = df_geo.set_index('zipcode').join(df_map.set_index('zipcode'),on = 'zipcode')

# save new joined dataframe into new geo df
df_geo = df_geo.loc[df_geo['sub_borough'].isnull() == False]
df_geo.to_csv('dataframe_geography.csv')


# ## df_review

# In[8]:


# drop reviews with no comments
reviews = reviews[reviews['comments'].isnull()==False]
reviews.reset_index(drop=True,inplace=True)


# In[9]:


reviews


# In[10]:


# consolidate reviews
dict_rev = dict()
for i in range(len(reviews)):
    listing_id = reviews.loc[i,'listing_id']
    if dict_rev.get(str(listing_id)) == None:
        dict_rev[str(listing_id)] = reviews.loc[i,'comments']
    else:
        dict_rev[str(listing_id)] = dict_rev[str(listing_id)] + reviews.loc[i,'comments']


# In[ ]:


# change dict to dataframe and save
reviews_text = pd.DataFrame.from_dict(dict_rev,orient = 'index',columns = ['review_text'])
reviews_text.to_csv('review_text.csv')


# # Text mining

# ### prepare sampling data - randomly choose 10 listings from each borough

# In[52]:


# read from prepared data
reviews_text = pd.read_csv('review_text.csv')
df_geo = pd.read_csv('dataframe_geography.csv',index_col = ['Unnamed: 0'])


# In[65]:


# join review texts with geo map
df_review = df_geo.set_index('listing_id').join(reviews_text.set_index('listing_id'),on = 'listing_id')
df_review.drop(labels='index',axis=1,inplace=True)
df_review.reset_index(inplace=True)

# drop listing_id with no reviews
df_review = df_review[df_review['review_text'].isnull()==False]


# In[80]:


df_review.to_csv('review_for_text_mining.csv')


# In[78]:


# group by sub-boroughs
df_review.groupby('sub_borough')



# c = []
# for index in range(len(sample_review)):
#     a = sample_review.loc[index,'listing_id']
#     b = sample_review.loc[index,'review_text']
#     c.append((a,b))


# ### naive sentiment analysis

# In[ ]:


# naive sentiment analysis - functions
def get_pos_neg_words():
    def get_words(url):
        import requests
        words = requests.get(url).content.decode('latin-1')
        word_list = words.split('\n')
        #print(word_list)
        index = 0
        while index < len(word_list):
            word = word_list[index]
            if ';' in word or not word:
                word_list.pop(index)
            else:
                index+=1
        return word_list

    #Get lists of positive and negative words
    p_url = 'http://ptrckprry.com/course/ssd/data/positive-words.txt'
    n_url = 'http://ptrckprry.com/course/ssd/data/negative-words.txt'
    positive_words = get_words(p_url)
    negative_words = get_words(n_url)
    return positive_words,negative_words

positive_words,negative_words = get_pos_neg_words()

def do_pos_neg_sentiment_analysis(text_list,debug=False):
    positive_words,negative_words = get_pos_neg_words()
    from nltk import word_tokenize
    results = list()
    for text in text_list:
        cpos = cneg = lpos = lneg = 0
        for word in word_tokenize(text[1]):
            if word in positive_words:
                if debug:
                    print("Positive",word)
                cpos+=1
            if word in negative_words:
                if debug:
                    print("Negative",word)
                cneg+=1
        results.append((text[0],cpos/len(word_tokenize(text[1])),cneg/len(word_tokenize(text[1]))))
    return results

