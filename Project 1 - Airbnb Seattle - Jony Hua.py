#!/usr/bin/env python
# coding: utf-8

# # Project: Write a Data Science Blog Post - Airbnb Seattle

# In[36]:


# prepare packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from datetime import datetime
from datetime import date
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# # 1. Business Understanding

# Since 2008, guests and hosts have used Airbnb to travel in a more unique, personalized way. As part of the Airbnb Inside initiative, this dataset describes the listing activity of homestays in Seattle, WA.
# 

# Questions to address
# 1. What are the busiest times of the year to visit Seattle? By how much do prices spike?
# 2. Can you describe the vibe of each Seattle neighborhood using listing descriptions?
# 3. What are the factors influencing review score?

# # 2. Data Extraction

# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


listing = pd.read_csv('listings.csv')
listing.head()


# In[4]:


listing.describe()


# In[5]:


reviews = pd.read_csv('reviews.csv')
reviews.head()


# In[6]:


calendar = pd.read_csv('calendar.csv')
calendar.head()


# In[7]:


calendar.describe(include='all')


# # 3 Prepare Data

# # 3.1 Prepare Data and Analysis for Question 1

# In[8]:


calendar_reduced=calendar.drop(['price'],axis=1)


# In[9]:


calendar_reduced=calendar_reduced.groupby(['date','available']).count()


# In[10]:


calendar_reduced.head()


# In[11]:


#designing a metric to measure Airbnb popularity
calendar_reduced["avail_to_unavail_ratio"] = calendar_reduced.groupby(level=0)['listing_id'].transform(lambda x: x[1]/ x[0])
calendar_reduced=calendar_reduced.drop('listing_id',axis=1)
calendar_reduced=calendar_reduced.reset_index(level=1, drop=True)
calendar_reduced=calendar_reduced.drop_duplicates()
calendar_reduced


# In[12]:


calendar_reduced.reset_index(level=0, inplace=True)
calendar_reduced['date']  = pd.to_datetime(calendar_reduced['date'])
sns.lineplot(x="date", y="avail_to_unavail_ratio", data= calendar_reduced)


# Answer to Question 1: The Seattle Airbnb is busiest during the summer, from July to Aug. 

# In[13]:


calendar_reduced_2=calendar.drop(['available','listing_id'],axis=1)
calendar_reduced_2=calendar_reduced_2.dropna(how='any')


# In[14]:


def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)


# In[15]:


calendar_reduced_2['price'] = calendar_reduced_2['price'].apply(clean_currency).astype('float')
calendar_reduced_2.groupby(['date']).mean()
calendar_reduced_2['date']  = pd.to_datetime(calendar_reduced_2['date'])
sns.lineplot(x="date", y="price", data= calendar_reduced_2)


# Answer to Question 1: The price increases on average about 10% during that time.

# # 3.2 Prepare Data and Analysis for Question 2

# In[16]:


listing.head()


# In[17]:


listing.shape


# In[18]:


# Getting key features of this analysis
listing_reduced=listing[['id','name','summary','description','neighborhood_overview','neighbourhood_group_cleansed','property_type','room_type','review_scores_rating']]


# In[19]:


listing_reduced.head()


# In[20]:


neighbourhood=listing_reduced.groupby('neighbourhood_group_cleansed')


# In[21]:


neighbourhood.mean().sort_values(by="review_scores_rating",ascending=False).head(10)


# In[22]:


listing_reduced.neighbourhood_group_cleansed.unique()


# In[23]:


# Getting the list of neighbourhood of interest
neighbourhood=['Downtown','University District','Central Area','Ballard','Magnolia',
            'Rainier Valley','Beacon Hill','Cascade']



# Plotting wordcloud for each neighbourhood
for n in neighbourhood:  
    text = listing_reduced[listing_reduced['neighbourhood_group_cleansed']==n].description.values
    stopwords = set(STOPWORDS)
    stopwords.update(["house", "Seattle", "seattle", "home", "room","access","bedroom","kitchen","apartment","one","will","one","guest","minute","private","downtown","space","bed","two","bathroom","guests","minutes","available"])
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(str(text))
    plt.title(n)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# Looks like we have many insights!

# # 3.3 Exploratory Analysis and Data Cleaning for Modeling

# In[103]:


listing.head()


# In[53]:


# reduce to core features, considering null values and correlations covered later
listing_df=listing[['host_id','host_since','host_response_time','host_response_rate','host_about','host_acceptance_rate','host_is_superhost',
                    'room_type','host_neighbourhood','host_listings_count','host_total_listings_count','host_verifications','host_has_profile_pic','host_identity_verified',
                    'neighbourhood_cleansed',
                    'neighbourhood_group_cleansed','zipcode','is_location_exact','property_type','room_type','accommodates','bathrooms','bedrooms','beds',
                    'bed_type','amenities','square_feet','price','weekly_price','monthly_price','security_deposit','cleaning_fee','guests_included',
                    'extra_people','minimum_nights','maximum_nights','calendar_updated','has_availability','availability_30','availability_60','availability_90',
                    'availability_365','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
                    'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','cancellation_policy',
                    'instant_bookable','requires_license','require_guest_profile_picture','require_guest_phone_verification','calculated_host_listings_count',
                    'reviews_per_month']]


# In[54]:


listing_df.head()


# In[27]:


listing_df.shape


# In[28]:


# checking null values
np.sum(listing_df.isnull())/listing_df.shape[0]


# In[29]:


listing_df.dtypes


# In[30]:


corr=listing_df.corr() # Calculates correlation matrix

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(26, 26))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,vmin=-1,
            annot=True,square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Based on high correlations, we dropped these columns: host_total_listings_count, accommodates, availability_60, availability_90

# In[55]:


listing_df=listing_df.drop(columns=['host_total_listings_count','accommodates','availability_60','availability_90'])


# In[45]:


listing_df.review_scores_rating.hist(bins=100)


# Taking a look at the distribution of score. 

# In[56]:


# dropping variables related to review_scores to avoid leakage
listing_df=listing_df.dropna(subset=['review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value'])


# In[57]:


#listing_df.head()


# In[58]:


# dropping varaible that is not addding value
listing_df=listing_df.drop(columns=['square_feet','weekly_price','monthly_price','security_deposit','host_about','host_verifications','security_deposit','host_id'])


# In[59]:


# creating a feature as days since hosting from the host since date
listing_df['days_since_host']=(datetime.today()- pd.to_datetime(listing_df['host_since'])).dt.days
listing_df=listing_df.drop(columns=['host_since'])


# In[60]:


# we corrected the formating of percentage numbers from string to float
s = listing_df['host_response_rate'].str.replace(r'%', r'').astype('float')/100


# now we convert the host_response_rate into numeric. As the original number has a '%', we will get all NAs, therefore we are filling the numbers with s variable created above
listing_df['host_response_rate'] = pd.to_numeric(listing_df['host_response_rate'], errors='coerce').fillna(s)


# In[61]:


# we corrected the formating of percentage numbers from string to float
s =listing_df['host_acceptance_rate'].str.replace(r'%', r'').astype('float')/100

# now we convert the host_acceptance_rate into numeric. As the original number has a '%', we will get all NAs, therefore we are filling the numbers with s variable created above
listing_df['host_acceptance_rate'] = pd.to_numeric(listing_df['host_acceptance_rate'], errors='coerce').fillna(s)


# In[62]:


listing_df.head()


# In[63]:


# cleaning currency variable
listing_df['price'] = listing_df['price'].apply(clean_currency).astype('float')
listing_df['cleaning_fee'] = listing_df['cleaning_fee'].apply(clean_currency).astype('float')
listing_df['extra_people'] = listing_df['extra_people'].apply(clean_currency).astype('float')


# In[64]:


listing_df.head()


# In[65]:


corr=listing_df.corr() # Calculates correlation matrix

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(26, 26))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,vmin=-1,
            annot=True,square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looks good to me

# In[66]:


# creating dummies for ameniities field
listing_df_comb = pd.concat([listing_df, listing_df['amenities'].str.get_dummies(sep=',').add_prefix('amenities_')], axis = 1).drop(['amenities','amenities_{}'], 1)


# In[102]:


def create_dummy_df(df, cat_cols,num_cols, dummy_na):
        '''
        INPUT
        df - pandas dataframe 
        cat_cols - columns that are categories
        num_cols - columns that are numeric
        dummy_na - boolean for creating column for na
        
        OUTPUT
        df - pandas dataframe 
        
        Create dummy columns for all the categorical variables in X, drop the original columns
        '''

        for col in cat_cols:
            try:
                df=pd.concat([df.drop(col, axis=1),pd.get_dummies(df[col],prefix=col,drop_first=True,dummy_na=dummy_na)],axis=1)
            except:
                continue
                
        for col in num_cols:
            try:
                fill_mean = lambda col: col.fillna(col.mean())
                # given data availability is good and lack of other feasible solutions, we will use mean to fill na
                df=pd.concat([df.drop(col, axis=1),df[[col]].apply(fill_mean, axis=0)],axis=1)
            except:
                continue
        return df


# In[68]:


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    df=df.dropna(subset=['review_scores_rating'])
    y=df['review_scores_rating']
    X=df.drop(columns=['review_scores_rating'])
    X=X.drop(columns=['calendar_updated'])
    
    cat_df = X.select_dtypes(include=['object']).copy()
    cat_cols_lst = cat_df.columns
    
    num_df = X.select_dtypes(include=['float64','int64']).copy()
    num_cols_lst = num_df.columns
    
    X = create_dummy_df(X, cat_cols_lst, num_cols_lst,dummy_na=False) #Use your newly created function


    
    
    return X, y
    
#Use the function to create X and y
X, y = clean_data(listing_df_comb)    


# In[69]:


X.head()


# In[70]:


#checking na
sum(np.sum(X.isnull())>0)


# In[71]:


y.head()


# # 4. Modeling - 1) Linear Regression

# In[72]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.metrics as metrics


# In[73]:


listing_df.head()


# In[77]:


# reducing features for linear regression 
listing_df_lm_reduced=listing_df.drop(['amenities','has_availability','availability_30','availability_365',
                                       'number_of_reviews','reviews_per_month'],axis=1)


# In[78]:


listing_df_lm_reduced=listing_df[['host_response_rate','host_response_time','price',
                                  'require_guest_phone_verification','cancellation_policy','host_acceptance_rate',
                                  'host_is_superhost','days_since_host','host_has_profile_pic',
                                  'review_scores_rating',
                                 'require_guest_profile_picture','instant_bookable',
                                 'bed_type',
                                  'host_listings_count']]


# In[79]:


sns.heatmap(listing_df_lm_reduced.corr(), annot=True, fmt=".2f");


# Looks good

# In[80]:


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    df=df.dropna(subset=['review_scores_rating'])
    y=df['review_scores_rating']
    X=df.drop(columns=['review_scores_rating'])
    
    '''X=X.drop(columns=['calendar_updated','beds','review_scores_accuracy','review_scores_cleanliness',
                      'review_scores_checkin','review_scores_communication','review_scores_location',
                      'review_scores_value','neighbourhood_cleansed','neighbourhood_group_cleansed',
                     'zipcode','host_neighbourhood'])
                     '''
    
    cat_df = X.select_dtypes(include=['object']).copy()
    cat_cols_lst = cat_df.columns
    
    num_df = X.select_dtypes(include=['float64','int64']).copy()
    num_cols_lst = num_df.columns
    
    X = create_dummy_df(X, cat_cols_lst, num_cols_lst,dummy_na=False) #Use your newly created function


    
    
    return X, y
    
#Use the function to create X and y
X, y = clean_data(listing_df_lm_reduced) 


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[84]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[85]:


X_train.head()


# In[86]:


clf = LinearRegression(normalize=True)
clf.fit(X_train,y_train)


# In[87]:


y_test_preds = clf.predict(X_test)


# # 5. Model Evalution - Linear Regression

# In[88]:


print(r2_score(y_train, clf.predict(X_train)))
print(r2_score(y_test, y_test_preds)) #In this case we are predicting a continuous, numeric response.  Therefore, common
print(mean_squared_error(y_test, y_test_preds)) #metrics to assess fit include Rsquared and MSE.


# R-square about 10% and MSE is about 36.3

# In[91]:


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = clf.coef_
    coefs_df['abs_coefs'] = np.abs(clf.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(clf.coef_, X_train)

#A quick look at the top results
coef_df.sort_values('coefs', ascending=False)


# In[92]:


# running OLS to validate results seen, and present the confidence intervals for variables
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train)
results = sm.OLS(y_train,X_train_new).fit()
results.summary()  


# # 4. Modeling - 2) Gradiant Boost

# In[93]:


from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance


# In[94]:


listing_df_comb.head()


# In[95]:


listing_df.head()


# In[96]:


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    
    Perform to obtain the correct X and y objects
    This function cleans df using the following steps to produce X and y:
    1. Drop all the rows with no salaries
    2. Create X as all the columns that are not the Salary column
    3. Create y as the Salary column
    4. Drop the Salary, Respondent, and the ExpectedSalary columns from X
    5. For each numeric variable in X, fill the column with the mean value of the column.
    6. Create dummy columns for all the categorical variables in X, drop the original columns
    '''
    df=df.dropna(subset=['review_scores_rating'])
    y=df['review_scores_rating']
    X=df.drop(columns=['review_scores_rating'])
    
    X=X.drop(columns=['review_scores_accuracy','review_scores_cleanliness',
                      'review_scores_checkin','review_scores_communication','review_scores_location',
                      'review_scores_value','neighbourhood_cleansed','neighbourhood_group_cleansed',
                     'zipcode','host_neighbourhood',
                     'has_availability','availability_30','availability_365',
                                       'number_of_reviews','reviews_per_month','calendar_updated','amenities','host_listings_count'])
                     
    #'calendar_updated','beds'
    
    cat_df = X.select_dtypes(include=['object']).copy()
    cat_cols_lst = cat_df.columns
    
    num_df = X.select_dtypes(include=['float64','int64']).copy()
    num_cols_lst = num_df.columns
    
    X = create_dummy_df(X, cat_cols_lst, num_cols_lst,dummy_na=False) #Use your newly created function


    
    
    return X, y
    
#Use the function to create X and y
X_2, y_2 = clean_data(listing_df) 


# In[97]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2, y_2, test_size=0.3, random_state=42)


# hyperparameters
params = {'n_estimators': 220,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}


# In[98]:


print(X_train_2.shape)
print(y_train_2.shape)
print(X_test_2.shape)
print(y_test_2.shape)


# # 5. Model Evaluation - Gradiant Boost

# In[99]:


reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train_2, y_train_2)

mse = mean_squared_error(y_test_2, reg.predict(X_test_2))
y_pred_2=reg.predict(X_test_2)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# Slightly better than linear regression!

# In[100]:


test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred_2 in enumerate(reg.staged_predict(X_test_2)):
    test_score[i] = reg.loss_(y_test_2, y_pred_2)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()


# 220 is about the right number of estimators

# In[101]:


feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(20, 30))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X_train_2.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, X_test_2, y_test_2, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(X_train_2.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


# Answer for Question 3: The most important features are: calculated_host_listings_count,host_is_superhost, days_since_host. Looks like experience matters a lot! Price and other variables also play a role.

# In[ ]:




