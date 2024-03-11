#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Analyzing the dataset for orders sheet
order = pd.read_excel("Downloads\\Retail_data.xlsx", sheet_name="orders")
order.head()


# In[3]:


# Check the shape

order.shape


# In[4]:


# Check column info

order.info()


# In[5]:


# Check for any duplicated order id

order.order_id.duplicated().sum()


# In[6]:


# Check for duplicated customer id

order.customer_id.duplicated().sum()


# In[7]:


order.customer_id.value_counts().sort_values(ascending= False).head()


# In[8]:


order.head()


# In[9]:


# Check the order status field

order.order_status.value_counts(normalize = True) * 100


# Since almost 97% of the records are of delivered status, for this case study we are going to consider only the orders which have been successfully delivered.
# 
# 

# In[10]:


# Checking the types of status in order_status

order["order_status"].unique() 


# In[11]:


# Keep the "delivered" orders and drop the rest

orders = order[order.order_status == 'delivered']


# In[12]:


# Check the shape again

orders.shape


# In[13]:


# Check for the order_status field

orders.order_status.value_counts()


# In[14]:


#Now, the order_status field has only one value that is 'delivered'. So, we are good to proceed.
# Check for any missing value

orders.isna().sum().sort_values(ascending=False)


# In[15]:


# We have two columns with missing values. Those are 'order_approved_at' and 'order_delivered_timestamp'.

# Now, we can treat these missing values 

# We can assume that the order approval time and order delivery timestamp to be equivalent to/same as the
# order purchase timestamp and order estimated delivery date respectively.In this case it would be our best and 
#safest treatment method without losing the entire rows of data.


# In[16]:


# Replace the missing values

orders.order_approved_at.fillna(orders.order_purchase_timestamp, inplace=True)

orders.order_delivered_timestamp.fillna(orders.order_estimated_delivery_date, inplace=True)


# In[17]:


# Check for any missing value again

orders.isna().sum().sort_values(ascending=False)


# In[18]:


#'Order_items' Worksheet :
# Read the order_items data
order_items = pd.read_excel("Downloads\\Retail_data.xlsx", sheet_name="order_items")
order_items.head()


# In[19]:


# Check the shape

order_items.shape


# In[20]:


# Check column info

order_items.info()


# In[21]:


# Check for any duplicates

order_items[['order_id','order_item_id']].duplicated().sum()


# In[22]:


# Check for missing values

order_items.isna().sum().sort_values(ascending=False)


# In[23]:


#Customers' Worksheet :
# Read customers data

customers = pd.read_excel("Downloads\\Retail_data.xlsx", sheet_name="customers")
customers.head()


# In[24]:


# Check the shape

customers.shape


# In[25]:


# Check column info

customers.info()


# In[26]:


# Check for any duplicates

customers.customer_id.duplicated().sum()


# In[27]:


#if we want to see which values are duplicated

customers [ customers.customer_id.duplicated() ]


# In[28]:


#To get rid of the duplicate records, we will only keep the first occurance of any such value and drop the rest 
# Drop duplicate customer ids, keep only the first occurance

customers.drop_duplicates(subset="customer_id", keep="first", inplace=True)


# In[29]:


# Check the shape again

customers.shape


# In[30]:


# Check for any more duplicates

customers.customer_id.duplicated().sum()


# In[31]:


# Check for missing vlaues

customers.isna().sum().sort_values(ascending=False)


# In[32]:


#'Payments' Worksheet :
# Read the payments data

payments = pd.read_excel("Downloads\\Retail_data.xlsx", sheet_name="payments")
payments.head()


# In[33]:


# Check column info

payments.info()


# In[34]:


# Check for any duplicates

payments[['order_id','payment_sequential']].duplicated().sum()


# In[35]:


# Check for missing values

payments.isna().sum().sort_values(ascending=False)


# In[36]:


#'Products' Worksheet :
# Read products data

products = pd.read_excel("Downloads\\Retail_data.xlsx", sheet_name="products")
products.head()


# In[37]:


# Check the shape

products.shape


# In[38]:


# Check column info

products.info()


# In[39]:


# Check for any duplicates

products.product_id.duplicated().sum()


# In[40]:


# Check for missing values

products.isna().sum().sort_values(ascending=False)


# In[41]:


# Check the value count of product_category_name column

products.product_category_name.value_counts(normalize=True)*100


# In[42]:


#Since 75% of the data belongs to 'toys' category, we can replace the missing values of product_category_name column with 'toys'
# Replace the missing value with mode i.e. "toys"
products.product_category_name.fillna(products.product_category_name.mode()[0], inplace=True)


# In[43]:


# Check for missing values again

products.isna().sum().sort_values(ascending=False)


# In[44]:


#describe the data to see the distribution and other statistical values for these columns.
# Describe

products.describe()


# In[45]:


# Check the distribution

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
sns.distplot(products.product_weight_g)
plt.axvline(products.product_weight_g.mean(), color="red")
plt.axvline(products.product_weight_g.median(), color="green")
#plt.show()

plt.subplot(2,2,2)
sns.distplot(products.product_length_cm)
plt.axvline(products.product_length_cm.mean(), color="red")
plt.axvline(products.product_length_cm.median(), color="green")
#plt.show()

plt.subplot(2,2,3)
sns.distplot(products.product_height_cm)
plt.axvline(products.product_height_cm.mean(), color="red")
plt.axvline(products.product_height_cm.median(), color="green")
#plt.show()

plt.subplot(2,2,4)
sns.distplot(products.product_width_cm)
plt.axvline(products.product_width_cm.mean(), color="red")
plt.axvline(products.product_width_cm.median(), color="green")

plt.show()


# In[46]:


# Check the distribution

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
sns.boxplot(y= products.product_weight_g)

plt.subplot(2,2,2)
sns.boxplot(y= products.product_length_cm)

plt.subplot(2,2,3)
sns.boxplot(y= products.product_height_cm)

plt.subplot(2,2,4)
sns.boxplot(y= products.product_width_cm)


plt.show()


# In[47]:


#the data is right-skewed and there is no  outlier,
# we will use the median instead of mean to replace the missing values.

# Consider only the numerical columns for the missing value treatment

ncols = products.describe().columns.to_list()
ncols


# In[48]:


for i in ncols:
    products[i].fillna(products[i].median(), inplace=True)


# In[49]:


# Check for missing values again

products.isna().sum().sort_values(ascending=False)


# In[50]:


#we need to merge all the data sheets in to one excel file and extract it.

#Preparation of data for Market Basket Analysis by exporting data to excel file:

#We will use the cleaned data set to prepare data for Market Basket analysis.

# Create a Pandas Excel writer using XlsxWriter as the engine

ex_writer = pd.ExcelWriter('Retail_Dataset_Cleaned.xlsx', engine='xlsxwriter')


# In[51]:


# Write each dataframe to a different worksheet.

orders.to_excel(ex_writer, sheet_name='Orders', index = False)
order_items.to_excel(ex_writer, sheet_name='Order_items', index = False)
customers.to_excel(ex_writer, sheet_name='Customers', index = False)
payments.to_excel(ex_writer, sheet_name='Payments', index = False)
products.to_excel(ex_writer, sheet_name='Products', index = False)


# In[52]:


ex_writer.save()
print("Files exported successfully.")


# In[53]:


# check if the export was successful
import os
os.getcwd()


# In[54]:


#  create a new dataframe for Delivered orders merging all the others sheets

Del_orders= pd.merge(orders,order_items,how='inner',on='order_id')


# In[55]:


Del_orders.head()


# In[56]:


Del_orders.shape


# In[57]:


Del_orders.isnull().sum()


# In[58]:


#Merging 'Del_orders' with 'Products_sheet'

Delivered_orders = pd.merge(Del_orders,products,how='inner',on='product_id')


# In[59]:


Delivered_orders.head()


# In[60]:


Delivered_orders.shape


# In[61]:


Delivered_orders.isnull().sum()


# In[62]:


#Merging 'Delivered_orders' with 'Payments_sheet'

Delivered_orders = pd.merge(Delivered_orders,payments,how='inner',on='order_id')


# In[63]:


Delivered_orders.head()


# In[64]:


Delivered_orders.shape


# In[65]:


Delivered_orders.isnull().sum()


# In[66]:


#Merging 'Del_orders' with 'Customers_sheet'

Del_orders = pd.merge(Delivered_orders,customers,how='inner',on='customer_id')


# In[67]:


Del_orders.head()


# In[68]:


Del_orders.shape


# In[69]:


Del_orders.isna().sum()


# In[70]:


Delivered_orders.columns


# In[71]:


Del_orders.head()


# In[72]:


get_ipython().system('pip install mlxtend')
#Market Basket Analysis
#Creating a new dataframe with only the required columns for analysis

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MBA_set = Delivered_orders[['order_id','product_category_name', 'order_item_id']]


# In[73]:


#Creating a new dataframe with only the required columns for analysis

MBA_set.shape


# In[74]:


#Checking the duplicates after updating

MBA_set.duplicated().sum()


# In[75]:


#Dropping the duplicates keeping the first occurence

MBA_set.drop_duplicates(keep='first', inplace=True)


# In[76]:


MBA_set.shape


# In[77]:


MBA_set.info()


# In[78]:


#Again Creating a new dataframe using pandas pivot

New_MBA = pd.pivot_table(data=MBA_set,index='order_id',columns='product_category_name',
                              values='order_item_id',fill_value=0)


# In[79]:


New_MBA.head()


# In[80]:


New_MBA.info()


# In[81]:


MBA_set.product_category_name.value_counts().nlargest(10)


# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
b=MBA_set.product_category_name.value_counts().nlargest(20)
# Checking most ordered product
MBA_set.product_category_name.value_counts().nlargest(20).plot(kind='bar', figsize=(10,4))
plt.title("TOP 20 most number of product delivered")
plt.ylabel('Count')
plt.xlabel('Category');
b


# In[83]:


#plotting the null value percentage
sns.set_style("white")
fig = plt.figure(figsize=(12,5))
null_lead = pd.DataFrame((MBA_set.isnull().sum())*100/MBA_set.shape[0]).reset_index()
ax = sns.pointplot("index",0,data=null_lead)
plt.xticks(rotation =90,fontsize =9)
ax.axhline(45, ls='--',color='red')
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# In[84]:


#Strategy: Here we can see that all columns have below 45% of null value, so we don't have to drop them

MBA_set['product_category_name'].value_counts()[:50].plot(kind='bar', figsize=(15,5))


# In[85]:


#sns.pairplot(New_MBA)


# In[86]:


#For basket analysis converting/encoding the data to 1s and 0s 
def encode_data(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
MBA_Data_encode = New_MBA.applymap(encode_data)


# In[87]:


#Identifying the product category which are ordered more than 5 times and dropping the product category which are ordered less than 5 times

for column in MBA_Data_encode.columns:
    if (MBA_Data_encode[column].sum(axis=0, skipna=True)<=5):
        MBA_Data_encode.drop(column, inplace=True, axis=1)


# In[88]:


MBA_Data_encode.shape


# In[89]:


#identify the combinations of product categories which are frequently ordered together
MBA_Data_encode =MBA_Data_encode[(MBA_Data_encode>0).sum(axis=1)>=2]
MBA_Data_encode.head()


# In[90]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[91]:


# Now, we need to call apriori for creating frequently bought item with support = 3%(0.03)

frequent_itemlist = apriori(MBA_Data_encode, min_support=0.03, use_colnames=True)
frequent_itemlist.head()


# In[93]:


# apply association rules on frequent itemset to find product combinations. 

Confidence_data = association_rules(frequent_itemlist, metric="confidence", min_threshold=0.1)
Confidence_data


# In[94]:


Lift_data=Confidence_data[(Confidence_data['lift'] > 1)]
Lift_data


# In[107]:


#Extracting the market basket data to be visualized

# call apriori function and pass minimum support here we are passing 0.03%. 
# means 0.03 times in total number of transaction the item should be present.


# In[104]:


frequent_itemlist["itemsets"] = frequent_itemlist["itemsets"].apply(lambda x: ', '.join(list(x))).astype("unicode")
Confidence_data["antecedents"] = Confidence_data["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
Confidence_data["consequents"] = Confidence_data["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
Lift_data["antecedents"] = Lift_data["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
Lift_data["consequents"] = Lift_data["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode") 


# In[109]:


with pd.ExcelWriter(r"MBA_Dataset.xlsx") as excel_sheets:    
    frequent_itemlist.to_excel(excel_sheets, sheet_name="support", index=False)
    Confidence_data.to_excel(excel_sheets, sheet_name="confidence", index=False)
    Lift_data.to_excel(excel_sheets, sheet_name="lift", index=False)


# In[ ]:




