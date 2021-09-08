# # Task
# 
# ## Exploratory Data analysis of AirBNB in New York City
# This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.
# What was done in this analysis are followes : 
# Import python package and load data.
# EDA and visualization.
# Geospatial analysis.
# Compare prices by region by looking at Airbnb on a New York City map.
# Analyse whether there is a Difference price by room type.
# Predict NYC Airbnb Rental Prices.

# Importng the necessary packages

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cufflinks as cf
import folium as fm
import folium
from folium.plugins import HeatMap
import holoviews as hv
import urllib


# Importing the csv file that was collected from kaggle

df = pd.read_csv(r'D:\Upal\Python\AirBNB\AB_NYC_2019.csv')


# Checkinf the data set
df

df.describe()

df.info()

df.isnull().sum()

df.sort_values(by=['price'], ascending=False)

# There are some appartments which has no price listed. 
df[df['price']<1]
# 11 appartments

dfbk = df[df['neighbourhood_group'].str.contains('Brooklyn')]

dfmn = df[df['neighbourhood_group'].str.contains('Manhattan')]

dfqn = df[df['neighbourhood_group'].str.contains('Queens')]


dfsi = df[df['neighbourhood_group'].str.contains('Staten Island')]


dfbx = df[df['neighbourhood_group'].str.contains('Bronx')]


fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=df['neighbourhood_group'],y=df['price'], palette = 'husl')
g.set_title("Average Prices In New York City", weight = "bold")
plt.xticks(rotation=90)
plt.show()



fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=dfqn['neighbourhood'],y=dfqn['price'], palette = 'husl')
g.set_title("Average Prices in Queens", weight = "bold")
plt.xticks(rotation=90)
plt.show()


fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=dfbx['neighbourhood'],y=dfbx['price'], palette = 'husl')
g.set_title("Average Prices in Bronx", weight = "bold")
plt.xticks(rotation=90)
plt.show()


fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=dfmn['neighbourhood'],y=dfmn['price'], palette = 'husl')
g.set_title("Average Prices in Manhattan", weight = "bold")
plt.xticks(rotation=90)
plt.show()


fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=dfbk['neighbourhood'],y=dfbk['price'], palette = 'husl')
g.set_title("Average Prices in Brooklyn", weight = "bold")
plt.xticks(rotation=90)
plt.show()

fig,axs=plt.subplots(figsize=(25,5))
g=sns.barplot(x=dfsi['neighbourhood'],y=dfsi['price'], palette = 'husl')
g.set_title("Average Prices in Staten Island", weight = "bold")
plt.xticks(rotation=90)
plt.show()

df[df.duplicated('neighbourhood')]
# as we can see some the neighbourhood does not have more than 1 listing since the row count has dropped 

# bar chart of relation among Room type, price and neighbourhood group 
plt.figure(figsize=(14, 6))
sns.barplot(df.neighbourhood_group, df.price, hue=df.room_type , ci= None)


df['name'].nunique()

# Its a scattered plot of appartments of AirBNB on the map of NYC 
plt.figure(figsize=(14, 6))
sns.scatterplot(df.longitude, df.latitude, hue=df.neighbourhood_group , ci= None)

# Its a heatmap of the NewYork City containing the AirBNB appartments
m=fm.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'red',0.6:'yellow',1.0:'black'}).add_to(m)
display(m)

# checking the hightest number of room in the neighbourhood
# observation: williumsburg has hightest number of room 
# Here a pie chartand bar chart was done aginst the neighbourhood and the number of room each of them contains
plt.style.use('fivethirtyeight')
fig,ax=plt.subplots(1,2,figsize=(35,20))
clr = ('black', 'blue', 'yellow', 'green', 'grey','orange','pink','red','cyan','purple')
df.neighbourhood.value_counts().sort_values(ascending=False)[:20].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title("Top 20 neighbourhood by the number of rooms",size=20)
ax[0].set_xlabel('Number of rooms',size=18)
ax[0].set_ylabel('Neighbourhood',size=18)
count=df['neighbourhood'].value_counts()
groups=list(df['neighbourhood'].value_counts().index)[:20]
counts=list(count[:20])
counts.append(count.agg(sum)-count[:20].agg('sum'))
groups.append('Other')
type_dict=pd.DataFrame({"group":groups,"counts":counts})
clr1=('purple', 'cyan', 'red', 'pink', 'orange','grey','green','yellow','blue','black')
qx = type_dict.plot(kind='pie', y='counts', labels=groups,colors=clr1,autopct='%1.1f%%', pctdistance=0.9, radius=1.2,ax=ax[1])
plt.legend(loc=0, bbox_to_anchor=(1.15,0.4)) 
plt.subplots_adjust(wspace =0.5, hspace =0)
plt.ioff()
plt.ylabel('')
pass

# From the graphs and the average values it is shown that appartment price over 1000 is outlier.
# Its a scatter plot that shows the price and location of those appartments 
dflow=df[df['price']<=1000]

plt.figure(figsize=(14, 6))
sns.scatterplot(dflow.longitude, dflow.latitude, hue=dflow.price , ci= None)

#Importng the necessary packages for mapping
import holoviews as hv
import geoviews as gv
import datashader as ds
from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr
from datashader.colors import colormap_select, Greys9
from holoviews.streams import RangeXY
from holoviews.operation.datashader import datashade, dynspread, rasterize
from bokeh.io import push_notebook, show, output_notebook
output_notebook()
hv.extension('bokeh')

from datashader.utils import lnglat_to_meters as webm
x, y = webm(df.longitude, df.latitude)
df['x'] = pd.Series(x)
df['y'] = pd.Series(y)

NewYork = df
agg_name = "price"
NewYork[agg_name].describe().to_frame()

get_ipython().run_line_magic('opts', "Overlay [width=800 height=600 toolbar='above' xaxis=None yaxis=None]")
get_ipython().run_line_magic('opts', "QuadMesh [tools=['hover'] colorbar=True] (alpha=0 hover_alpha=0.2)")
T = 0.05
PX = 1

def plot_map(data, label, agg_data, agg_name, cmap):
    url="http://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"
    geomap = gv.WMTS(url)
    points = hv.Points(gv.Dataset(data, kdims=['x', 'y'], vdims=[agg_name]))
    agg = datashade(points, element_type=gv.Image, aggregator=agg_data, cmap=cmap)
    zip_codes = dynspread(agg, threshold=T, max_px=PX)
    hover = hv.util.Dynamic(rasterize(points, aggregator=agg_data, width=100, height=50, streams=[RangeXY]), operation=hv.QuadMesh)
    hover = hover.options(cmap=cmap)
    img = geomap * zip_codes * hover
    img = img.relabel(label)
    return img


plot_map(NewYork,'Hotel_Prices_in_NewYork', ds.min(agg_name), agg_name, cmap=rainbow)

# Showing the NYC neighberhood in NewYork city map with each appartment's location and price
import plotly.express as px
fig = px.scatter_mapbox(df, lat = "latitude", lon = "longitude", color = "neighbourhood", size = "price", size_max = 30, opacity = .70, zoom = 12)
fig.layout.mapbox.style = 'carto-positron'
fig.update_layout(modebar_activecolor='red')
fig.update_layout(font_color='cyan')
fig.update_layout(paper_bgcolor='grey')
fig.update_traces(marker_color='red', selector=dict(type='scattergl'))
fig.update_layout(title_text = 'NYC Neighbourhood', height = 750)
fig.show()

# dropping two extra collumns that was created during mapping
df=df.drop(['x','y'], axis=1)

#Assigning numbers to "neighbourhood_group", "neighbourhood" and "room_type" for correlation plot
df['neighbourhood_group_number']=df.groupby('neighbourhood_group').ngroup()
df['neighbourhood_number']=df.groupby('neighbourhood').ngroup()
df['room_type_number']=df.groupby('room_type').ngroup()


df['neighbourhood_group_number'] = df['neighbourhood_group_number'].replace('0', '5')

import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
dfcor = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# The correlation plot among variables
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
ax=plt.axes()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlations between variables')
#corr.style.background_gradient(cmap='coolwarm')

# dropping three extra collumns that was created during correlation
df=df.drop(['neighbourhood_group_number','neighbourhood_number','room_type_number'], axis=1)

dfml=df

target_columns = ['neighbourhood_group','room_type','price','minimum_nights','calculated_host_listings_count','availability_365']
dfml = df[target_columns]

#Below we encode values of the forst column since they are strings and cannot be converted to float for linear regression
dfml['room_type'] = dfml['room_type'].factorize()[0]
dfml['neighbourhood_group'] = dfml['neighbourhood_group'].factorize()[0]
dfml.head()


# Importing packages for  Lasso, Linear and forest Regression

import seaborn as sns
import scipy.stats as st
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


Y = dfml['price']
X = dfml.drop(['price'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1, random_state = 42)


def regression_models(X_train, Y_train, X_test, Y_test):
#Linear Regression
    lr = LinearRegression()
    lr.fit(X_train,Y_train)
    y_pred = (lr.predict(X_test))
    print("LINEAR REGRESSION\nR-squared train score: ", lr.score(X_train, Y_train))
    print("R-squared test score: ", lr.score(X_test, Y_test))
#Lasso Regression
    las = Lasso(alpha = 0.0001)
    las.fit(X_train, Y_train)
    print("\nLASSO REGRESSION\nR-squared train score: ", las.score(X_train, Y_train))
    print("R-squared test score: ", las.score(X_test, Y_test))
#Decision Tree
    dec_tree = dtr(min_samples_leaf = 25)
    dec_tree.fit(X_train, Y_train)
    print("\nDECISION TREE\nR-squared train score: ", dec_tree.score(X_train, Y_train))
    print("R-squared test score: ", dec_tree.score(X_test, Y_test))
#Random Forest Regressor
    rfr = rfc()
    rfr.fit(X_train, Y_train)
    print("\nRANDOM FOREST REGRESSOR\nR-squared train score: ", rfr.score(X_train, Y_train))
    print("R-squared test score: ", rfr.score(X_test, Y_test))


regression_models(X_train, Y_train, X_test, Y_test)





