import pandas as pd
import numpy as np
import pickle #store model in file
import json # store datatype in file(list,dict,string,etc)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Loading data

df=pd.read_csv("data\\Bengaluru_House_Data.csv")

# selecting specific features

# df=df[["size","total_sqft","bath","balcony","price"]]

# extracting only digit from size
df = df.dropna(subset=["size"])
df["size"]=df["size"].str.extract("(\d+)").astype(int)

# converts range value like (2150-2225) to singel value in total_sqft

def convert_sqft(x):
    if '-' in str(x):
        parts = x.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# +++++++++++++++ basic Cleaning +++++++++++++++++
# df= df.dropna()
df = df.drop_duplicates()
# this feature dont affect data so we fill first 
df["location"]=df["location"].fillna('other')
df["balcony"]=df["balcony"].fillna(0)

# this features are importent cant fill rendomly so we remove row with null
df=df.dropna(subset=["size","bath","total_sqft"])

# we remove unrealistic data
df = df[df["total_sqft"] / df["size"] >= 300]


# detect outliers of price
df["price_per_sqft"] = df["price"]*100000 / df["total_sqft"]

# delete outlaiers  locations with few occurence
location_stats = df.groupby("location")["location"].agg("count").sort_values(ascending=False)
location_stats_less = location_stats[location_stats <= 10]

df["location"] = df["location"].apply(lambda x: "other" if x in location_stats_less else x)
#-----------

# delete outliers price per sqrt
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)
#------------------



# convert catagorecal data location to int 
df = pd.get_dummies(df, columns=["location"], drop_first=True)
#---------------




# ----------------------------------------------
df=df.drop(["area_type","availability","society","price_per_sqft"],axis=1)



# separate features and target

X = df.drop("price", axis=1)

Y = df["price"]



# save columns name in json file
columns=X.columns.tolist()
with open("columns.json","w") as f:
    json.dump(columns,f)




# train test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# train model

model=LinearRegression()
model.fit(X_train,Y_train)

# Evaluate

Y_pred=model.predict(X_test)

print("MAE: ",mean_absolute_error(Y_test,Y_pred))
print("R2 score: ",r2_score(Y_test,Y_pred))

# Save model

with open("model.pkl","wb") as f:
    pickle.dump(model,f)
print("Model saved successfully.")

