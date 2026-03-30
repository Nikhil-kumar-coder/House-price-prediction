import pickle
import json
import pandas as pd

with open("model.pkl","rb") as f:
    model=pickle.load(f)

with open("columns.json","r") as f:
    columns=json.load(f)

df=pd.DataFrame([0]*len(columns)).T

df.columns=columns



size=int(input("Enter size(bhk): "))
total_sqft=int(input("Enter total_sqft: "))
bath=int(input("Enter total bath room: "))
balcony=int(input("Enter total balcony: "))


location=input("Enter location(Devarachikkanahalli): ")
location_name="location_"+location


df.at[0,"size"]=size
df.at[0,"total_sqft"]=total_sqft
df.at[0,"bath"]=bath
df.at[0,"balcony"]=balcony

if location_name in columns:
    
    df.at[0,location_name]=1
else:
    df.at[0,"location_other"]=1

price=model.predict(df) 
print("Pridicted price:",price[0],"lakh")   