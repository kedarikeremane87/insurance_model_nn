import joblib
from tensorflow.keras.models import load_model
import json
from flask import  Flask, request
 
model=load_model(r"C:\Users\KW585EL\OneDrive - EY\Desktop\AI and ML training docs\Azure and Python Training\Day 3\insurance.h5")
smoker_encoder=joblib.load(r"C:\Users\KW585EL\OneDrive - EY\Desktop\AI and ML training docs\Azure and Python Training\Day 3\smoker_encoder.pkl")
region_encoder=joblib.load(r"C:\Users\KW585EL\OneDrive - EY\Desktop\AI and ML training docs\Azure and Python Training\Day 3\region_encoder.pkl")
gen_encoder=joblib.load(r"C:\Users\KW585EL\OneDrive - EY\Desktop\AI and ML training docs\Azure and Python Training\Day 3\gen_encoder.pkl")
 
 
app=Flask(__name__)
@app.route('/', methods=['POST'])
 
def myfunction():
    data=request.get_json(force=True)
    print(data)
    data=pd.DataFrame([data])
    data['sex']=gen_encoder.transform(data['sex'])
    data['smoker']=smoker_encoder.transform(data['smoker'])
    region_out=region_encoder.transform(data[['region']])
    region_out=pd.DataFrame(region_out.toarray(),columns=['region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest'])
    flatfile=pd.concat([data,region_out],axis='columns')
    flatfile=flatfile.drop('region',axis='columns')
    print(flatfile)
    output=model.predict(flatfile)
    print(output)
 
    #data=data['info']
    #print(data)
    return str(output)
app.run(port=5002)
