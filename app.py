from flask import Flask,request,jsonify,render_template
from sklearn.ensemble import IsolationForest
import numpy as np
app=Flask(__name__)
@app.get("/")
def home(): return render_template("index.html")
@app.post("/api/detect")
def detect():
 d=request.get_json(silent=True) or {}; values=d.get("values")
 if not isinstance(values,list) or len(values)<5: return jsonify(error="values must contain at least 5 numbers"),400
 try: x=np.array([[float(v)] for v in values]); model=IsolationForest(contamination="auto",random_state=42); labels=model.fit_predict(x); scores=model.decision_function(x); return jsonify(results=[{"value":v,"anomaly":int(l==-1),"score":round(float(s),4)} for v,l,s in zip(values,labels,scores)])
 except (ValueError,TypeError): return jsonify(error="values must be numeric"),400
if __name__=="__main__": app.run(debug=True)