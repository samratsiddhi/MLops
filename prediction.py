import joblib

# Load the model
model = joblib.load('model\prediction1.joblib')

#function to print the churn prediction using churn
def prediction(data):
  predictions = model.predict([data])

  if predictions[0] == 1 :
    return "Customer is more likely to churn"
  
  else:
    return "Customer will not churn"

data = [113,0,0,0,0,125.2,93,39,10.32,8.3]

print(prediction(data))