import joblib

# Load the model
model = joblib.load('model\prediction2.joblib')

#function to print the churn prediction using churn
# def prediction(data):
#   predictions = model.predict([data])

#   if predictions[0] == 1 :
#     return "Customer is more likely to churn"
  
#   else:
#     return "Customer will not churn"

data = [40,1,1,1.67,2,148.1,74,56.7,8.48,6.2]
predictions = model.predict([data])
print (predictions)

# print(prediction(data))