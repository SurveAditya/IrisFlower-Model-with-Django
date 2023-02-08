from django.shortcuts import render
from joblib import load
model = load("./savedModels/model.joblib")

def flowerPrediction(request):
    if request.method == 'POST':
        sepal_length = request.POST.get('sepal_length')
        sepal_width = request.POST.get('sepal_width')
        petal_length = request.POST.get('petal_length')
        petal_width = request.POST.get('petal_width')
        y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        if y_pred[0] == 0:
            y_pred = "Iris Setosa"
        elif y_pred[0] == 1:
            y_pred = "Iris Versicolor"
        else:
            y_pred = "Versicolor Iris"
        return render(request, "predict.html", {"result": y_pred})
    return render(request, "predict.html")
