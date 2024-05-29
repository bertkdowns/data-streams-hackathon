from river import linear_model
from river.metrics import Accuracy
from river.evaluate import progressive_val_score
model =   linear_model.LogisticRegression()#insert Classifier/algorithm here

metric = Accuracy() #Gives the accuracy as data is "Streamed"
progressive_val_score(dataset=stream,
                        model=model,
                        metric=metric,
                        show_memory=True,
                        show_time=True,
                        print_every=50000)  #gives us the current Metric every X datapoint


cm = metric.cm  #confusion matrix based on accuracy
print("Recall per class")
for i in cm.classes:
    recall = cm.data[i][i] / cm.sum_row[i] \
        if cm.sum_row[i] != 0 else 'Ill-defined'
    print("Class {}: {:.4f}".format(i, recall)) #Gives us per class accuracy