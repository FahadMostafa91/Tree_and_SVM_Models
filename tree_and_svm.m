%Tree and SVM Models
%Instructions are in the task pane to the left. Complete and submit each task one at a time.
%This code loads the data.
load data
whos data dataTrain dataTest

%Task 1
treeMdl = fitrtree(dataTrain,"y");
treeLoss = loss(treeMdl,dataTest)
yPred = predict(treeMdl,dataTest);

%Task 2
svmMdl = fitrsvm(dataTrain,"y")
svmLoss = loss(svmMdl,dataTest)
yPred = predict(svmMdl,dataTest)


%Task 3

svmMdl2 = fitrsvm(dataTrain,"y","KernelFunction","Polynomial")
svmLoss2 = loss(svmMdl2,dataTest)
yPred = predict(svmMdl2,dataTest)

Plot the data and predicted response.
plot(data.x,data.y,".")
hold on
plot(dataTest.x,yPred,".")
hold off
legend("All Data","Predicted Response")