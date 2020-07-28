inputSize = 89190;
outputSize = inputSize;
outputMode = 'last';
numClasses = 9;

input = imageInputLayer([1 inputSize]);
conv = convolution2dLayer([5 5],10);
relu = reluLayer;
fcl = fullyConnectedLayer(10);
sml = softmaxLayer;
col = classificationLayer;

layers = [...
    %input
    %conv
    relu
    fcl
    sml
    col];

maxEpochs = 150;
miniBatchSize = 27;
options = trainingOptions('sgdm', ...
'MaxEpochs',maxEpochs, ...
'MiniBatchSize',miniBatchSize);

net = trainNetwork(S3, layers, options);