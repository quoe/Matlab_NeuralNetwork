% TrainAutoencoderStack
% [X,T] = wine_dataset;
if 1 == 1
    Tc = T;
    Tu = unique(Tc); % Уникальные значения
    numTu = max(Tu, 2); % Количество уникальных значений
    Tmax = max(Tu);
    Tvec = full(ind2vec(Tc, Tmax));
else
    Tvec = T;
end

IN = X; OUT = Tvec;
hiddenSize = 25;
autoenc1 = trainAutoencoder(IN,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'EncoderTransferFunction','satlin',...
    'DecoderTransferFunction','purelin',...
    'L2WeightRegularization',0.01,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.10,...
    'ShowProgressWindow',false);
features1 = encode(autoenc1,IN);
Xpred1 = predict(autoenc1, IN);
mse1 = mse(IN - Xpred1);
fprintf('MSE1 = %0.3f \n', mse1)

hiddenSize = 25;
autoenc2 = trainAutoencoder(features1,hiddenSize,...
    'L2WeightRegularization',0.001,...
    'EncoderTransferFunction','satlin',...
    'DecoderTransferFunction','purelin',...
    'L2WeightRegularization',0.01,...
    'SparsityRegularization',4,...
    'SparsityProportion',0.10,...
    'ShowProgressWindow',false,...
    'ScaleData',false);
features2 = encode(autoenc2,features1);
Xpred2 = predict(autoenc2, features1);
mse2 = mse(features1 - Xpred2);
fprintf('MSE2 = %0.3f \n', mse2)

softnet = trainSoftmaxLayer(features2,OUT,'LossFunction','crossentropy','ShowProgressWindow',false);
deepnet = stack(autoenc1,autoenc2,softnet);
deepnet = train(deepnet,IN,OUT,'showResources','no');
Y = deepnet(IN);
mseDN = mse(OUT - Y);
perf = perform(deepnet,OUT,Y);
fprintf('MSE_DN = %0.3f; Perform = %0.3f; \n', mseDN, perf)
% plotconfusion(OUT,Y);
% plotroc(OUT,Y)
[M, I] = max(Y,[],1);
figure('Name','График Target и ответа нейросети Y');
plot(OUT,'r.');
hold on
plot(Y,'go');
figure('Name','График регрессии Target и ответа нейросети Y');
plotregression(OUT, Y, 'График регрессии Target и ответа нейросети Y');
view(deepnet)
figure('Name','Рисунок с выходными цветами');
imagesc(Y) % Рисунок с выходными цветами
% colormap(gray); оттенки серого
% colorbar('Ticks',[-5,-2,1,4,7],'TickLabels',{'Cold','Cool','Neutral','Warm','Hot'})
colorbar