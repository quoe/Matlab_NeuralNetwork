% Layer-Recurrent Neural Networks
%load phoneme
%p = con2seq(y);
%t = con2seq(t);
lrn_net = layrecnet(1,10);
lrn_net.trainFcn = 'trainlm';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 1000;
lrn_net = train(lrn_net,IN,OUT);
y1 = lrn_net(X);
plot(cell2mat(y1))
perf = perform(lrn_net,IN,y1)
