if ~iscell(X) % Вход
    IN = con2seq(X);
else
    IN = X;
end
if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
%T = simplenar_dataset;
net = bestNet;
[Xs,Xi,Ai,Ts] = preparets(net,IN, OUT); % Подготовка данных
%net = train(net,Xs,Ts,Xi,Ai);
%view(net)
[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y);
[netc,Xic,Aic] = closeloop(net,Xf,Af);
%view(netc)
%[Xs,Xi,Ai,Ts] = preparets(netc,{},{},OUT);
y2 = netc(cell(0,300),Xic,Aic); % netc(Xs,Xic,Aic); y2 = netc(cell(0,150),Xic,Aic); cell - прогноз на количество значений
%plot(cell2mat(OUT)); hold on; plot(cell2mat(y2));
%plot(cell2mat(y2));
plot(cell2mat([OUT cell2mat(y2)])); hold on; plot(cell2mat(OUT)); 