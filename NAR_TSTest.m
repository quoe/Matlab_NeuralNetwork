if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
%T = simplenar_dataset;
net = bestResultNetsSort{1}; % bestResultSort(1, 2) - лучшая
[Xs,Xi,Ai,Ts] = preparets(net,{},{},OUT);
%net = train(net,Xs,Ts,Xi,Ai);
%view(net)
[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y);
[netc,Xic,Aic] = closeloop(net,Xf,Af);
%view(netc)
%[Xs,Xi,Ai,Ts] = preparets(netc,{},{},OUT);
y2 = netc(Xs,Xic,Aic); % y2 = netc(cell(1,150),Xic,Aic); cell - прогноз на количество значений
%plot(cell2mat(OUT)); hold on; plot(cell2mat(y2));
%plot(cell2mat(y2));

%plot(cell2mat([OUT cell2mat(y2)]), 'DisplayName', ['Closeloop: ' net.userdata]); hold on; plot(cell2mat(OUT), 'DisplayName', 'Target'); 
plot(cell2mat([OUT cell2mat(y2)]), 'DisplayName', ['Closeloop: ' net.userdata]); hold on; plot(Target); 

%plot(cell2mat(OUT), 'DisplayName', net.userdata); hold on; plot(cell2mat(Y), 'DisplayName', 'Target'); 
