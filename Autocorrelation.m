%Using Autocorrelation
% Нужно чтобы все нейросети были с одикавым запаздыванием!
if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
ac_y = [];
ac = []; ac_lag = [];
ac_pkshMax = []; ac_pkshMin = [];
short = []; ac_pklg = [];  ac_lclg = []; long = [];
figAcAll = figure;
figure(figAcAll);
for i = 1:9 %size(bestResultSort, 1) 
    net = ResultNet{bestResultSort(i, 2)};
    [Xs,Xi,Ai,Ts] = preparets(net,{},{},OUT);
    [Y,Xf,Af] = net(Xs,Xi,Ai);
    perf = perform(net,Ts,Y);
    [netc,Xic,Aic] = closeloop(net,Xf,Af);
    % Для ac_y нужно чтобы все нейросети были с одикавым запаздыванием
    ac_y = [ac_y ; cell2mat(netc(Xs,Xic,Aic))]; % y2 = netc(cell(1,150),Xic,Aic); cell - прогноз на количество значений
    [autocor,lags] = xcorr(cell2mat(OUT),ac_y(i, :)); % xcorr(OUT,y1,'coeff');
    ac = [ac; autocor];
    ac_lag = [ac_lag; lags];
    [pksh, lcsh] = findpeaks(autocor);
    ac_pkshMax = [ac_pkshMax; max(pksh)]; %  min(pksh) или  max(pksh), по ситуации, хз
    ac_pkshMin = [ac_pkshMin; min(pksh)]; %  min(pksh) или  max(pksh), по ситуации, хз
    short = [short; mean(diff(lcsh))];
    [pklg,lclg] = findpeaks(autocor, ...
    'MinPeakDistance',ceil(short(i, :)),'MinPeakheight',0.3);
    ac_pklg = [ac_pklg; size(pklg, 2)];
    ac_lclg = [ac_lclg; size(lclg, 2)];
    long = [long; mean(diff(lclg))];
    plot(ac_lag(i, :),ac(i, :), 'DisplayName', sprintf('%i', i)); hold on;
end
        
[pkshMax, pkshMaxI] = max(ac_pkshMax(:));
[pkshMin, pkshMinI] = min(ac_pkshMin(:));
%[maxPkshRow, maxPkshI] = ind2sub(size(A),I); % maxPkshRow - Строка с лучшей автокорреляцией
%[mM, mI] = max(pkshM);
net = ResultNet{bestResultSort(pkshMaxI, 2)}; % Нейросеть с лучшей автокорреляцией
acMax = ac(pkshMaxI, :);
acMax_lag = ac_lag(pkshMaxI, :);
acMax_pkshMax = ac_pkshMax(pkshMaxI, :);
acMax_pkshMin = ac_pkshMin(pkshMaxI, :);
shortMax = short(pkshMaxI, :);
acMax_pklg = ac_pklg(pkshMaxI, :);
acMax_lclg = ac_lclg(pkshMaxI, :);
longMax = long(pkshMaxI, :);
AcResultMax = sprintf('MAX: ac_pkshMax=%0.3f; short=%0.3f; ac_pklgSize=%0.3f; long=%0.3f;', acMax_pkshMax, shortMax, acMax_pklg, longMax);
fprintf(AcResultMax); fprintf('\n');

acMin = ac(pkshMinI, :);
acMin_lag = ac_lag(pkshMinI, :);
acMin_pkshMax = ac_pkshMax(pkshMinI, :);
acMin_pkshMin = ac_pkshMin(pkshMinI, :);
shortMin = short(pkshMinI, :);
acMin_pklg = ac_pklg(pkshMinI, :);
acMin_lclg = ac_lclg(pkshMinI, :);
longMin = long(pkshMinI, :);
AcResultMin = sprintf('MIN: ac_pkshMax=%0.3f; short=%0.3f; ac_pklgSize=%0.3f; long=%0.3f;', acMin_pkshMax, shortMin, acMin_pklg, longMin);
fprintf(AcResultMin); fprintf('\n');
        
figTS = figure;
figAutocor = figure;
figure(figTS);

plot(cell2mat([OUT ac_y(pkshMaxI, :)]), 'DisplayName','pkshMax'); hold on; plot(cell2mat([OUT ac_y(pkshMinI, :)]), 'DisplayName','pkshMin');
legend('show');
figure(figAutocor);
plot(ac_lag(pkshMaxI, :),ac(pkshMaxI, :), 'DisplayName','pkshMax'); hold on; plot(ac_lag(pkshMinI, :),ac(pkshMinI, :), 'DisplayName','pkshMin');
xlabel('Lag');
ylabel('Autocorrelation');
legend('show');
