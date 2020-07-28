%NN2WordsLoop
if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
Dict = Dic; % Dic - String Array. Словарь. 1 колонна слово, 2 номер
y_cl = []; y_fit = [];
WordsArray = [];
removeDelay = 1; % 1 - убрать запаздание (будет прогноз вперёд), 0 - не убирать
for i = 1:3%size(bestResultNetsSort, 2)
    if removeDelay == 1 % Убрать запаздывание
        net = removedelay(bestResultNetsSort{i}); % removedelay(net); adddelay(net)
    else
        net = bestResultNetsSort{i}; 
    end
    [Xs,Xi,Ai,Ts] = preparets(net,{},{},OUT);
    [Y,Xf,Af] = net(Xs,Xi,Ai);
    %perf = perform(net,Ts,Y);
    %[netc,Xic,Aic] = closeloop(net,Xf,Af);
    %y2 = netc(Xs,Xic,Aic); % y2 = netc(cell(1,150),Xic,Aic); cell - прогноз на количество значений
    %y_cl = [y_cl; y2];
    y_fit = [y_fit; Y];
    
    y_N2W = cell2mat(y_fit(i,:)); % Nums to words
    yN2W1 = y_N2W(1,1:end-1); % Смещение, чтобы найти разницу
    yN2W2 = y_N2W(1,2:end); % Смещение, чтобы найти разницу
    yN2W3 = yN2W2 - yN2W1; % Найдены разницы
    nums = [y_N2W(1:1) yN2W3]; % Найдены номера реальных слов
    nums = round(nums);
    nums(nums < 0) = 1;
    nums(nums == 0) = 1;
    nums(nums > size(Dict, 1)) = str2double(Dict(end, 2));
    Words = [""]; % Нейросетью
    for i = 1:size(nums, 2)
        Words(i) = Dict(nums(i), 1);
    end
    WordsArray = [WordsArray; Words];
end 
y_mean = mean(cell2mat(y_fit),1);
y_N2W = y_mean; % Nums to words
yN2W1 = y_N2W(1,1:end-1);
yN2W2 = y_N2W(1,2:end);
yN2W3 = yN2W2 - yN2W1;
nums = [y_N2W(1:1) yN2W3];
nums = round(nums);
nums(nums < 0) = 1;
nums(nums == 0) = 1;
nums(nums > size(Dict, 1)) = str2double(Dict(end, 2));
Words = [""]; % Нейросетью
for i = 1:size(nums, 2)
    Words(i) = Dict(nums(i), 1);
end
WordsArray = [WordsArray; Words];