% Nonlinear Autoregressive Neural Network (narnet)
% Нейросеть обучается именно на том типе данных, который был на неё подан!
% Если был подан тип cell, то и выдавать тот же результат НС будет на cell
% load phoneme
% X = con2seq(y);
% T = con2seq(t);
%T = simplenar_dataset;
% if ~iscell(X) % Вход
%     IN = con2seq(X);
% else
%     IN = X;
% end
name = 'NAR_';
T = Target(1:end-round(size(Target, 2)/3)); % берём первую треть для обучения. Остатки - для проверки проверки прогноза
if ~iscell(T) % Выход
    OUT = con2seq(T);
else
    OUT = T;
end
rng('shuffle')
%IN = con2seq(X); OUT = con2seq(T); % IN - вход, OUT - выход
%IN = X; OUT = T; % IN - вход, OUT - выход
minHN = [2 2]; % [2 2]; Минимальное количесвто скрытых нейронов. Или minHN = 2;
maxHN = [10 10]; % [5 5]; Максимальное количесвто скрытых нейронов. Или норм maxHN = 5;
%trainDelayArr = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150]; % Массив возможных задержек
trainDelayArr = [20];
maxEpochs = 1000; 
maxLoop = 50; % Максимальное количество циклов поиска лучшей нейросети
saveNetCount = 5; % Количество сохраняемых лучших отсортированных нейросетей

trainParamStr = ["trainlm" "trainbr" "trainbfg" "trainrp" "trainscg" "traincgb" "traincgf" "traincgp" "trainoss" "traingdx" "traingdm" "traingd"];
%trainParamStr = ["trainlm"];
trainParamIW = ones(1, size(trainParamStr, 2)); % Веса алгоритмов тренировки
trainParamIWadd = 0.01; % значение для управления весами алгоритмов
Result = []; % Разные параметры результатов
ResultHN = [];
ResultNet = {}; % Массив нейросетей
bestResultRSort = []; % Отсортированный список лучших регрессий и их номеров
bestResultNetsInfoSort = []; % Отсортированный список данных лучших нейросетей 
bestResultNetsSort = {}; % Отсортированный список лучших нейросетей (самое ценное)
bestIdx = 0;
if maxLoop >= 100
    autoSaveEvery = round(maxLoop*0.25/(maxLoop/100)); 
else
    autoSaveEvery = round(maxLoop*0.25); 
end

MinHNStr = '[ ';
for j = 1:size(minHN, 2) % Вывод минимальных скрытых нейронов
    MinHNStr = [MinHNStr sprintf('%i ', minHN(j))]; 
end
MinHNStr = [MinHNStr ']'];
MaxHNStr = '[ ';
for j = 1:size(maxHN, 2) % Вывод максимальных скрытых нейронов
    MaxHNStr = [MaxHNStr sprintf('%i ', maxHN(j))]; 
end
MaxHNStr = [MaxHNStr ']'];
trainDelayArrStr = '[ ';
for j = 1:size(trainDelayArr, 2)  % Вывод возможных временных задержек
    trainDelayArrStr = [trainDelayArrStr sprintf('%i ', trainDelayArr(j))]; 
end
trainDelayArrStr = [trainDelayArrStr ']'];
startDateTime = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
nnetGlobalTrainInfo = sprintf('minHN=%s; maxHN=%s; trainDelayArr=%s; maxEpochs=%i; maxLoop=%i; \n', MinHNStr, MaxHNStr, trainDelayArrStr, maxEpochs, maxLoop);
fprintf([ nnetGlobalTrainInfo 'Start: ' char(string(startDateTime)) '\n']);
for i = 1:maxLoop
    tic; % Начало таймера
    HN = [];  % Hidden neurons, количество скрытых нейронов
    ResultHNStr = '[ ';
    for j = 1:size(minHN, 2) % Вывод скрытых нейронов
        randHN = round((maxHN(j)-minHN(j)).*rand(1) + minHN(j));
        HN = [HN randHN];
        ResultHN(i, j) = HN(j); % Сохраняем скрытые нейроны в переменную
        ResultHNStr = [ResultHNStr sprintf('%i ', HN(j))]; % за одно формирум инфо-строку
    end
    ResultHNStr = [ResultHNStr ']'];
    
    trainRand = round((size(trainParamStr,2)-1).*rand(1) + 1); % Случайный алгоритм тренировки
    if rand(1) >= 0.5 && bestIdx ~= 0
        [bestTrainVal, bestTrainIdx] = max(trainParamIW(1,:)); % Выбираем лучший алгоритм (с наибольшим весом)
        trainRand = bestTrainIdx; % Выбираем лучший алгоритм по его индексу
    end
    trainDelayRand = round((size(trainDelayArr,2)-1).*rand(1) + 1); % Случайная задержка
    nnet = narnet(1:trainDelayArr(trainDelayRand), HN); % Создание нейросети RNN
    nnet.trainFcn = char(trainParamStr(trainRand)); % Тренировочный алгоритм
    %nnet.trainParam.show = 5;
    nnet.trainParam.epochs = maxEpochs;
    nnet.trainParam.showWindow = false; % Не показывать окно тренировки
    nnet.divideParam.trainRatio = 70/100; % Set up Division of Data for Training, Validation, Testing
    nnet.divideParam.valRatio = 15/100; 
    nnet.divideParam.testRatio = 15/100; 
    [Xs,Xi,Ai,Ts] = preparets(nnet,{},{},OUT); % Подготовка данных
    nnet = train(nnet,Xs,Ts,Xi,Ai); %  Тренировка нейросети
    %IN = Xs; OUT = Ts; % Мистичный новый вход и выход
    y = nnet(Xs,Xi); % Выходной ответ нейросети
    %plot(cell2mat(y))
    perf = perform(nnet,Ts,y); % Ошибка на выходе нейросети и данных
    r = mean(regression(Ts, y)); % Значение регрессии нейросети и данных
    Result = [Result; mean(HN) perf r trainRand trainDelayArr(trainDelayRand) toc i];
    
    bestIdx = i; % Для сохранения информации внуть нейросети в поле userdata
    ResultStr = sprintf('№ %i; HN = %s; Delay = %i; trainParam = %s; R = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, ResultHNStr, Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), Result(bestIdx, 3), Result(bestIdx, 2), Result(bestIdx, 6));
    nnet.userdata = ResultStr; 
    ResultNet{end+1} = nnet;
    if mod(i, autoSaveEvery) == 0
        [bestVal, bestIdx] = max(Result(:,3)); % Выбираем лучшую регрессию
        bestNet = ResultNet{bestIdx}; % Выбираем нейросеть с лучшей регрессией
        for j = 1:size(trainParamIW, 2) 
            if j ~= Result(bestIdx, 4) % Увеличиваем веса у хороших алгоритмов, уменьшаем у плохих
                trainParamIW(1, j) = trainParamIW(1, j) - trainParamIWadd;
            else
                trainParamIW(1, j) = trainParamIW(1, j) + trainParamIWadd;
            end
        end
        fprintf('iter = %i; ', i);
        dateTime = char(string(datetime('now','Format','yyyy-MM-dd HHmmss'))); % Убойная конструкция даты
        ResultHNStr = '[ ';
        for j = 1:size(minHN, 2)
            ResultHNStr = [ResultHNStr sprintf('%i ', ResultHN(bestIdx, j))];
        end
        ResultHNStr = [ResultHNStr ']'];
        ResultStr = sprintf('№ %i; HN = %s; Delay = %i; trainParam = %s; R = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, ResultHNStr, Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), Result(bestIdx, 3), Result(bestIdx, 2), Result(bestIdx, 6));
        fileName = ['bestNet_' name dateTime '_' sprintf('HN=%s R=%0.3f', ResultHNStr, bestVal) '.mat'];
        fprintf(ResultStr); fprintf('\n');
        bestNet.userdata = ResultStr; 
        % Массив лучших нейросетей
        bestResultRSort = []; % Отсортированный список лучших регрессий и их номеров
        bestResultNetsInfoSort = []; % Отсортированный список данных лучших нейросетей 
        bestResultNetsSort = {}; % Отсортированный список лучших нейросетей (самое ценное)
        if i >= saveNetCount
            [bestResultRSort(:, 1),bestResultRSort(:, 2)] = sort(Result(:,3), 'descend'); % Несколько отсортированных нейросетей с лучшей только регрессией и их индексами
            bestResultNetsInfoSort = Result(bestResultRSort(1:saveNetCount, 2), :); % Полные данные N отсортированных нейросетей
            bestResultNetsSort = ResultNet(bestResultRSort(1:saveNetCount, 2)'); % Отсортированный список лучших нейросетей (самое ценное)
            save(fileName, 'bestNet', 'ResultStr', 'bestResultNetsInfoSort', 'bestResultNetsSort', 'nnetGlobalTrainInfo');  % Сохраняем лучшую нейросеть
        end
    end
end
[bestVal, bestIdx] = max(Result(:,3)); % Выбираем лучшую регрессию
bestNet = ResultNet{bestIdx}; % Выбираем нейросеть с лучшей регрессией
dateTime = char(string(datetime('now','Format','yyyy-MM-dd HHmmss'))); % Убойная конструкция даты
ResultHNStr = '[ ';
for j = 1:size(minHN, 2)
    ResultHNStr = [ResultHNStr sprintf('%i ', ResultHN(bestIdx, j))];
end
ResultHNStr = [ResultHNStr ']'];
ResultStr = sprintf('№ %i; HN = %s; Delay = %i; trainParam = %s; R = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, ResultHNStr, Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), bestVal, Result(bestIdx, 2), Result(bestIdx, 6));
fileName = ['bestNet_' name dateTime '_' sprintf('HN=%s R=%0.3f', ResultHNStr, bestVal) '.mat'];
fprintf(ResultStr); fprintf('\n');
bestNet.userdata = ResultStr; 
% Массив лучших нейросетей
bestResultRSort = []; % Отсортированный список лучших регрессий и их номеров
bestResultNetsInfoSort = []; % Отсортированный список данных лучших нейросетей 
bestResultNetsSort = {}; % Отсортированный список лучших нейросетей (самое ценное)
[bestResultRSort(:, 1),bestResultRSort(:, 2)] = sort(Result(:,3), 'descend'); % Несколько отсортированных нейросетей с лучшей только регрессией и их индексами
bestResultNetsInfoSort = Result(bestResultRSort(1:saveNetCount, 2), :); % Полные данные N отсортированных нейросетей
bestResultNetsSort = ResultNet(bestResultRSort(1:saveNetCount, 2)'); % Отсортированный список лучших нейросетей (самое ценное)
save(fileName, 'bestNet', 'ResultStr', 'bestResultNetsInfoSort', 'bestResultNetsSort', 'nnetGlobalTrainInfo');  % Сохраняем лучшую нейросеть

[Xs,Xi,Ai,Ts] = preparets(bestNet,{},{},OUT); % Подготовка данных
y = bestNet(Xs,Xi); % Выходной ответ нейросети
%ResultStr = sprintf('№ %i; HN = %0.3f; Delay = %i; trainParam = %s; \nR = %0.3f; Perf = %0.3f; Time = %0.3f;', bestIdx, Result(bestIdx, 1), Result(bestIdx, 5), trainParamStr(Result(bestIdx, 4)), bestVal, Result(bestIdx, 2), Result(bestIdx, 6));
if size(Ts, 1) == 1 % Если выход 1, то выводим график, иначе будет белеберда
    figOutput = figure('Name','Target and NAR output');
    figure(figOutput); % figOutput figRegress
    if iscell(Ts) % Выход
        plot(cell2mat(Ts), 'LineWidth',2, 'DisplayName', 'Target'); hold on; 
        plot(cell2mat(y), 'DisplayName', bestNet.userdata)
    else
        plot(Ts, 'LineWidth',2, 'DisplayName', 'Target'); hold on; 
        plot(y, 'DisplayName', bestNet.userdata)
    end
    title(ResultStr)
end
figRegress = figure('Name','Target and NAR regression');
figure(figRegress); % figOutput figRegress
plotregression(Ts, y, ResultStr)
view(bestNet)
% net = closeloop(net); net = openloop(net);

figHistRegress = figure('Name','Histogram of regression');
figure(figHistRegress); % figOutput figRegress
h_r = histogram(Result(:, 3), 30); % Гистограмма регрессий
endDateTime = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
deltaDateTime = endDateTime - startDateTime;
fprintf(['End: ' char(string(endDateTime)) '; Total time: ' char(string(deltaDateTime)) '\n']);