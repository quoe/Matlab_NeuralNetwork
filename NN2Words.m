%NN2Words
Dict = Dic; % Dic - String Array. Словарь. 1 колонна слово, 2 номер
%DictProb вероятность i-того слова. Строки соответствуют словам Dict
nums = cell2mat(y);
nums = round(nums);
nums(nums < 0) = 1;
nums(nums == 0) = 1;
nums(nums > size(Dict, 1)) = str2double(Dict(end, 2));

Str = [""]; % Нейросетью
for i = 1:size(nums, 2)
    Str(i) = Dict(nums(i), 1);
end

Str2 = [""]; % Случайно
for i = 1:size(nums, 2)
    n = round((size(Dict, 1) - 1).*rand(1) + 1); 
    Str2(i) = Dict(n, 1);
end

Str3 = [""]; % По исходным вероятностям частот
for i = 1:size(nums, 2)
    n = (DictProb(1, 1)).*rand(1) + 0; 
    for j = 1:size(DictProb, 1)
        if DictProb(j, 1) <= n
           Str3(i) = Dict(j, 1);
           break
        else
            Str3(i) = Dict(1, 1);
        end
    end
end