function [unkStr, unkIndex] = findUnkStr(vocabMap)
%%
% Find a representation for unknown word in a vocabMap
% Author: Thang Luong
%%
  unkStrs = {'UUUNKKK', 'UNKNOWN', '*UNKNOWN*', 'UNK', '<UNK>', '<unk>'};
  
  %% unkStr
  unkStr = '';
  for ii=1:length(unkStrs)
    curUnkStr = unkStrs{ii};
    if isKey(vocabMap, curUnkStr)
      if strcmp(unkStr, '')
        unkStr = curUnkStr;
        break;
      end
    end
  end
  
  if strcmp(unkStr, '')
    error('No vector representing unknown words\n');
  end
  
  if isKey(vocabMap, unkStr)
    unkIndex = vocabMap(unkStr);
  end
end