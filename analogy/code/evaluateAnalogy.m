function [semantic_acc, syntactic_acc, total_acc] = evaluateAnalogy(modelFile, modelFormat, dataDir, isNormalized, isColVector, We, words)
%%
% Evaluate word vectors using Google analogy data.
% 
% modelFormat: 0 -- mat file, 
%              1 -- text file with a header line <numWords> <embDim>.
%              Subsequent lines has <word> <values>
%              2 -- text file with each line has <word> <values>
% isNormalize: 1 if word vectors are normalized, 
%              0 if word vectors are not normalized, we'll normalize.
% isColVector: 0 when W(i, :) corrresonds to words{i},
%              1 when W(:, i) corresponds to words{i}. 

% Authors: 
%   Jeffrey Pennington <jpennin@stanford.edu>
%   Thang Luong <lmthang@stanford.edu>: add minor enhancements
%%

  verbose = 1;
  if ~exist('We', 'var') || ~exist('words', 'var')
    [We, words] = loadWeWords(modelFile, modelFormat);
  end

  if isNormalized==0 % not normalized yet
    if isColVector
      We = bsxfun(@rdivide, We, sqrt(sum(We.*We,1)));
    else
      We = bsxfun(@rdivide, We,sqrt(sum(We.*We,2)));
    end
  end
  
  filenames = {'capital-common-countries' 'capital-world' 'currency' 'city-in-state' 'family' 'gram1-adjective-to-adverb' ...
      'gram2-opposite' 'gram3-comparative' 'gram4-superlative' 'gram5-present-participle' 'gram6-nationality-adjective' ...
      'gram7-past-tense' 'gram8-plural' 'gram9-plural-verbs'};

  
  correct_sem = 0;
  correct_syn = 0;
  correct_tot = 0;
  count_syn = 0;
  count_sem = 0;
  count_tot = 0;
  full_count = 0;

  wordMap = containers.Map(words, 1:length(words));

  % find unkkey
  unkStrs = {'UUUNKKK', 'UNKNOWN', '*UNKNOWN*', 'UNK', '<UNK>', '<unk>'};
  unkkey=1;
  for ii=1:length(unkStrs)
    if isKey(wordMap, unkStrs{ii})
      unkkey = wordMap(unkStrs{ii});
      break;
    end
  end

  for j=1:length(filenames);
    fid=fopen([dataDir '/' filenames{j} '.txt']);
    temp=textscan(fid,'%s%s%s%s');
    fclose(fid);
    
    % look up word indices
    [ind1] = wordLookup(wordMap, temp{1}, unkkey); %ind1 = cellfun(@WordLookup,temp{1});
    [ind2] = wordLookup(wordMap, temp{2}, unkkey); %ind2 = cellfun(@WordLookup,temp{2});
    [ind3] = wordLookup(wordMap, temp{3}, unkkey); %ind3 = cellfun(@WordLookup,temp{3});
    answers = temp{4}; %[ind4] = wordLookup(wordMap, temp{4}, unkkey); %ind4 = cellfun(@WordLookup,temp{4});
    
    
    full_count = full_count + length(ind1);
    ind = (ind1 ~= unkkey) & (ind2 ~= unkkey) & (ind3 ~= unkkey); % & (ind4 ~= unkkey);
    ind1 = ind1(ind);
    ind2 = ind2(ind);
    ind3 = ind3(ind);
    answers = answers(ind); %ind4 = ind4(ind);
    %disp([filenames{j} ':']);
    val = zeros(size(ind1));
    for ii=1:length(ind1)

        if isColVector % if We is organized by columns
          dist = full((We(:, ind2(ii))' - We(:, ind1(ii))' +  We(:, ind3(ii))') * We);
        else
          dist = full(We * (We(ind2(ii),:)' - We(ind1(ii),:)' +  We(ind3(ii),:)'));
        end
        
        dist(ind1(ii)) = -Inf;
        dist(ind2(ii)) = -Inf;
        dist(ind3(ii)) = -Inf;
        [~, mx] = max(dist);
        if strcmp(words{mx}, answers{ii}) %mx == ind4(i)
            val(ii) = 1;
        end
    end
    count_tot = count_tot + length(ind1);
    correct_tot = correct_tot + sum(val);
    %disp(['ACCURACY TOP1: ' num2str(mean(val)*100,'%-2.2f') '%  (' num2str(sum(val)) '/' num2str(length(val)) ')']);
    if j < 6
        count_sem = count_sem + length(ind1);
        correct_sem = correct_sem + sum(val);
    else
        count_syn = count_syn + length(ind1);
        correct_syn = correct_syn + sum(val);
    end

    %disp(['Total accuracy: ' num2str(100*correct_tot/count_tot,'%-2.2f') '%   Semantic accuracy: ' num2str(100*correct_sem/count_sem,'%-2.2f') '%    Syntactic accuracy: ' num2str(100*correct_syn/count_syn,'%-2.2f') '%']);

  end

%   disp('________________________________________________________________________________');
%   disp(['Questions seen/total: ' num2str(100*count_tot/full_count,'%-2.2f') '%  (' num2str(count_tot) '/' num2str(full_count) ')']);
%   disp(['Total Accuracy: ' num2str(100*correct_tot/count_tot,'%-2.2f') '%   (' num2str(correct_tot) '/' num2str(count_tot) ')']);
  
  if count_tot==0
    total_acc = 0;
  else
    total_acc = 100*correct_tot/count_tot;
  end
  
  if count_sem==0
    semantic_acc = 0;
  else
    semantic_acc = 100*correct_sem/count_sem;
  end
  
  if count_syn==0
    syntactic_acc = 0;
  else
    syntactic_acc = 100*correct_syn/count_syn;
  end
  
  if verbose == 1
    fprintf(2, ' sem %2.2f syn %2.2f all %2.2f\n', semantic_acc, syntactic_acc, total_acc);
    fprintf(2, 'eval analogy %2.2f %2.2f %2.2f\n', semantic_acc, syntactic_acc, total_acc);
  end
end

function [indices] = wordLookup(wordMap, words, unkkey)
  indices = unkkey*ones(length(words), 1);
  flags = isKey(wordMap, words);
  indices(flags) = cell2mat(values(wordMap, words(flags)));
end
