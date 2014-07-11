function [corrScores, data] = evaluateWordSim(modelFile, modelFormat, lang, We, words)
%%
% Run word similarity evaluation
% 
% modelFile contains either 'We', 'words' or allW, 'params', 'words'
% modelFormat: 0 -- mat file, 
%              1 -- text file with a header line <numWords> <embDim>.
%              Subsequent lines has <word> <values>
%              2 -- text file with each line has <word> <values>
% testFiles: a cell of different word similarity data files
%
% Author: Thang Luong
%%
  dataDir = '../data';
  if strcmp(lang, 'en') == 1
    dataSets = {'wordsim353', 'MC', 'RG', 'scws_nodup', 'morphoWordSim_new'};
  elseif strcmp(lang, 'zh') == 1
    dataSets = {'zh'};
  elseif strcmp(lang, 'de') == 1
    dataSets = {'de'};
  end
  
  addpath(genpath('./sltoolbox_r101/'));
  verbose=0;
  if ~exist('We', 'var') || ~exist('words', 'var')
    [We, words] = loadWeWords(modelFile, modelFormat);
  end
  vocabMap = cell2map(words); % map words to to indices

  %% unkStr
  unkStr = findUnkStr(vocabMap);

  %% Evaluation
  [corrScores, data] = simEval(We, vocabMap, unkStr, dataDir, dataSets);
end

function [corrScores, data] = simEval(We, vocabMap, unkStr, dataDir, dataSets)
  % settings
  distType = 'corrdist';
  
  numDatasets = length(dataSets);
  data = cell(1, numDatasets);
  corrScores = zeros(1, numDatasets); 
  for kk = 1:numDatasets
    testFile = [dataDir '/' dataSets{kk} '.txt'];
    
    %% read and convert vocab data
    [datum.wordPairs, datum.humanScores] = loadWordSimData(testFile, 0, '\t'); % no header, '\s+'

    simScores = getSimScores(datum.wordPairs, We, vocabMap, distType, unkStr);
    simScores(1) = simScores(1) + 1e-10; % hack to avoid N/A value return by corr() if simScores all have the same value
    corrScores(kk) = corr(simScores, datum.humanScores, 'type', 'spearman');

    data{kk} = struct('wordPairs', {datum.wordPairs}, 'humanScores', datum.humanScores, 'simScores', simScores, 'testFile', testFile);
    fprintf(2, ' %s %2.2f', dataSets{kk}, corrScores(kk)*100);
  end
  fprintf(2, '\n'); 
end
