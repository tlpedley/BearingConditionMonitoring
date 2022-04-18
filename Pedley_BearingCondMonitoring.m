% Taylor Pedley - Bearing Condition Monitoring Using Machine Learning
% Updated 4/13/22

% The purpose of this script is to extract key features from the vibration
% signature of a bearing for use in MATLAB's Classification Learner App
% Read README and DATAFORMAT before using.

% Written in MATLAB R2021b
% Toolboxes:	Signal Processing Toolbox
%				Statistics and Machine Learning Toolbox

clear
clc

%% Get data location from user
datName = string(input('Folder containing the data: '));
datPath = strcat(cd, '\', datName);
if ~isfolder(datPath)
	fprintf('Folder not found in current working directory.\n');
	return;
end

%% Pull data into directory
% Seeks out .xls files
datFiles = dir(fullfile(datPath,'**/*.xls'));


%% Cycle through the data, extracting the features
% Assumes each sample is a 2 column (time, acc) .xls stored in a folder with HB_10_1300 format
% The Excel file itself can have any name, but the folder name matters
% All data collected under the same load/rpm should be in one subfolder
currentFolder = "";

for k = 1:length(datFiles)
	currentSample = readmatrix(strcat(datFiles(k).folder,'\',datFiles(k).name));
	
	% If this is a new folder, normalize the data from the previous folder and
	% record this new index to normalize the next set of data
	if ~strcmp(currentFolder, string(datFiles(k).folder))
		% Put all samples for this load/speed into one directory
		currentFolder = string(datFiles(k).folder);
		normSet = dir(fullfile(currentFolder,'*.xls'));
		
		% Pull the entire directory into an array (bc math can't be applied to a
		% structure like this)
		normArray = [];
		for m = 1:length(normSet)
			normArray = [normArray; readmatrix(strcat(normSet(m).folder,'\',normSet(m).name))];
		end
		
		% Calculate the average and standard deviation of the whole subset of
		% data. This will be used to normalize the features
		normMean = mean(normArray(:,2),1);
		normStd = std(normArray(:,2),0,1);

		clear normArray normSet
	end

	% Get bearing type to classify data
	pathLength = strlength(datFiles(k).folder);
	BearingType(k,1) = extractBetween(datFiles(k).folder,pathLength-9,pathLength-8);

	% Extract and normalize features
	TrimMean(k,1) = (trimmean(currentSample(:,2),10,1)-normMean)/normStd;
	ProbDensity(k,1) = (sqrt(abs(rms(currentSample(:,2),1)))-normMean)/normStd;
	FirOrdMoment(k,1) = (moment(currentSample(:,2),1,1)-normMean)/normStd;
	FouOrdMoment(k,1) = (moment(currentSample(:,2),4,1)-normMean)/normStd;
	Median(k,1) = (median(currentSample(:,2),1)-normMean)/normStd;
	Kurtosis(k,1) = (kurtosis(currentSample(:,2),0,1)-normMean)/normStd;
	SigNDRatio(k,1) = sinad(currentSample(:,2),currentSample(4,1)-currentSample(3,1));
	BandPower(k,1) = (bandpower(currentSample(:,2))-normMean)/normStd;

	% Every 100 loops, print to let user know code is working
	if mod(k,100) == 0
		fprintf(strcat('\nExtracted features from\t',string(k),' samples.'));
	end
end

%% Combine and save data
% Combine features into table and randomize data order
BearingFeatures = [TrimMean ProbDensity FirOrdMoment FouOrdMoment Median Kurtosis SigNDRatio BandPower];
numRows = int32(size(BearingFeatures,1));
randInd = randperm(numRows);
BearingFeatures = BearingFeatures(randInd,:);
BearingType = BearingType(randInd,:);

% Split into 85% training and 15% testing data
Bearing_Train = BearingFeatures(0.15*numRows+1:numRows,:);
Bearing_Test = BearingFeatures(1:0.15*numRows,:);
Bearing_Train = array2table(set_E_train,'VariableNames',{'Trimmed Mean','Probability Density',...
	'1st Order Moment','4th Order Moment','Median','Kurtosis','Signal Noise Distortion Ratio','Band Power'});
Bearing_Train.Type = BearingType(0.15*numRows+1:numRows,:);
Bearing_Test = array2table(Bearing_Test,'VariableNames',{'Trimmed Mean','Probability Density',...
	'1st Order Moment','4th Order Moment','Median','Kurtosis', 'Signal Noise Distortion Ratio','Band Power'});
Bearing_Test.Type = BearingType(1:0.15*numRows,:);

% Export to .mat file
save(strcat(datName,'_Train.mat'),'Bearing_Train');
save(strcat(datName,'_Test.mat'),'Bearing_Test');

fprintf(strcat('\n\nFeature extraction complete.\t', string(k), ' samples processed.\n' ));