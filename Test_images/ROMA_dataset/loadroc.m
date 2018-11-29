% this function reads all the file generated by the extraction algorithm
% 
% sequence is the images subdirectory (BDXD54, BDXN01, IRC041500, IRC04510, LRAlargeur13032003, LRAlargeur14062002, RAlargeur26032003, RD116, RouenN8IRC051900, RouenN8IRC052310)
% situation is the name of the subdirectory where files generated by the extraction algorithm are
% imagelist is the name of the file with the list of the images to be used (img.mov, imgnormal.mov, imgadvlight.mov, imghighcurv.mov)
% algoname is the name of the used extraction algorithm
% TP is the array of True Positive, the line index is the image number, the row index is threshold value of the extraction algorithm 
% FP is the array of False Positive, the line index is the image number, the row index is threshold value of the extraction algorithm 
% TN is the array of True Negative, the line index is the image number, the row index is threshold value of the extraction algorithm 
% FN is the array of False Negative, the line index is the image number, the row index is threshold value of the extraction algorithm 
% values is the vector of threshold values used by the extraction algorithm
%
% filenames generated by the extraction algorithm must be with the format : /sequence/situation/algoname_imagename.txt
%
function [TP,FP,TN,FN,values]=loadroc(sequence,situation,imagelist,algoname,TP,FP,TN,FN)

values=[];
lname=sprintf('%s/%s.mov',sequence, imagelist);
[names,num]=loadlist(lname);
for i = 1:num;
	filename = sprintf('%s/%s/%s_%s.txt',sequence,situation,algoname,char(names(i)));
	data = load(filename);   
    	TP = [ TP; data(:,2)' ];
    	FP = [ FP; data(:,3)'];
    	TN = [ TN; data(:,4)' ];
    	FN = [ FN; data(:,5)'];
	values= data(:,1)';
end
