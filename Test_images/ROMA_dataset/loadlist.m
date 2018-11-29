% this function reads a list of filenames in a given file
% 
% name is the filename of a list of filenames with a header (can be img.mov, imgnormal.mov, imgadvlight.mov, imghighcurv.mov)
%
% names is the list of filenames
% nelem is the number of filenames
%
function [names,nelem]=loadlist(name)

data=textread(name,'%s','delimiter','\n','whitespace','');

% first line of the file contains a header 
% the number of elements
nelem = str2num(char(data(1)));
% all the names 
names={};
fullnames=data(2:nelem+1);
for i=1:nelem;
	tmp=fullnames(i);
	names(i)=regexprep(tmp, '.jpg', '');
end
