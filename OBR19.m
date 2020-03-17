% 19th m file ( Hog Feature Extraction )
%%
clc
clear
close all

%%
% Select the image from a file
[filename, pathname] = ...
     uigetfile({'*.jpg';'*.jpeg';'*.png';'*.*'},'Select Image File');
 image_path=strcat(pathname,filename);
 
%%
tic
image_original=imread(image_path); % import original image
gray_image=rgb2gray(image_original); %Convert rgb image into gray
gray_thresh=graythresh(gray_image); %find gray pixcel in original image

%%
%%figure(01)
%%set(gcf,'units','normalized','outerposition',[0 0 1 1]);% Enlarge figure to full screen
%%subplot(3,3,1)
%%imshow(image_original);
%%title('Original Image');
%%
SE=strel('disk',1); % Morphological structuring element

dl=imdilate(gray_image,SE); % Dilate function 
er=imerode(gray_image,SE); % Erode function
edit_01=imsubtract(dl,er); % Subtract er from dl

%%subplot(3,3,2)
%%imshow(edit_01);
%%title('dilated-erode image');

gray_thresh2=graythresh(edit_01); % Generate gray threshold value

%%
% Binary image
edit_02=im2bw(edit_01,gray_thresh2); % Convert edit_01 into binary image
%%subplot(3,3,3)
%%imshow(edit_02);
%%title('Binary image');


%%
% Image enhancement 
edit_03=imfill(edit_02,'holes'); % Fill the holes
%%subplot(3,3,4)
%%imshow(edit_03);
%%title('Holes filled image');

%%
edit_04=bwareaopen(edit_03,10); % remove noise
%%subplot(3,3,5)
%%imshow(edit_04);
%%title('filtered image');

%%
theta=(0:179)';
%determine the lines on the picture using radon transform
[R,xp]=radon(edge(gray_image),theta);
%finding the angle of the most visible line on the pic
[r,c]=find(R==max(R(:)));
thetap=theta(c(1));
angle=90-thetap;
rotimage=imrotate(edit_04,angle,'bilinear');

%%subplot(3,3,6)
%%imshow(rotimage);
%%title('Rotated image');
%%
[y,x]=size(rotimage);

edit_05=imresize(rotimage,[480 NaN]); % Resize the image
%%subplot(3,3,7)
%%imshow(edit_05)
%%title('resized image');

%%
edit_05=padarray(edit_05,[1 1]); % Add padding 

%%
% In this section I take the sum of the each bits in Rows
[y1, x1]=size(edit_05);

% edit_06=edit_05(1:250,1:x1);
S_row=sum(edit_05,2); % sum of rows
S_row=rot90(S_row);

L_row=length(S_row);  % Length of the S_row array


% YL=1:L_row;    
% figure(03)
% %set(gcf,'units','normalized','outerposition',[0 0 1 1]);% Enlarge figure to full screen
% %set(gca,'FontSize',20);
% set(0,'DefaultAxesFontName', 'Times New Roman')
% set(0,'DefaultAxesFontSize', 26)
% 
% stairs(S_row,YL,'LineWidth',1,'Color','k');

% title('Grapah of height of the image vs sum of the pixels along the rows of pixels', 'FontSize',18)
% ylabel('Height of the image','FontSize',26)
% xlabel('Pixels Sum','FontSize',26)

ppr=find(S_row);  % Find position of non zero elements in S_row array. Therefore ppr array contains the positions of non zero elements in S_row array 
diff_ppr=diff(ppr);  % Calculate differences between adjacent elements in ppr array
max_ppr=max(diff_ppr); % find the maximum value from diff_ppr array
filt_diff_ppr=diff_ppr(diff_ppr~=1); % Extract the values greater than 1 from diff_ppr array
min_filt_diff_ppr=min(filt_diff_ppr); % find the minimum value of filt_diff_ppr array
mean_ppr=round(mean(filt_diff_ppr)); % find the mean of filt_diff_ppr array
ppr2=find(diff_ppr>=mean_ppr);   % Find position of elements which are greater than mean in diff_ppr array
ppr3=find(diff_ppr>=min_filt_diff_ppr); % Find position of elements which are greater than min in diff_ppr array

% Abobe I have created ppr array wich contains positions of non zero non
% zero elements in S_row array. Simply means it shows the positions of white
% pixels.
% But I want to sepetate a Braille Cell without disturbing white pixels.
% When I go along the S_row array I need to find the position before the
% white pixel occure(starting point) and position after the white pixel
% occur(end point). Both these positions are black pixels. 

%%
% In this section I take the sum of the each bits in columns 
% edit_06=edit_05(1:y1,1:150);
S_column=sum(edit_05); % sum of value of pixels along the columns

L_column=length(S_column);  % Length of the S_column array

% XL=1:(L_column);
% figure(05)
% % set(gcf,'units','normalized','outerposition',[0 0 1 1]);% Enlarge figure to full screen 
% set(0,'DefaultAxesFontName', 'Times New Roman')
% set(0,'DefaultAxesFontSize', 26)
% stairs(XL,S_column,'LineWidth',1,'Color','k')

% title('Grapah of sum of the pixels along the columns of pixels vs Width of the image', 'FontSize',18)
% xlabel('Width of the image by pixels','FontSize',18)
% ylabel('Pixels Sum','FontSize',18)

ppc=find(S_column);  % Find position of non zero elements in S_column array. Therefore ppc array contains the positions of non zero elements in S_column array 
diff_ppc=diff(ppc); % Calculate differences between adjacent elements in ppc array
max_ppc=max(diff_ppc); % find the maximum value from diff_ppc array
filt_diff_ppc=diff_ppc(diff_ppc~=1); % Extract the values greater than 1 from diff_ppc array
min_filt_diff_ppc=min(filt_diff_ppc);% find the minimum value of filt_diff_ppc array
mean_ppc=round(mean(filt_diff_ppc));% find the mean of filt_diff_ppc array
ppc2=find(diff_ppc>=mean_ppc);   % Find position of elements which are greater than mean in diff_ppc array
ppc3=find(diff_ppc>=min_filt_diff_ppc); % Find position of elements which are greater than min in min_filt_diff_ppc array

%%
temp_r_ele=zeros(1,length(ppr2));    % Here I create an array of 1 by length of ppr2 array 
temp_r_ele2=zeros(1,length(ppr2));   % Here I create an array of 1 by length of ppr2 array

temp_er_ele=zeros(1,length(ppr3));   % alternate array 
temp_er_ele2=zeros(1,length(ppr3));  % alternate array

for i=1:length(ppr2)                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp_r_ele(i)=ppr2(i);            % Here I create two types of array. temp_r_ele array is same as the ppr2 array 
    temp_r_ele2(i)=ppr2(i)+1;         % but temp_c_ele2 array's elements are increment by one. both arrays have same dimensions 
end                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

for i=1:length(ppr3)                  
    temp_er_ele(i)=ppr3(i);            
    temp_er_ele2(i)=ppr3(i)+1;         
end       



r_in=ppr(1); % This is the first element of the row_sep_points array 
r_final=ppr(length(ppr)); % This is the final element of the row_sep_points array

temp_r_ele3=[temp_r_ele,temp_r_ele2];     % Combine above created temp_ele and temp_r_ele2 arrays together
temp_r_ele3=sort(temp_r_ele3);          % Sort out the temp_r_ele3 array

temp_er_ele3=[temp_er_ele,temp_er_ele2];     
temp_er_ele3=sort(temp_er_ele3);        % alternate array  


r_middle=ppr(temp_r_ele3);            % Now create array of position of middle non zero elements. This array contains the middle elemnts of ppr array
row_sep_points=[r_in,r_middle,r_final];  % This is final row separating points array


er_middle=ppr(temp_er_ele3);  
erow_sep_points=[r_in,er_middle,r_final]; % alternate array

%%
% Create a directory by the name "Rows"
fn_row = fullfile(cd,'Rows');
 if exist(fn_row,'dir')
    rmdir(fn_row,'s')
    mkdir(fn_row)
else
  mkdir(fn_row)
end;

%%
r_canceled=0; % Define the r_canceled variable as 0
rno=1; % Define the rno variable as 1
for k=1:2:length(row_sep_points)
    Row_Output_FullFile=fullfile(fn_row,sprintf('Row_0%d.jpg',rno)); % Define a name for each row of characters
    Row_No=edit_05(row_sep_points(k)-1:row_sep_points(k+1)+1,1:x1); % Crop image depending on the row_sep_points array
    [y(rno), x(rno)]=size(Row_No); % Find size of image of each row of characters 
    med_y=median(y); % Find the median value of height among all images
    %%%%%%%%%%%%%%%%%%%%%%%
    rsp=row_sep_points;   % change the row_sep_points array as the rsp
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    if y(rno)>1.2 * med_y % Check whether height of current image larger than the existing       
        r_canceled=1;  % if it is large, cancel the current loop 
            %%%%%%%%%%%%%%%%%%%%%%%
            rsp=erow_sep_points; % define the erow_sep_points as rsp
            %%%%%%%%%%%%%%%%%%%%%%%%%
        break;
    end
    
    imwrite(Row_No,Row_Output_FullFile) % write the image into Rows directory 
    rno=rno+1;     
end

if r_canceled==1;
    rno=1;
    rmdir(fn_row,'s')
    mkdir(fn_row)

    %alternate step for the above cancelled step 
for k=1:2:length(rsp)
    Row_Output_FullFile=fullfile(fn_row,sprintf('Row_0%d.jpg',rno));
    Row_No=edit_05(erow_sep_points(k)-1:erow_sep_points(k+1)+1,1:x1); 
    imwrite(Row_No,Row_Output_FullFile)
    rno=rno+1;     
end

end


%%

%for n=1:length(row_sep_points)/2
%    Row_Iutput_FullFile=fullfile(fn_row,sprintf('Row_0%d.jpg',n));
%    Input_Row=imread(Row_Iutput_FullFile);   
%    binary_Input_Row=im2bw(Input_Row);   
%    subplot(7,2,n)
%    imshow(binary_Input_Row)
%end


%%
temp_c_ele=zeros(1,length(ppc2));    % Here I create an array of 1 by length of ppc2 array 
temp_c_ele2=zeros(1,length(ppc2));   % Here I create an array of 1 by length of ppc2 array

temp_ec_ele=zeros(1,length(ppc3));
temp_ec_ele2=zeros(1,length(ppc3));

for i=1:length(ppc2)                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    temp_c_ele(i)=ppc2(i);            % Here I create two types of array. temp_c_ele array is same as the ppc2 array 
    temp_c_ele2(i)=ppc2(i)+1;         % but temp_c_ele2 array's elements are increment by one. both arrays have same dimensions  
end                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


for i=1:length(ppc3)                  
    temp_ec_ele(i)=ppc3(i);            
    temp_ec_ele2(i)=ppc3(i)+1;         
end       

c_in=ppc(1);                      % This is the first element of the column_sep_points array 
c_final=ppc(length(ppc));         % This is the final element of the column_sep_points array

temp_c_ele3=[temp_c_ele,temp_c_ele2];   % Combine  above created temp_ele and temp_c_ele2 arrays together
temp_c_ele3=sort(temp_c_ele3);          % Sort out the temp_r_ele3 array

temp_ec_ele3=[temp_ec_ele,temp_ec_ele2];     
temp_ec_ele3=sort(temp_ec_ele3);          


c_middle=ppc(temp_c_ele3);               % Now create array of position of middle non zero elements. This array contains the middle elemnts of ppc array
column_sep_points=[c_in,c_middle,c_final];  % This is final column separatig points array

ec_middle=ppc(temp_ec_ele3);  
ecolumn_sep_points=[c_in,ec_middle,c_final];

%%
% Here I create a 'BrailleCells' folder
% BraillCells folder is used to save raughly separated Braille Cells
fn_cell = fullfile(cd,'BrailleCells');
 if exist(fn_cell,'dir')
    rmdir(fn_cell,'s')
    mkdir(fn_cell)
else
  mkdir(fn_cell)
end;


%%
c_canceled=0;
for n=1:length(rsp)/2
    cno=1;
    cn=1;
    Row_Iutput_FullFile=fullfile(fn_row,sprintf('Row_0%d.jpg',n));
    Input_Row=imread(Row_Iutput_FullFile);
    binary_Input_Row=im2bw(Input_Row); 
    [y(n), x(n)]=size(binary_Input_Row);
    
    for k=1:2:length(column_sep_points)
       
        
        Cell_Output_FullFile=fullfile(fn_cell,sprintf('I%d_%d.jpg',n,cno));
        Cell_No=binary_Input_Row(1:y(n),column_sep_points(k)-1:column_sep_points(k+1)+1); 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        csp=column_sep_points;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [yc(cn), xc(cn)]=size(Cell_No);
        ratio=yc(cn)/xc(cn);
        
        if ratio<1
            c_canceled=1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            csp=ecolumn_sep_points;
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            
            break;
                 
        else
            imwrite(Cell_No,Cell_Output_FullFile) 
        end
       imwrite(Cell_No,Cell_Output_FullFile)
        cn=cn+1;
        cno=cno+1;
    end
    
    if c_canceled==1;
      
        break;
    end
        
    
end

% This loop is started if above loop is breaked
if c_canceled==1;
    rmdir(fn_cell,'s')
    mkdir(fn_cell)
    
    for n=1:length(rsp)/2
    cno=1;
    
    Row_Iutput_FullFile=fullfile(fn_row,sprintf('Row_0%d.jpg',n));
    Input_Row=imread(Row_Iutput_FullFile);
    binary_Input_Row=im2bw(Input_Row); 
    [y(n), x(n)]=size(binary_Input_Row);
    
    for k=1:2:length(csp)
       
        Cell_Output_FullFile=fullfile(fn_cell,sprintf('I%d_%d.jpg',n,cno));
        Cell_No=binary_Input_Row(1:y(n),ecolumn_sep_points(k)-1:ecolumn_sep_points(k+1)+1); 
        imwrite(Cell_No,Cell_Output_FullFile)

        cno=cno+1;
        
    end
    
    end  
end


%%
% Here I create 'ReBrailleCells' folder
fn2 = fullfile(cd,'ReBrailleCells');
 if exist(fn2,'dir')
    rmdir(fn2,'s')
    mkdir(fn2)
else
  mkdir(fn2)
end;

%% 
% Here I resize the above separated braille cells.

for ra2=1:length(rsp)/2
    for ca2=1:length(csp)/2
        Cell_Iutput_FullFile=fullfile(fn_cell,sprintf('I%d_%d.jpg',ra2,ca2));
        cell=imread(Cell_Iutput_FullFile);
        binary_cell=im2bw(cell);
        
        Output_FullFile=fullfile(fn2,sprintf('I%d_%d.jpg',ra2,ca2));
        imwrite(imresize(binary_cell,[21 16]),Output_FullFile)
    end
end
%%
fn3 = fullfile(cd,'FinalBrailleCells');
 if exist(fn3,'dir')
    rmdir(fn3,'s')
    mkdir(fn3)
else
  mkdir(fn3)
end;

imcono=1;
imrono=0;
for ra2=1:length(rsp)/2
    for ca2=1:length(csp)/2
        Cell_Iutput_FullFile=fullfile(fn2,sprintf('I%d_%d.jpg',ra2,ca2));
        cell=imread(Cell_Iutput_FullFile);
        
        Output_FullFile=fullfile(fn3,sprintf('%3d.jpg',imcono));
        imwrite(cell,Output_FullFile)
        imcono=imcono+1;
    end
end



% braille_cells_dir=fullfile(cd,'ReBrailleCells');
% braille_cells = imageDatastore(braille_cells_dir, 'LabelSource', 'none');
% numBrailleImages = numel(braille_cells.Files);
% for im=1:numBrailleImages
%     rename_image = readimage(braille_cells,im);
%     Output_FullFile=fullfile(fn3,sprintf('I%d.jpg',im));
%     imwrite(rename_image,Output_FullFile);
% end



%%
% Use of HOG feature extraction 

trainindDataSetDir = fullfile(cd,'Training data set');
testDataSetDir = fullfile(cd,'FinalBrailleCells');

trainingSet = imageDatastore(trainindDataSetDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(testDataSetDir, 'LabelSource', 'none');

%img = readimage(trainingSet, 206);

% Extract HOG features and HOG visualization
%[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
%[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
%[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

cellSize = [4 4];
%hogFeatureSize = length(hog_4x4);

numImages = numel(trainingSet.Files);
%trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    %img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);

%%
%img_test = readimage(testSet,8);

% Extract HOG features and HOG visualization
%[hog_2x2_t, vis2x2_t] = extractHOGFeatures(img_test,'CellSize',[2 2]);
%[hog_4x4_t, vis4x4_t] = extractHOGFeatures(img_test,'CellSize',[4 4]);
%[hog_8x8_t, vis8x8_t] = extractHOGFeatures(img_test,'CellSize',[8 8]);

cellSize_test = [4 4];
%hogFeatureSize_test = length(hog_4x4_t);

%%

numImages_test = numel(testSet.Files);
%testFeatures = zeros(numImages_test, hogFeatureSize_test, 'single');

for i = 1:numImages_test
    img_test = readimage(testSet, i);
    
    %img_test = rgb2gray(img_test);
    
    % Apply pre-processing steps
    img_test = imbinarize(img_test);
    
    testFeatures(i, :) = extractHOGFeatures(img_test, 'CellSize', cellSize_test);  
end

% Get labels for each image.
%testLabels = testSet.Labels;

%%
% % Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);
% 
% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(testLabels, predictedLabels);
% 
% helperDisplayConfusionMatrix(confMat)
%%

pr = rot90(predictedLabels);

% Translate to language
font=[];
lan=1;


for ca7=1:numImages_test 
    value=pr(ca7);
    letter=cellRead2(value,lan);
    font=[font letter]; 
end

%%
lan=2;

for i=1:length(font);
    if (i-1)==numel(font)
        break;
    end
    if font(i)=='#'
        f1=font(1:i-1);
        ff=font(i+1:length(font));
        
        for j=1:length(ff)           
            if ff(j)==' ' || length(ff)== 1;
                f2=ff(1:j);
                f3=ff(j+1:length(ff));
                break;
            end
        end
        
        for n=1:length(f2)                 
            let3=f2(n);
            num=number(let3,lan);
            f2(n)=num;
        end
        font=[f1,f2,f3];
        
    end
   
end

%%

for i=1:length(font);
    if font(i)=='@'
        let=font(i+1);
        cap=capLet(let);
        font(i+1)=cap;
        
    end           
end

font=font(font~='@');

toc
%%
tic
fnt=fullfile(cd,'Text Files');

d=dir(fullfile(fnt,'*.docx'));

if isempty(d)==0;
    dates=[d.datenum];
    [~,newestIndex]=max(dates);
    lastfilename=d(newestIndex).name;
    numlastfile=double(lastfilename);
    
    no1=numlastfile(1)-48;
    no2=numlastfile(2)-48;
    no3=numlastfile(3)-48;
else
    no1=0;
    no2=0;
    no3=0; 
end

if no3 == 9; %%%%%%%%%%%% no3
    if no2 == 9;
        if no1== 9; %%%%%%%%%%%%%%%% no1
                  rmdir(fnt,'s')
                  mkdir(fnt)
                  
                  no1=0;
                  no2=0;
                  no3=1;
                  
        else
            no1=no1+1;
            no2=0;
            no3=0;
            
        end   %%%%%%%%%%%%%%%%%%%% no1  
    else
        no1=no1;
        no2=no2+1;
        no3=0;
        
    end       %%%%%%%%%%%%%%%% no2
else
    no1=no1;
    no2=no2;
    no3=no3+1;
end          %%%%%%%%%%%%%%%%% no3
newno=cat(1,no1,no2,no3);
newno=num2str(newno);


newfilename=sprintf('%s.docx',newno);
nft=fullfile(fnt,newfilename);

wd = actxserver('Word.Application');
document = wd.Documents.Add;
selection = wd.Selection;
selection.Font.Size=12;

if lan == 1
    selection.Font.Name='AA Amali';
else
    selection.Font.Name='Times New Roman';

end

selection.TypeText(font);

document.SaveAs2(nft);
wd.Quit();
toc

 %fileID = fopen(nft,'w');
 %fprintf(fileID,'%s',font);
 %fclose(fileID);
 
 %winopen(nft)
           
 

            
            