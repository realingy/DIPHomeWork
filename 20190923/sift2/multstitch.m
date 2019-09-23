%分别求1-n的loc 和des
%分别求每两个图的match
%对match中的匹配点个数进行排序
%拼接匹配点最多的两个图像 得到新图像 代替原来的图像
%重复 直到只剩一张
%每次迭代 n-1
%每次提出两个放在最后 把合成的放在第n-1
clear
clc
close all
n=4;%一共多少个图片
I1=imread('1.jpg');
I2=imread('2.jpg');
I3=imread('3.jpg');
I4=imread('4.jpg');
I=cell(1,n);%用元祖存储一系列图像
p=0.5;%压缩比例
I{1}=double(rgb2gray(imresize(I1,p)));
I{2}=double(rgb2gray(imresize(I2,p)));
I{3}=double(rgb2gray(imresize(I3,p)));
I{4}=double(rgb2gray(imresize(I4,p)));
des=cell(1,n);
loc=cell(1,n);
for i=1:n
    [des{i},loc{i}] = getFeatures(I{i});
end
    matched=cell(n,n);%n*n元数组
    match_num=zeros(n,n);
while n>1

    for i =1:n
        for j=1:n
            if i<j
                matched{i,j}=match(des{i},des{j});
                match_num(i,j)=sum(sum(matched{i,j}~=0));%统计每个匹配的个数
            end
        end
    end
    max_num=0;
    max_i=0;
    max_j=0;
    for i=1:n
        for j=1:n
            if match_num(i,j)>max_num
                max_i=i;
                max_j=j;
                max_num=match_num(i,j);
            end
        end
    end
    %获得匹配点最多的两个图片的下标 max_i和max_j
    I_new=stitch(I{max_j},I{max_i},loc{max_j},loc{max_i},matched{max_i,max_j});
    figure,imshow(uint8(I_new))
    I{max_i}=I{n-1};%n和i交换
    loc{max_i}=loc{n-1};
    des{max_i}=des{n-1};
    I{max_j}=I{n};
    loc{max_j}=loc{n};
    des{max_j}=des{n};
    I{n-1}=I_new;
    [des{n-1},loc{n-1}]=getFeatures(I{n-1});
    n=n-1;
end

