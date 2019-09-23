%�ֱ���1-n��loc ��des
%�ֱ���ÿ����ͼ��match
%��match�е�ƥ��������������
%ƴ��ƥ�����������ͼ�� �õ���ͼ�� ����ԭ����ͼ��
%�ظ� ֱ��ֻʣһ��
%ÿ�ε��� n-1
%ÿ���������������� �Ѻϳɵķ��ڵ�n-1
clear
clc
close all
n=4;%һ�����ٸ�ͼƬ
I1=imread('1.jpg');
I2=imread('2.jpg');
I3=imread('3.jpg');
I4=imread('4.jpg');
I=cell(1,n);%��Ԫ��洢һϵ��ͼ��
p=0.5;%ѹ������
I{1}=double(rgb2gray(imresize(I1,p)));
I{2}=double(rgb2gray(imresize(I2,p)));
I{3}=double(rgb2gray(imresize(I3,p)));
I{4}=double(rgb2gray(imresize(I4,p)));
des=cell(1,n);
loc=cell(1,n);
for i=1:n
    [des{i},loc{i}] = getFeatures(I{i});
end
    matched=cell(n,n);%n*nԪ����
    match_num=zeros(n,n);
while n>1

    for i =1:n
        for j=1:n
            if i<j
                matched{i,j}=match(des{i},des{j});
                match_num(i,j)=sum(sum(matched{i,j}~=0));%ͳ��ÿ��ƥ��ĸ���
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
    %���ƥ�����������ͼƬ���±� max_i��max_j
    I_new=stitch(I{max_j},I{max_i},loc{max_j},loc{max_i},matched{max_i,max_j});
    figure,imshow(uint8(I_new))
    I{max_i}=I{n-1};%n��i����
    loc{max_i}=loc{n-1};
    des{max_i}=des{n-1};
    I{max_j}=I{n};
    loc{max_j}=loc{n};
    des{max_j}=des{n};
    I{n-1}=I_new;
    [des{n-1},loc{n-1}]=getFeatures(I{n-1});
    n=n-1;
end

