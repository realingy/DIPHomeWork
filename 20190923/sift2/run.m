clear
clc

%��ȡ��Ƶ����֡������ͼƬ�󱣴���Ŀ¼���ļ�����
video=VideoReader('Rocks.mp4');
frame_number=video.NumberOfFrames;
scale=4;%���ų߶�
[m n d]=size(read(video,1));
array_img=zeros(frame_number,m/scale,n/scale,d);
for i=1:frame_number
    img=read(video,i);
    imgname=strcat('C:\Users\71405\Desktop\����վ\������Ӿ�\SIFT\sift2\frames\img_',num2str(i,'%04d'),'.jpg');
    imwrite(img,imgname,'jpg');
    array_img(i,:,:,:)=imresize(img,[m/scale,n/scale]);
end

%ʾ���������е�ͼƬ��ʾ��array_img(i,:,:,:)��Ϊ��ά����1*160*360*3�� ��reshapeת��160*360*3�Ϳ����� ��
for i=1:frame_number
    pause(0.01) 
    a=reshape(array_img(i,:,:,:),[m/scale,n/scale,d]);
    a=uint8(a);
    imshow(a)
end
save('data.mat')

