load('data.mat')
thresold=0.5;
I1=reshape(array_img(1,:,:,:),[m/scale,n/scale,d]);
rotate=0;
for item=1:frame_number

    I1=reshape(array_img(1,:,:,:),[m/scale,n/scale,d]);
    I2=reshape(array_img(item,:,:,:),[m/scale,n/scale,d]);
    I1=uint8(I1);
    I2=uint8(I2);
    [des1,loc1] = getFeatures(I1);
    [des2,loc2] = getFeatures(I2);
    matched = match(des1,des2);
    drawMatched(matched,I1,I2,loc1,loc2);
    %������ loc1��loc2�ֱ���1��2ͼ�����������������  matched��loc1��loc2����������ͬ��ӳ��  ���� ���
    %matched(i)=j  ���j��Ϊ0  ��loc1(i)��loc2(j)ƥ�� ������ͼ2��ƥ����
    %����˼·��  ȷ��ӳ���ϵ����ͼһ��ȡ���ظ���x��������������ӳ���ϵ�ĵ���ɵ�ֱ�� ��ͼ�����Ӧ��ֱ�� Ȼ��ֱ���
    %����ֱ�ߵ�б��֮��theta ������ƽ��theta ��Ϊ��ת�Ƕ� ��ͼ����ʱ��ת��ȥ�Ϳ��ԣ���ת���ͼ�����һЩ�� [ֻ���м䲿�ֵĵ�ĺϼ�]
    %�������ŵ���ת�ǶȲ���
    %ƽ�ƣ�  ��������ƶ�Ϊ����ֱ���˶� ����ת���ÿһ��ֱ�� �����ÿһ֡�е����ĵ� �������ĵ� ��һ֡�͵�һ֡  �ó����ĵ���˶�����
    %���ó���ͼ���ƽ���˶����� ����ÿһ֡ʵ���˶�ƫ�ƺͷ��̵����ĵ�λ�� ����һ֡�����෴��ƽ�Ƽ���


    num=0;
    for i=1:length(matched)
        if matched(i)~=0
            num=num+1;
        end
    end
    theta=zeros(1,num*(num-1)/2);
    k=1;
    for i=1:length(loc1)
        if matched(i)~=0
            for j=1:length(loc1)
                if matched(j)~=0 && i~=j %��������ӳ���Ҳ�ͬ�ĵ�
                    
                    theta(k)=(loc1(i,1)-loc1(j,1))/(loc1(i,2)-loc1(j,2)) - (loc2(matched(i),1)-loc2(matched(j),1))/(loc2(matched(i),2)-loc2(matched(j),2));
                    if theta(k)>thresold || theta(k)<-thresold
                        theta(k)=0;
                    end
                    k=k+1;
                end
            end
        end
    end
    theta(isnan(theta))=0; %��theta�д���NaN����ֵ�滻��0
    rotate_theta=-mean(theta);
    rotate=atan(rotate_theta)*180/pi
    readname=strcat('C:\Users\71405\Desktop\����վ\������Ӿ�\SIFT\sift2\frames\img_',num2str(item,'%04d'),'.jpg');
    I=imread(readname);
    B=imrotate(I,rotate);
    gcfname=strcat('C:\Users\71405\Desktop\����վ\������Ӿ�\SIFT\sift2\rotate2\gcf_',num2str(item,'%04d'),'.jpg');
    imgname=strcat('C:\Users\71405\Desktop\����վ\������Ӿ�\SIFT\sift2\rotate2\img_',num2str(item,'%04d'),'.jpg');
    drawMatched(matched,I1,I2,loc1,loc2);
    imwrite(B,imgname,'jpg');
    
    saveas(gcf,gcfname);
    close
end

