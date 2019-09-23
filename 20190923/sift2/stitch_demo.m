 I1=imread('1.jpg');
 I2=imread('4.jpg');
 I1=double(rgb2gray(imresize(I1,0.5)));
 I2=double(rgb2gray(imresize(I2,0.5))); 
 [des1,loc1] = getFeatures(I1);
 [des2,loc2] = getFeatures(I2);
 matched = match(des1,des2);
drawMatched(matched,I1,I2,loc1,loc2);
    map=getmap(matched);
    [h1,d1]=size(I1);
    [h2,d2]=size(I2);
    result=zeros(h1+2*max(h2,d2),d1+2*max(h2,d2));
    result(max(h2,d2)+1:max(h2,d2)+h1,max(h2,d2)+1:max(h2,d2)+d1)=I1;
    pos_x=0;
    pos_y=0;%平移
    for i=1:length(map)
        pos_x=pos_x+loc1(map(i,1),1)-loc2(map(i,2),1);
        pos_y=pos_y+loc1(map(i,1),2)-loc2(map(i,2),2);
    end
    pos_x=round(pos_x/length(map));
    pos_y=round(pos_y/length(map));
    dev_x=max(h2,d2)+pos_x; %x偏移
    dev_y=max(h2,d2)+pos_y; 


    mask=getmask(pos_x,pos_y);
    %result(dev_x+1:dev_x+h2,dev_y+1:dev_y+d2)=I2;
    for i =1:h2
        for j=1:d2
            if i<=pos_x && j<=pos_y %遮罩范围内
                result(dev_x+i,dev_y+j)=mask(i,j)*I2(i,j)+(1-mask(i,j))*result(dev_x+i,dev_y+j);
            else
                result(dev_x+i,dev_y+j)=I2(i,j);
            end
        end
    end
    imshow(uint8(result));
    %调整
    y_min=1;
    [x_max,y_max]=size(result);
    for x_min=1:x_max
        if sum(result(x_min,:)~=0)
            break
        end
    end
    for y_min=1:y_max
        if sum(result(:,y_min)~=0)
            break
        end
    end

    for x_max=x_max:-1:x_min
        if sum(result(x_max,:)~=0)
            break
        end
    end
    for y_max=y_max:-1:y_min
        if sum(result(:,y_max)~=0)
            break
        end
    end
    result=result(x_min:x_max,y_min:y_max);
    %旋转
%     theta=0;
%     p=0;
%     for i=1:length(map)
%         for j=1:length(map)
%             if i~=j
%                 theta=theta+(loc1(map(i,1),1)-loc1(map(j,1),1))/(0.01+loc1(map(i,1),2)-loc1(map(j,1),2)) - ...
%                             (loc2(map(i,2),1)-loc2(map(j,2),1))/(0.01+loc2(map(i,2),2)-loc2(map(j,2),2));
%                 len1=sqrt((loc1(map(i,1),1)-loc1(map(j,1),1))^2+(loc1(map(i,1),2)-loc1(map(j,1),2))^2);
%                 len2=sqrt((loc2(map(i,2),1)-loc2(map(j,2),1))^2+(loc2(map(i,2),2)-loc2(map(j,2),2))^2);
%                 p=p+(len1/len2);
%             end
%         end
%     end
%     theta=theta/(length(map)*length(map)-length(map));
%     p=p/(length(map)*length(map)-length(map));
%     dir=-theta;
%     rotate=atan(-theta)*180/pi;
%     B=zeros(2*h2,2*d2);
%     B(h2+1:2*h2,d2+1:2*d2)=I2;
%     B=imrotate(B,rotate);
%     B(1:h1,1:d1)=I1;
%     imshow(uint8(B));
%     figure,
%     S=imresize(B,p,'nearest');
%     imshow(uint8(S));
%     loc1=uint8(loc1);
%     loc2=uint8(loc2);
%     R=[cos(dir),-sin(dir);sin(dir),cos(dir)];
%     for i=1:length(map)
%         I1(loc1(map(i,1),1),loc1(map(i,1),2))=0;
%         I2(loc2(map(i,2),1),loc2(map(i,2),2))=0;
%     end
%     subplot(121),imshow((I1))
%     subplot(122),imshow((I2))
    
    
    function map=getmap(matched)%返回 二维矩阵 表示点的对应关系
        map=zeros(sum(matched~=0),2);
        k=1;
        for i =1:length(matched)
            if matched(i)~=0
                map(k,1)=i;
                map(k,2)=matched(i);
                k=k+1;
            end
        end
    end
   

    
function mask=getmask(pos_x,pos_y) %根据偏移值返回一个m*n的mask  pox>0 poy>0时候 右下为1 左上为0
    if pos_x>1
        d_x=0:1/(pos_x-1):1;
    elseif pos_x<-1
        d_x=1:1/(pos_x+1):0;
    else
        d_x=ones(1,1);
    end
    
	if pos_y>1
        d_y=0:1/(pos_y-1):1;
	elseif pos_y<-1
        d_y=1:1/(pos_y+1):0;
	else
        d_y=ones(1,1);
	end
    
	mask=d_x'*d_y;
end
