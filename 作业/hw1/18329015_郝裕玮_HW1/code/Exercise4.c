#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main()
{

    srand((int)time(0));
    int i,j,k,cnt=0;
    int temp=20000;//temp代表20000次重复试验 
    int a[8][8];//地图数组 

	//蚂蚁的起点横纵坐标(1,1) 
    int x=1;
    int y=1;
    while(temp--){
    	//printf("第%d次尝试\n",temp);
    	//每次试验初始化
		//起点坐标重置为(1,1)
		//a[i][j]为非0值代表访问过，为0代表尚未访问 
        x=1;
        y=1;
        for(i=0;i<=7;i++){
            for(j=0;j<=7;j++){
            	if(i==0||j==0){
            		//不属于地图的边缘值设置为1，禁止访问 
            		a[i][j]=1;
				}
				else{
					a[i][j]=0;
				}
            }
        }
        //开始本次试验的随机探路，直至到达终点或无法再走时跳出循环 
        while(1){
        	//若该点上下左右有点(4,4)且已访问2次，则跳出循环 
        	if(x-1==4&&y==4){
        		if(a[x-1][y]==2){
        			break;
				}
			}
        	else if(x+1==4&&y==4){
        		if(a[x+1][y]==2){
        			break;
				}
			}
        	else if(x==4&&y-1==4){
        		if(a[x][y-1]==2){
        			break;
				}
			}
        	else if(x==4&&y+1==4){
        		if(a[x][y+1]==2){
        			break;
				}
			}
			//若无点(4,4)但上下左右均访问过，也跳出循环 
            else if(a[x-1][y]!=0&&a[x+1][y]!=0&&a[x][y-1]!=0&&a[x][y+1]!=0){
                break;
            }
            //选择随机方向 
            int step=rand()%4+1;
            if(step==1){//往上走
            	//若访问次数尚未达到上限，则更新当前坐标，并设置新坐标访问次数+1 
                if(x-1==4&&y==4&&a[x-1][y]<=1){
                    a[x-1][y]++;
                    x--;
                    //printf("%d %d\n",x,y); 
                }            
                else if(x-1>=1&&a[x-1][y]==0){
                    a[x-1][y]++;
                    x--;
                    //printf("%d %d\n",x,y); 
                }
                //若已到达终点则跳出循环，并设置成功次数+1 
                if(x==7&&y==7){
                	//printf("到达终点！\n"); 
                    cnt++;
                    break;
                }
            }
            if(step==2){//往下走
            	//若访问次数尚未达到上限，则更新当前坐标，并设置新坐标访问次数+1 
                if(x+1==4&&y==4&&a[x+1][y]<=1){
                    a[x+1][y]++;
                    x++;
                    //printf("%d %d\n",x,y); 
                }  
                else if(x+1<=7&&a[x+1][y]==0){
                    a[x+1][y]++;
                    x++;
                    //printf("%d %d\n",x,y); 
                }
                //若已到达终点则跳出循环，并设置成功次数+1
                if(x==7&&y==7){
                	//printf("到达终点！\n");
                    cnt++;
                    break;
                }
            }
            if(step==3){//往左走
            	//若访问次数尚未达到上限，则更新当前坐标，并设置新坐标访问次数+1 
                if(x==4&&y-1==4&&a[x][y-1]<=1){
                    a[x][y-1]++;
                    y--;
                    //printf("%d %d\n",x,y); 
                }  
                else if(y-1>=1&&a[x][y-1]==0){
                    a[x][y-1]++;
                    y--;
                    //printf("%d %d\n",x,y); 
                }
                //若已到达终点则跳出循环，并设置成功次数+1
                if(x==7&&y==7){
                	//printf("到达终点！\n");
                    cnt++;
                    break;
                }
            }
            if(step==4){//往右走
            	//若访问次数尚未达到上限，则更新当前坐标，并设置新坐标访问次数+1
                if(x==4&&y+1==4&&a[x][y+1]<=1){
                    a[x][y+1]++;
                    y++;
                    //printf("%d %d\n",x,y); 
                }  
                else if(y+1<=7&&a[x][y+1]==0){
                    a[x][y+1]++;
                    y++;
                    //printf("%d %d\n",x,y); 
                }
                //若已到达终点则跳出循环，并设置成功次数+1
                if(x==7&&y==7){
                	//printf("到达终点！\n");
                    cnt++;
                    break;
                }
            }
        }
    }
    //计算20000次试验中成功走到终点的比率 
    printf("%f\n",1.0*cnt/20000);
}