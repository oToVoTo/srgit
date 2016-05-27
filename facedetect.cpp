#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
//
//
//add dev branch
using namespace std;
using namespace cv;
extern int get_dir_count(char *root);

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
              "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

int detectAndDraw( Mat& img,vector<Mat>& imageRoi,vector<Rect>& faces, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

int get_image_full_name(char *root,vector<Mat>& images,vector<int>& labels,int label)
{
	DIR *dir;
	struct dirent * ptr;
	int total = 0;
	char path[1024];
	dir = opendir(root); /* 打开目录*/
	if(dir == NULL)
	{
		perror("fail to open dir");
		exit(1);
	}

	errno = 0;
	while((ptr = readdir(dir)) != NULL)
	{
		//顺序读取每一个目录项；
		//跳过“..”和“.”两个目录
		if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
		{
			continue;
		}
		//printf("%s%s/n",root,ptr->d_name);
		//如果是目录，则递归调用 get_file_count函数

		if(ptr->d_type == DT_DIR)
		{
		//sprintf(path,"%s%s/",root,ptr->d_name);
		//printf("%s/n",path);
		//total += get_file_count(path);
			//total++;
		}

		if(ptr->d_type == DT_REG)
		{
			//total++;
			//printf("%s%s\n",root,ptr->d_name);
			char imagePath[1024];
			sprintf(imagePath,"%s%s%s",root,"/",ptr->d_name);
			printf("The imagePath is : %s\n",imagePath);
			images.push_back(imread(imagePath,0));
			labels.push_back(label);
		}
	}
	if(errno != 0)
	{
		printf("fail to read dir"); //失败则输出提示信息
		exit(1);
	}
	closedir(dir);
	return total;
}

void trainModel(const char* trainDir,vector<Mat>& images,vector<int>& labels)
{
	DIR* dir;
    struct dirent* ptr;
    int total = 0;
    char path[1024];
    dir = opendir(trainDir);
    if(dir == NULL)
    {
        perror("fail to open dir");
        exit(1);
    }
    errno = 0;
 	while((ptr = readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
        {
            continue;
        }
        if(ptr->d_type == DT_DIR)
        {
            //printf("%s\n",ptr->d_name);
            printf("%d\n",atoi(ptr->d_name));
            char personDirName[255];
            sprintf(personDirName,"%s%s","/home/pig/Face/FaceRecognition/src/",ptr->d_name);
            get_image_full_name(personDirName,images,labels,atoi(ptr->d_name));
        }

   }

}

string cascadeName = "/home/pig/Face/FaceDetect/CascadeDetector_OpenCV/haarcascade_frontalface_alt2.xml";
string nestedCascadeName = "/home/pig/Face/FaceDetect/CascadeDetector_OpenCV/haarcascade_frontalface_alt.xml";

int main(int argc,const char** argv)
{
	
    const char* trainDir="/home/pig/Face/FaceRecognition/src/";
    vector<Mat>images;
    vector<int>labels;

    CvCapture* capture = 0;
    Mat frame,frameCopy;
    capture = cvCaptureFromCAM(0);
    bool tryflip = false;
    double scale = 1.3;
    int regist = 0;
    string yORn;
    CascadeClassifier cascade, nestedCascade;
    cascade.load( cascadeName );
    nestedCascade.load(nestedCascadeName);
    cout<<"Do you want to register a face? y or n"<<endl;
    cin>>yORn;
    cout<<endl;
    cout<<"There will be register 8 times."<<endl;
    if (yORn == "y")
    {
	char faceDirName[25];
	char faceDirFullName[255];
	int faceNum =  get_dir_count("/home/pig/Face/FaceRecognition/src/");
	sprintf(faceDirName, "%d", faceNum);//保存的图片
	sprintf(faceDirFullName,"%s%s","/home/pig/Face/FaceRecognition/src/",faceDirName);
	printf("There has %d faces now!\n",faceNum);
	printf("The faceDirFullName is %s\n",faceDirFullName);
	if( mkdir(faceDirName,0777) == 0 )
	{
		printf( "Directory %s was successfully created\n", "szDirName");
		chdir(faceDirFullName);
	}
	else
		printf( "Problem creating directory '\\testtmp'\n" );


    int registerNum = 0;
	int i = 1;
	  if(capture)
 	    {
	       for(;;)
	 	{
		         string registeryORn = "n";

			if (registerNum == 8)
				{
					cout<<"Ok,you have register 8 times."<<endl;
					break;
				}
	      		IplImage* iplImg = cvQueryFrame( capture );
            		frame = iplImg;
             		if( frame.empty() )
                 	    break;
             		if( iplImg->origin == IPL_ORIGIN_TL )
                 	    frame.copyTo( frameCopy );
             		else
                  	    flip( frame, frameCopy, 0 );
			vector<Mat> imageRoi;
			vector<Rect> faces;
           	int faceNum = detectAndDraw( frameCopy,imageRoi,faces,cascade, nestedCascade, scale, false );
			if (faceNum == 1)
				{
					imshow("imageROi",imageRoi[0]);
					waitKey(10);
				}
 			char image_name[25];
			if (faceNum == 1)
			  {
				cout<<"Register..."<<endl;
				cout<<"Register this face,y or n?"<<endl;
				cin>>registeryORn;
				if (registeryORn == "y")
				{
				     //imshow("imageRoi",imageRoi);

				registerNum++;
				sprintf(image_name, "%d%s", ++i, ".jpg");//保存的图片名  
        			//cvSaveImage( image_name, imageRoi);   //保存一帧图片   
				imwrite(image_name,imageRoi[0]);
				waitKey(100);
				}
		          }
			
		}
 	    }
 }  
    
         trainModel(trainDir,images,labels);
         Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    if( capture )
     {
		
         cout << "In capture ..." << endl;
         for(;;)
        {
			vector<Rect> faces;
             IplImage* iplImg = cvQueryFrame( capture );
             frame = iplImg;
             if( frame.empty() )
                 break;
             if( iplImg->origin == IPL_ORIGIN_TL )
                 frame.copyTo( frameCopy );
             else
                 flip( frame, frameCopy, 0 );
		//imshow("result",frameCopy);
 	     vector<Mat> imageRoi1;
         int FaceNumForCapture = detectAndDraw( frameCopy,imageRoi1,faces, cascade, nestedCascade, scale, tryflip );
 	     if(FaceNumForCapture != 0)
			{
				//cout<<"Detected "<<FaceNumForCapture<<": faces"<<endl;
				model->train(images, labels);
                int predictedLabel = -1;
                double confidence = 0.0;
				int index = 0;
				int i = 0;
				for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++,i++)
					{	
						if (faces.size()!=imageRoi1.size())
							{
								cout<<"Opps,detectandshow has someting wrong!Check it."<<endl;
								return 0;	
							}
						//cout<<"faces.size()="<<faces.size()<<endl;
						//Mat imageRoiDetect = imageRoi1.pop_back();
						Mat imageForPredict = imageRoi1[i];
						if (imageRoi1.size()!=0)
							{
								//imageRoi1.pop_back();
							}
		                model->predict(imageForPredict, predictedLabel, confidence);
		                //cout<<"The label is:"<<predictedLabel<<endl;
		                //cout<<"The confidence is :"<<confidence<<endl;
						
						 if(confidence<70)
		                    {
								index = index+1;
							    if (index==1)
                                cout<<"==================================find a face=================================>"<<endl;

								cout<<"FACE:"<<index<<endl;
        		                char personNum[25];
                		        sprintf(personNum,"%d",predictedLabel);
                        		//cout<<">>>>>>>>>>>>>>>>>>>"<<predictedLabel<<endl;
                       			//cout<<"~~~~~~~~~~~~~~~~~~"<<personNum<<endl;
                       			string personNumstring = personNum;
								//cout<<"r->x=:"<<r->x*scale<<endl;
								//cout<<"r-y=:"<<r->y*scale<<endl;
								//cout<<"(r->x+r->width-1)*scale="<<cvRound((r->x + r->width-1)*scale)<<endl;
								//cout<<"(r->y+r->height-1)*scale="<<cvRound((r->y + r->height-1)*scale)<<endl;
        		                cout<<"The label is:"<<predictedLabel<<endl;
		                        cout<<"The confidence is :"<<confidence<<endl;

                                putText(frameCopy,personNumstring, cvPoint(cvRound(r->x*scale),cvRound(r->y*scale-10)),FONT_HERSHEY_SIMPLEX,3, CV_RGB(255,0,0),4);
								//putText(frameCopy,personNumstring, cvPoint(cvRound(100),cvRound(100)),FONT_HERSHEY_SIMPLEX,3, CV_RGB(255,0,0),4);
                                rectangle( frameCopy, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                                cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                                CV_RGB(255,0,0), 3, 8, 0);
                            		
								//cout<<"<================================================done==============================================>"<<endl;
		                    }
						//cout<<"YES it's here,"<<endl;		
					}
                   

			}
		
        	else
			{
                   /*   for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++ )
                            {
                                rectangle( frameCopy, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                                cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                                CV_RGB(0,0,255), 3, 8, 0);
                            }*/
			}
	
        	imshow("result",frameCopy);
		 	if( waitKey( 10 ) >= 0 )
                 goto _cleanup_;
        
		}
	         waitKey(0);
 
 _cleanup_:
         cvReleaseCapture( &capture );    
}
}


int detectAndDraw( Mat& img, vector<Mat>& imageRoi, vector<Rect>& faces,CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    	Mat imgOrigin;
	//img.copyTo( imgOrigin );
	//imshow("imgOrigin",imgOrigin);
	//waitKey(10);

    int i = 0;
    double t = 0;
    vector<Rect>  faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    cvtColor(img,imgOrigin,CV_BGR2GRAY);
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(130, 130) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            //circle( img, center, radius, color, 3, 8, 0 );
           rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);

        }
        else
           rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                       cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
	  // printf("x=%ff,y=%ff,width=%ff,height=%ff",(r->x*scale),(r->y*scale),((r->width-1)*scale),(( r-> height-1)*scale));

	  // printf("x=%ff,y=%ff,width=%ff,height=%ff",cvRound(r->x*scale),cvRound(r->y*scale),cvRound((r->width-1)*scale),cvRound(( r-> height-1)*scale));
	  
	  //putText(img,"YC", cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),FONT_HERSHEY_SIMPLEX,1, CV_RGB(255,0,0));
	   imageRoi.push_back(imgOrigin(Range(cvRound(r->y*scale),cvRound((r->y + r->height-1)*scale)),Range(cvRound(r->x*scale),cvRound((r->x + r->width-1)*scale))));


       // if( nestedCascade.empty() )
        if(true)   
		 continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            //|CV_HAAR_DO_CANNY_PRUNING
            |CV_HAAR_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
 //   cv::imshow( "result", img );
    //cout<<"Faces num is: "<<faces.size()<<endl;
    return faces.size();
}

int get_dir_count(char *root)
{
DIR *dir;
struct dirent * ptr;
int total = 0;
char path[1024];
dir = opendir(root); /* 打开目录*/
if(dir == NULL)
{
perror("fail to open dir");
exit(1);
}

errno = 0;
while((ptr = readdir(dir)) != NULL)
{
//顺序读取每一个目录项；
//跳过“..”和“.”两个目录
if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
{
continue;
}
//printf("%s%s/n",root,ptr->d_name);
//如果是目录，则递归调用 get_file_count函数

if(ptr->d_type == DT_DIR)
{
//sprintf(path,"%s%s/",root,ptr->d_name);
//printf("%s/n",path);
//total += get_file_count(path);
total++;
}

if(ptr->d_type == DT_REG)
{
//total++;
//printf("%s%s\n",root,ptr->d_name);
}
}
if(errno != 0)
{
printf("fail to read dir"); //失败则输出提示信息
exit(1);
}
closedir(dir);
return total;
}
