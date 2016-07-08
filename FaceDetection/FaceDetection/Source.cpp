#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>
#include<iostream>
#include<fstream>
#include<math.h>
#include<conio.h>
#include<stdio.h>
#define PI 3.141
#define ANGLETHRESH 0.1
#define POINTTHRESH 0.03
#define NAMESTART 1

using namespace cv;
using namespace std;

int counter = 255;

//struct eyeC{
//	int x1,x2,y1,y2;
//};

struct FaceFeatures{
	vector<Point2f> dataPoints;
	vector<float> angles;
	vector<float> distances;
	Point2f imageSize;

};


//class eyecoords{
//public:
//	eyecoords(eyeC left,eyeC right)
//	{
//		x1=(left.x1+left.x2)*0.5;
//		y1=(left.y1+left.y2)*0.5;
//
//		x2=(right.x1+right.x2)*0.5;
//		y2=(right.y1+right.y2)*0.5;
//		dx=x2-x1;
//		dy=y2-y1;
//	dist=sqrt((dx*dx)+(dy*dy));
//		
//	}
//	int x1,y1,x2,y2;
//	double dx,dy;
//	double dist;
//
//	void recal(eyeC left,eyeC right)
//	{
//		x1=(left.x1+left.x2)*0.5;
//		y1=(left.y1+left.y2)*0.5;
//
//		x2=(right.x1+right.x2)*0.5;
//		y2=(right.y1+right.y2)*0.5;
//	}
//};

int detectFace(Mat &pan, CascadeClassifier fd, vector<Rect> &rectFaces, Mat &extFace, Rect &extFaceRect)
{
	fd.detectMultiScale(pan, rectFaces, 1.1f, 4, CASCADE_SCALE_IMAGE, Size(10, 10));
	Rect temp;
	if (rectFaces.size() == 1)
	{
		temp = rectFaces.at(0);
		Mat face(pan, Rect(temp.x, temp.y, temp.width, temp.height));
		//resize(face,face,Size(120,120));
		extFace = face;
		extFaceRect = temp;
		return 1;
	}
	else return 0;
}
float getAngle(Point pt1, Point pt2, Mat &faceImage){
	Point2f vect;
	vect.x = pt1.x - pt2.x; vect.y = pt1.y - pt2.y;
	counter -= 3;

	line(faceImage, pt1, pt2, Scalar(counter), 1, 8);
	if (vect.y == 0) {
		return 90;
	}
	else {
		return (atan(vect.x / vect.y))*(180 / PI);
	}

}
double getDist(Point2f x1, Point2f x2)
{
	float dx = x1.x - x2.x;
	float dy = x1.y - x2.y;
	return sqrt((double)(dx*dx) + (dy*dy));
}

int detectEyes(vector<Rect> &eyes, Mat &face, CascadeClassifier ed, int scaleFactor, float &angle)//,vector<Rect> eyeRects)
{

	vector<Rect> temp;
	ed.detectMultiScale(face, temp, 1.1f, 3, CASCADE_SCALE_IMAGE, Size(10 * (scaleFactor + 1), 10 * (scaleFactor + 1)));

	if (temp.size()>1)
	{
		if (temp.at(0).x > temp.at(1).x){ eyes.push_back(temp.at(1)); eyes.push_back(temp.at(0)); }
		else { eyes.push_back(temp.at(0)); eyes.push_back(temp.at(1)); }
		Mat null(100, 100, 1);
		angle = getAngle(Point(eyes.at(0).x, eyes.at(0).y), Point(eyes.at(1).x, eyes.at(1).y), null);
		return 1;
	}
	else
	{
		return 0;
	}



}
/*

void pre_process_stage_1(Mat before,Mat &after,eyecoords c)
{
double eyecenterx=(c.x1+c.x2)*0.5;
double eyecentery=(c.y1+c.y2)*0.5;
double dx=c.x2-c.x1;
double dy=c.y2-c.y1;
double eyedist=sqrt((dx*dx)+(dy*dy));
double angle=(atan2(dy,dx)*180)/CV_PI;

const double dist_left_eye_f_l_x=0.16;
const double dist_left_eye_f_r_x=1-0.16;

const int desired_width=70;
const int desired_height=70;

//cout<<"\n\neyedist="<<eyedist;

double desired_f_length= dist_left_eye_f_r_x-0.16;
double scale=desired_f_length*desired_width/eyedist;

cout<<"\n\n\n"<<eyecenterx<<"\n"<<eyecentery<<"\n"<<angle<<"\n"<<scale;

Mat rot_mat=getRotationMatrix2D(Point2f(eyecenterx,eyecentery),angle,scale);

double nex=desired_width*0.5-eyecenterx;
double ney=desired_height*0.14-eyecentery;

rot_mat.at<double>(0,2)+=nex;
rot_mat.at<double>(1,2)+=ney;

Mat temp(desired_width,desired_height,CV_8U,Scalar(128));
warpAffine(before,temp,rot_mat,temp.size());


after=temp;
}
*/

void pre_process_stage_2(Mat before, Mat &after)
{
	int i, k, w, h, v;
	Mat lefthalf, righthalf, whole;
	before.copyTo(whole);
	equalizeHist(whole, whole);

	w = whole.cols;
	h = whole.rows;
	lefthalf = before(Rect(0, 0, w / 2, h));
	righthalf = before(Rect(w / 2, 0, w - w / 2, h));

	equalizeHist(lefthalf, lefthalf);
	equalizeHist(righthalf, righthalf);

	for (k = 0; k<h; k++)
	{
		for (i = 0; i<w; i++)
		{
			v = 0;
			if (i <= w / 4)
			{
				v = lefthalf.at<uchar>(k, i);
			}
			else if (i>w / 4 && i<w / 2)
			{
				int lv = lefthalf.at<uchar>(k, i);
				int wv = whole.at<uchar>(k, i);

				float f = (i - w / 4) / (w / 4);
				v = cvRound(((1 - f)*lv) + (f*wv));
			}
			else if (i >= w / 2 && i <= (3 * w / 4))
			{
				int rv = righthalf.at<uchar>(k, i - w / 2);
				int wv = whole.at<uchar>(k, i);

				float f = (i - w / 2) / (w / 4);
				v = cvRound(((1 - f)*wv) + (f*rv));
			}
			else
			{
				v = righthalf.at<uchar>(k, i - w / 2);
			}
			after.at<uchar>(k, i) = v;
		}
	}
}

void pre_process_stage_3(Mat before, Mat &after)
{
	bilateralFilter(before, after, 0, 20.0, 2.0);
}

void pre_process_stage_4(Mat &after)
{
	double fw = 70;
	double fh = 70;

	Mat mask(after.size(), CV_8UC1, Scalar(255));
	Point c(fw*0.5, fh*0.4);
	Size s(cvRound(fw*0.5), cvRound(fh*0.8));
	ellipse(mask, c, s, 0, 0, 360.0f, Scalar(0), CV_FILLED);
	after.setTo(Scalar(128), mask);
}
/*
Mat pre_process(Mat facemat0,eyecoords e)
{
Mat facemat1;

pre_process_stage_1(facemat0,facemat1,e);

Mat facemat2(facemat1.size(),CV_8U);

pre_process_stage_2(facemat1,facemat2);

Mat facemat3(facemat2.size(),CV_8U);

pre_process_stage_3(facemat2,facemat3);

Mat facemat4=facemat3;

pre_process_stage_4(facemat4);

return facemat4;
}
*/

void saveFaceFeatures(FaceFeatures &facefeat, String filename){
	ofstream fout;
	String file;
	file = filename + ".dat";
	fout.open(file);
	fout.write((char*)&facefeat, sizeof(facefeat));
	fout.close();
}

void GetPoints(Mat face, Rect eyeLeft, Rect eyeRight, Rect nose, Rect mouth, FaceFeatures &faceFeat){
	faceFeat.imageSize.x = face.cols;
	faceFeat.imageSize.y = face.rows;

	faceFeat.dataPoints.push_back(Point(eyeLeft.x, eyeLeft.y));
	faceFeat.dataPoints.push_back(Point(eyeLeft.x + eyeLeft.width, eyeLeft.y));
	faceFeat.dataPoints.push_back(Point(eyeLeft.x + (eyeLeft.width / 2), eyeLeft.y + (eyeLeft.height / 2)));

	faceFeat.dataPoints.push_back(Point(eyeRight.x, eyeRight.y));
	faceFeat.dataPoints.push_back(Point(eyeRight.x + eyeRight.width, eyeRight.y));
	faceFeat.dataPoints.push_back(Point(eyeRight.x + (eyeRight.width / 2), eyeRight.y + (eyeRight.height / 2)));

	faceFeat.dataPoints.push_back(Point(nose.x, nose.y));
	faceFeat.dataPoints.push_back(Point(nose.x + nose.width, nose.y));
	faceFeat.dataPoints.push_back(Point(nose.x + (nose.width / 2), nose.y + (nose.height / 2)));

	faceFeat.dataPoints.push_back(Point(mouth.x, mouth.y));
	faceFeat.dataPoints.push_back(Point(mouth.x + mouth.width, mouth.y));
	faceFeat.dataPoints.push_back(Point(mouth.x + (mouth.width / 2), mouth.y + (mouth.height / 2)));

}



void computeFeatures(FaceFeatures &faceFeat, Mat &faceImage){

	for (int i = 0; i < faceFeat.dataPoints.size(); i++){
		for (int j = i + 1; j < faceFeat.dataPoints.size(); j++){
			faceFeat.angles.push_back((float)getAngle(faceFeat.dataPoints.at(i), faceFeat.dataPoints.at(j), faceImage));
			faceFeat.distances.push_back((float)getDist(faceFeat.dataPoints.at(i), faceFeat.dataPoints.at(j)));
		}
	}


}

float compareAngles(FaceFeatures test, FaceFeatures sample){
	assert(test.angles.size() == sample.angles.size());
	int angleMatch = 0;
	for (int i = 0; i < sample.angles.size(); i++){
		if (test.angles.at(i) <= (sample.angles.at(i)*(1 + ANGLETHRESH)) || test.angles.at(i) >= (sample.angles.at(i)*(1 - ANGLETHRESH))){
			angleMatch++;
		}
	}
	float percentage = (float)((float)angleMatch / (float)test.angles.size());
	percentage *= 100;
	return percentage;
}

float comparePoints(FaceFeatures test, FaceFeatures sample){
	assert(test.dataPoints.size() == sample.dataPoints.size());
	int pointMatch = 0;
	for (int i = 0; i < sample.dataPoints.size(); i++){
		test.dataPoints.at(i).x /= test.imageSize.x;    //convert per unit
		test.dataPoints.at(i).y /= test.imageSize.y;
		sample.dataPoints.at(i).x /= sample.imageSize.x;
		sample.dataPoints.at(i).y /= sample.imageSize.y;	//convert per unit
		//cout << "dist is "<<getDist(test.dataPoints.at(i), sample.dataPoints.at(i))<<"\n";
		if (getDist(test.dataPoints.at(i), sample.dataPoints.at(i)) < POINTTHRESH){

			pointMatch++;
		}
	}
	float percentage = (float)((float)pointMatch / (float)test.dataPoints.size());
	percentage *= 100;
	return percentage;
}

void compareFaces(FaceFeatures &test, vector<FaceFeatures> &sampleFaces){
	int result[66] = { 0 };
	int output = 0;
	for (int i = 0; i < sampleFaces.size(); i++){
		//	if (abs((test.angles.at(j) - sampleFaces.at(0).angles.at(j))) - abs((test.angles.at(j) - sampleFaces.at(1).angles.at(j))) > 0){ result[j] = 1; output++; }
		//	cout << " face " << i << "angle percent " << compareAngles(test, sampleFaces.at(i)) << "\n";

		cout << " pan" << NAMESTART + i << ".jpg point percent " << comparePoints(test, sampleFaces.at(i)) << "\n";

	}
	cout << output;
	cout << "\n";
}

void rotateByAngle(Mat dst, double angle)
{
	Point2f pt(dst.cols / 2., dst.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1);
	warpAffine(dst, dst, r, Size(dst.cols, dst.rows), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255, 0));

}

int main(int argc, char* argv[])
{
	double sum;
	int progress = 0;
	Mat image;
	Rect facerect;
	Mat extFace;
	CascadeClassifier fd;
	//eyeC left = { 0 }, right = { 0 };
	String faceClassifier = argv[3];
	faceClassifier+="haarcascade_frontalface_alt.xml";
	fd.load(faceClassifier);

	if (fd.empty())
	{
		cout << "2";//couldnt load classifiers
		return -1;
	}

	vector<Rect> faceRects;
	
	
    	String file = argv[1];
		//String filename("pan");
		//filename += char(namecount + 48);
		//file = filename + ".jpg";
		//String file = argv[0];
		image = imread(file);
		if (image.data == NULL){ cout << "3"; return -1; }
		if (image.channels() >= 2){
			cvtColor(image, image, CV_BGR2GRAY);
		}

		int scaleFactor = ((image.rows / 600) + (image.cols / 443)) / 2;
		if (detectFace(image, fd, faceRects, extFace, facerect))
		{
			float angle = 0;
			progress = 2;//face detected
			cout << "1";
		}
		else{ cout << "0"; }

		String destPath = argv[2];

		ofstream fout;
		fout.open(destPath);
		fout << "$" << facerect.x << "$" << facerect.y << "$" << facerect.width << "$" << facerect.height << "$";
		fout.close();




	return 0;
}
