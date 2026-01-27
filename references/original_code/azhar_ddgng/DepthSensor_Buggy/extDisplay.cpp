//
//  extDisplay.cpp
//  DepthSensor_Buggy
//
//  Created by Azhar Aulia Saputra on 2020/08/18.
//  Copyright © 2020 Azhar Aulia Saputra. All rights reserved.
//

#include <OpenGL/OpenGL.h>
#include <GLUT/GLUT.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include "extDisplay.hpp"
#include "main.h"
#include "gng.hpp"
#include "projection.h"
#include "surfMatching.hpp"

void GetColourGL(double v, double vmin, double vmax)
{
    double dv;
    float r = 1.0, g = 1.0, b = 1.0;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        r = 0;
        g = 4 * (v - vmin) / dv;
    }
    else if (v < (vmin + 0.5 * dv)) {
        r = 0;
        b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    }
    else if (v < (vmin + 0.75 * dv)) {
        r = 4 * (v - vmin - 0.5 * dv) / dv;
        b = 0;
    }
    else {
        g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        b = 0;
    }
    glColor3d(r, g, b);
}
void drawGridLine(){
    glLineWidth(1);
    glColor3f(0.5, 0.5, 0.5);

    for(float i = -10; i < 10; i++){
        glBegin(GL_LINES);
            glVertex2f(i, -10);
            glVertex2f(i, 10);
        glEnd();
        glBegin(GL_LINES);
            glVertex2f(-10, i);
            glVertex2f(10, i);
        glEnd();
    }
    
    glLineWidth(3);
    glColor3f(0,0, 1);
    glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(1, 0, 0);
    glEnd();
    glColor3f(0,1,0);
    glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 1, 0);
    glEnd();
    glColor3f(1,0,0);
    glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 1);
    glEnd();
}
void drawBox(double x_pos, double y_pos, double z_pos, double x_size, double y_size, double z_size, int label){
    glColor4f(1, 0, 0, 0.5);
    
    glBegin(GL_POLYGON);
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos - z_size/2);       // P1
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos - z_size/2);       // P2
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos - z_size/2);       // P3
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos - z_size/2);       // P4
    glEnd();
    
    glBegin(GL_POLYGON);
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glEnd();
    
    // Purple side - RIGHT
    glBegin(GL_POLYGON);
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glEnd();
    
    // Green side - LEFT
    glBegin(GL_POLYGON);
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos - z_size/2 );
    glEnd();
    
    // Blue side - TOP
    glBegin(GL_POLYGON);
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos + y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos + y_size/2, z_pos + z_size/2 );
    glEnd();
    
    // Red side - BOTTOM
    glBegin(GL_POLYGON);
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos - z_size/2 );
    glVertex3f( x_pos + x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos + z_size/2 );
    glVertex3f( x_pos - x_size/2, y_pos - y_size/2, z_pos - z_size/2 );
    glEnd();
}
int D_TYPE = GL_POINTS;   // drawing type

static float view_xyz[3];    // position x,y,z
static float view_hpr[3];    // heading, pitch, roll (degrees)

static void setCamera (float x, float y, float z, float h, float p, float r)
{
  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity();
  glRotatef (90, 0,0,1);
  glRotatef (90, 0,1,0);
  glRotatef (r, 1,0,0);
  glRotatef (p, 0,1,0);
  glRotatef (-h, 0,0,1);
  glTranslatef (-x,-y,-z);
}
void updateGLView(float hrp[], float xyz[]){
    memcpy (view_xyz,xyz,sizeof(float)*3);
    memcpy (view_hpr,hrp,sizeof(float)*3);
}
void drawLine(double p0[], double p1[], double s){
    glLineWidth(s);
    glBegin(GL_LINES);
        glVertex3f(p0[0], p0[1], p0[2]);
        glVertex3f(p1[0], p1[1], p1[2]);
    glEnd();
}
void drawPoint(double p[], double s){
    glPointSize(s);
    glBegin(GL_POINTS);
    glVertex3d(p[0],p[1],p[2]);
    glEnd();
}
void DrawGNG(struct gng *net)
{
    int i, j;
    dVector3 pos1, pos0;
    dVector3 bbpos;
    dVector3 sides;
    dMatrix3 RI;
    dRSetIdentity (RI);
    double posBuff[1600][3];
    double F[3][3];
    double v1[3], v2[3];
    for (i=0; i<3; i++) sides[i] = 0.005;
    EulerToMatrix(SLAMrot[0] ,SLAMrot[1] ,SLAMrot[2] , F);
    printf("%f\t%f\t%f\n",SLAMrot[0],SLAMrot[1],SLAMrot[2]);
    for (i = 0; i<net->node_n; i++) {
        for(int k=0; k<3; k++){
            v1[k] = net->node[i][k];
        }
        vectorFromMatrixRotation(F, v1, v2);
        for (int k=0; k<3; k++) {
            posBuff[i][k] = v2[k] + SLAMpos[k];
        }
        glColor3f(1, 1, 1);
        for (j = 0; j<i; j++) {
            if (net->edge[i][j] == 1 && net->edge[j][i] == 1) { // membuat koneksi antar node GNG
                drawLine(posBuff[i], posBuff[j], 1);
            }
        }
//        for (int k=0; k<3; k++)             bbpos[k] = net->node[i][k] + SLAMpos[k];
        if(net->normTriangle[i][0] == 0) glColor3f(0,1,0);
        else if(net->normTriangle[i][0] == 1)  glColor3f(1,0,0);
        else if(net->normTriangle[i][0] == 2)  glColor3f(1,1,0.3);
        else if(net->normTriangle[i][0] == 4)  glColor3f(1,0,1);
        else if(net->normTriangle[i][0] == 5)  glColor3f(0,1,1);
        else if(net->normTriangle[i][0] == 6)  glColor3f(0.3,1,0);
        else glColor3f(0.5,0.5,0.5);
        drawPoint(posBuff[i], 6);
    }
    
    
//    for(j=0; j<net->susNode_n; j++){
//        double Pos[3] = {net->node[net->susNode[j]][0],net->node[net->susNode[j]][1],net->node[net->susNode[j]][2]};
//        dsSetColorAlpha (1,0.0,0,0.3);
//        dsDrawSphere (Pos,RI, R_STRE);
//    }
//
}
void DrawCogMap(struct gng *net)
{
    int i, j;
    dVector3 pos1, pos0;
    dVector3 bbpos;
    dVector3 sides;
    dMatrix3 RI;
    dRSetIdentity (RI);
    double posBuff[1600][3];
    double F[3][3];
    double v1[3], v2[3];
    for (i=0; i<3; i++) sides[i] = 0.005;
    glColor3f(0.9,0.9,0.9);
    for (i = 0; i<net->node_n; i++) {
        for (j = 0; j<i; j++) {
            if (net->edge[i][j] == 1 && net->edge[j][i] == 1) { // membuat koneksi antar node GNG
                drawLine(net->node[i], net->node[j], 4);
            }
        }
        glColor3f(0.9,0.9,0.9);
        drawPoint(net->node[i], 12);
    }
}
void drawSuspectedObject(struct gng *net){
//    double F[3][3];
//    EulerToMatrix(SLAMrot[0] ,SLAMrot[1] ,SLAMrot[2] , F);
    for(int i=0; i<net->n_susLabel; i++){
        double size[3], pos[3];
        for(int k = 0; k < 3; k++){
            size[k] = net->susLabelSize[i][1][k] - net->susLabelSize[i][0][k];
            pos[k] =  SLAMpos[k] + (net->susLabelSize[i][1][k] + net->susLabelSize[i][0][k])/2;
        }
        drawBox(pos[0],pos[1],pos[2], size[0], size[1], size[2], 0);
//        drawBox(0,0,0, 2,2,2, 0);
    }
}static float color[4] = {0,0,0,0};
static void setColor (float r, float g, float b, float alpha)
{
  GLfloat light_ambient[4],light_diffuse[4],light_specular[4];
  light_ambient[0] = r*0.3f;
  light_ambient[1] = g*0.3f;
  light_ambient[2] = b*0.3f;
  light_ambient[3] = alpha;
  light_diffuse[0] = r*0.7f;
  light_diffuse[1] = g*0.7f;
  light_diffuse[2] = b*0.7f;
  light_diffuse[3] = alpha;
  light_specular[0] = r*0.2f;
  light_specular[1] = g*0.2f;
  light_specular[2] = b*0.2f;
  light_specular[3] = alpha;
  glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, light_ambient);
  glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, light_diffuse);
  glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, light_specular);
  glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, 5.0f);
}
void DrawCapturedGNG(struct gng *net)
{
    int i, j;
    dVector3 pos1, pos0;
    dVector3 bbpos;
    dVector3 bbpos2;
    dVector3 sides;
    dMatrix3 RI;
    dRSetIdentity (RI);
    double n[3];
    for (i=0; i<3; i++) sides[i] = 0.005;
    for(int c = 0; c < 1; c++){
        for (i = 0; i<net->n_mapData[c]; i++) {
            for (int k=0; k<3; k++){
                bbpos[k] = net->mapData[c][i][k];
                n[k] = net->mapData[c][i][k+3];
            }
            double a = 1;
            if(c == 0){
                if(net->mapAge[i] > 80 && net->mapAge[i] < 110){
                    a = double(110 - net->mapAge[i])/30.0;
                }else if(net->mapAge[i] >= 110){
                    a = 0;
                }
            }
//            if(net->mapData[c][i][6] == 0) glColor3f(0,1,0);
//            else if(net->mapData[c][i][6] == 1)  glColor3f(1,1,0);
//            else if(net->mapData[c][i][6] == 2)  glColor3f(1,0,0);
//            else if(net->mapData[c][i][6] == 4)  glColor3f(1,0,1);
//            else if(net->mapData[c][i][6] == 5)  glColor3f(0,1,1);
//            else glColor3f(0.7,0.2,1);
//            if(c == 1){
//                glColor3f(1,1,0);
//            }
            
            if(net->mapData[c][i][6] == 0) setColor(0,1,0,a);
            else if(net->mapData[c][i][6] == 1)  setColor(1,1,0,a);
            else if(net->mapData[c][i][6] == 2)  setColor(1,0,0,a);
            else if(net->mapData[c][i][6] == 4)  setColor(1,0,1,a);
            else if(net->mapData[c][i][6] == 5)  setColor(0,1,1,a);
            else setColor(0.7,0.2,1,a);
            if(c == 1){
                setColor(1,1,0,a);
            }
            
            if(a > 0)
                drawPoint(bbpos, 6);

//            for (j = 0; j<net->n_mapData[c]; j++) {
//                if(net->mapEdge[c][i][j] == 1|| net->mapEdge[c][j][i] == 1){
//                    for (int k=0; k<3; k++){
//                        bbpos2[k] = net->mapData[c][j][k];
//                    }
//                    drawLine(bbpos, bbpos2, 1);
//                }
//            }
            
            double *u  = vectorScale(net->mapData[c][i][7], n, 3);
            double *v  = vectorAdd(net->mapData[c][i], u, 3);
            if(a > 0)
                Arrow(net->mapData[c][i][0], net->mapData[c][i][1], net->mapData[c][i][2], v[0], v[1], v[2], 0.0015);
//            drawLine(net->mapData[c][i], v, 2);
            free(v);
            free(u);
            
        }
    }
    
//    glColor3f(1,1,1);
    setColor(1,1,1,1);
    drawPoint(SLAMpos, 40);
    
    double v1[3] = {0.7,0,0};
    double v2[3] = {0,0,0};
    double F[3][3];
    
    EulerToMatrix(SLAMrot[0] ,SLAMrot[1] ,SLAMrot[2] , F);
    vectorFromMatrixRotation(F, v1, v2);
    v1[0] = v2[0] + SLAMpos[0];
    v1[1] = v2[1] + SLAMpos[1];
    v1[2] = v2[2] + SLAMpos[2];
//    glColor3f(1,0,0);
    setColor(1,0,0,1);
    Arrow(SLAMpos[0], SLAMpos[1], SLAMpos[2], v1[0], v1[1], v1[2], 0.03);
}
void DrawPointCloud(){
    glPointSize(1);
    glColor3f(0.5, 0.0, 0.0);
    for(int i=0; i<pointCloudActiveTotal; i+=2){
        GetColourGL(sin(PointCloudActive[i][0] * 0.7), -1, 1);
        glBegin(GL_POINTS);
        glVertex3d(PointCloudActive[i][0],PointCloudActive[i][1],PointCloudActive[i][2]);
        glEnd();
    }
}
void cameraSetting(){
        glClearColor(0.0,0.0,0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //    glLoadIdentity();
        
        // snapshot camera position (in MS Windows it is changed by the GUI thread)
        float view2_xyz[3];
        float view2_hpr[3];
        memcpy (view2_xyz,view_xyz,sizeof(float)*3);
        memcpy (view2_hpr,view_hpr,sizeof(float)*3);
        setCamera(view2_xyz[0],view2_xyz[1],view2_xyz[2],
               view2_hpr[0],view2_hpr[1],view2_hpr[2]);
}
int tLoop = 0;
int countCapture = 0;
int tUpdate = 20;
double realPos[3] = {0,0,0};
double testAngle = 0;


void reshape(int width, int height)
{
//    glViewport (0,0,width,height);
    glViewport (0,0,2*width,2*height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    const float vnear = 0.1f;
    const float vfar = 100.0f;
    const float k = 0.8f;     // view scale, 1 = +/- 45 degrees
    if (width >= height) {
      float k2 = float(height)/float(width);
      glFrustum (-vnear*k,vnear*k,-vnear*k*k2,vnear*k*k2,vnear,vfar);
    }
    else {
      float k2 = float(width)/float(height);
      glFrustum (-vnear*k*k2,vnear*k*k2,-vnear*k,vnear*k,vnear,vfar);
    }
    
    glMatrixMode(GL_MODELVIEW);
}void reshape2(int width, int height)
{
//    glViewport (0,0,width,height);
    glViewport (0,0,2*width,2*height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    const float vnear = 0.1f;
    const float vfar = 100.0f;
    const float k = 0.8f;     // view scale, 1 = +/- 45 degrees
    if (width >= height) {
      float k2 = float(height)/float(width);
      glFrustum (-vnear*k,vnear*k,-vnear*k*k2,vnear*k*k2,vnear,vfar);
    }
    else {
      float k2 = float(width)/float(height);
      glFrustum (-vnear*k*k2,vnear*k*k2,-vnear*k,vnear*k,vnear,vfar);
    }
    
    glMatrixMode(GL_MODELVIEW);
}void reshape3(int width, int height)
{
//    glViewport (0,0,width,height);
    glViewport (0,0,2*width,2*height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity();
    const float vnear = 0.1f;
    const float vfar = 100.0f;
    const float k = 0.8f;     // view scale, 1 = +/- 45 degrees
    if (width >= height) {
      float k2 = float(height)/float(width);
      glFrustum (-vnear*k,vnear*k,-vnear*k*k2,vnear*k*k2,vnear,vfar);
    }
    else {
      float k2 = float(width)/float(height);
      glFrustum (-vnear*k*k2,vnear*k*k2,-vnear*k,vnear*k,vnear,vfar);
    }
    
    glMatrixMode(GL_MODELVIEW);
}
int window;
int window2;
int window3;
#pragma mark Timer
GLfloat top = -0.9;
int tcounter=0;
int tcounter2=0;
int tcounter3=0;
void InitGL(int Width, int Height)
{
    initMotionModel();
    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_LIGHTING);
    glEnable (GL_LIGHT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    reshape(Width, Height);
}void InitGL2(int Width, int Height)
{
    initMotionModel();
    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_LIGHTING);
    glEnable (GL_LIGHT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    reshape2(Width, Height);
}void InitGL3(int Width, int Height)
{
    initMotionModel();
    glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_LIGHTING);
    glEnable (GL_LIGHT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    reshape3(Width, Height);
}void cal_env(int vl)//使う
{
    static GLboolean isUp = GL_TRUE;
    
    if (top > 0.9F) isUp = GL_FALSE;
    else if (top <= -0.9F) isUp = GL_TRUE;
    top += (isUp == GL_TRUE ? 0.01 : -0.01);
    glutSetWindow(window);
    display();
    glutPostRedisplay();
    glutTimerFunc(100 , cal_env , 0);
    tcounter++;
}void cal_env2(int vl)//使う
{
    static GLboolean isUp = GL_TRUE;
    
    if (top > 0.9F) isUp = GL_FALSE;
    else if (top <= -0.9F) isUp = GL_TRUE;
    top += (isUp == GL_TRUE ? 0.01 : -0.01);
    glutSetWindow(window2);
    display2();
    glutPostRedisplay();
    glutTimerFunc(100 , cal_env2 , 0);
    tcounter2++;
}void cal_env3(int vl)//使う
{
    static GLboolean isUp = GL_TRUE;
    
    if (top > 0.9F) isUp = GL_FALSE;
    else if (top <= -0.9F) isUp = GL_TRUE;
    top += (isUp == GL_TRUE ? 0.01 : -0.01);
    glutSetWindow(window3);
    display3();
    glutPostRedisplay();
    glutTimerFunc(100 , cal_env3 , 0);
    tcounter3++;
}
void initialExtGL(){
    glutInitWindowSize(750, 450);
    glutInitWindowPosition(750, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    window = glutCreateWindow("Dynamic Attention");

    glutTimerFunc(100, cal_env, 0);
    glutDisplayFunc(&display);
    glutReshapeFunc(&reshape);
    glutKeyboardFunc(&keyboard);
    glutMouseFunc(&MouseEventHandler);
    glutMotionFunc(&MotionEventHandler);
    InitGL(750, 450);
    
    
    glutInitWindowSize(750, 450);
    glutInitWindowPosition(0, 516);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    window2 = glutCreateWindow("Temporal Environmental Reconstruction");

    glutTimerFunc(100, cal_env2, 0);
    glutDisplayFunc(&display2);
    glutReshapeFunc(&reshape2);
    InitGL2(750, 450);
    
    glutInitWindowSize(750, 450);
    glutInitWindowPosition(750, 516);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    window3 = glutCreateWindow("Cognitive Map");

    glutTimerFunc(100, cal_env3, 0);
    glutDisplayFunc(&display3);
    glutReshapeFunc(&reshape3);
    InitGL3(750, 450);
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 'w':
            switch (D_TYPE) {
                case GL_POINTS:
                    D_TYPE = GL_LINE_LOOP;
                    break;
                case GL_LINE_LOOP:
                    D_TYPE = GL_POINTS;
                    break;
            }
            break;
    }
}

static int  mouseButtonMode = 0;
static bool  mouseWithOption = false;    // Set if dragging the mouse with alt pressed
static bool mouseWithControl = false;    // Set if dragging the mouse with ctrl pressed

static int prev_x = 0;
static int prev_y = 0;

int GetModifierMask()
{
    return glutGetModifiers() & ~GLUT_ACTIVE_SHIFT;
}
void MouseEventHandler(int button, int state, int x, int y)
{
    prev_x = x;
    prev_y = y;
    bool buttonDown = false;
    switch( state ){
        case GLUT_DOWN:
            buttonDown = true;
        case GLUT_UP:
            if( button == GLUT_LEFT_BUTTON ){
                int modifierMask = GetModifierMask();
                if( modifierMask & GLUT_ACTIVE_CTRL ){
                    // Ctrl+button == right
                    button = GLUT_RIGHT_BUTTON;
                    mouseWithControl = true;
                }
                else if( modifierMask & GLUT_ACTIVE_ALT ){
                    // Alt+button == left+right
                    mouseButtonMode = 5;
                    mouseWithOption = true;
                    return;
                }
            }
            if( buttonDown ){
                if( button == GLUT_LEFT_BUTTON ) mouseButtonMode |= 1;        // Left
                if( button == GLUT_MIDDLE_BUTTON ) mouseButtonMode |= 2;    // Middle
                if( button == GLUT_RIGHT_BUTTON ) mouseButtonMode |= 4;        // Right
            }
            else{
                if( button == GLUT_LEFT_BUTTON ) mouseButtonMode &= (~1);    // Left
                if( button == GLUT_MIDDLE_BUTTON ) mouseButtonMode &= (~2);    // Middle
                if( button == GLUT_RIGHT_BUTTON ) mouseButtonMode &= (~4);  // Right
            }
            return;
    }
}

// constants to convert degrees to radians and the reverse
#define RAD_TO_DEG (180.0/M_PI)
#define DEG_TO_RAD (M_PI/180.0)

static void wrapCameraAngles()
{
  for (int i=0; i<3; i++) {
    while (view_hpr[i] > 180) view_hpr[i] -= 360;
    while (view_hpr[i] < -180) view_hpr[i] += 360;
  }
}
void dsTrackMotion (int mode, int deltax, int deltay)
{
  float side = 0.01f * float(deltax);
  float fwd = (mode==4) ? (0.01f * float(deltay)) : 0.0f;
  float s = (float) sin (view_hpr[0]* DEG_TO_RAD);
  float c = (float) cos (view_hpr[0]* DEG_TO_RAD);

  if (mode==1) {
    view_hpr[0] += float (deltax) * 0.5f;
    view_hpr[1] += float (deltay) * 0.5f;
  }
  else {
    view_xyz[0] += -s*side + c*fwd;
    view_xyz[1] += c*side + s*fwd;
    if (mode==2 || mode==5) view_xyz[2] += 0.01f * float(deltay);
  }
  wrapCameraAngles();
}
void MotionEventHandler(int x, int y)
{
    dsTrackMotion( mouseButtonMode, x - prev_x, y - prev_y );
    prev_x = x;
    prev_y = y;
}
void initMotionModel()
{
  view_xyz[0] = -0.5;
  view_xyz[1] = 0;
  view_xyz[2] = 1;
  view_hpr[0] = 0;
  view_hpr[1] = -10;
  view_hpr[2] = 0;
}
void display3()
{
    cameraSetting();
    
    if(tLoop > 1){
    DrawCogMap(Cognet);
//    DrawCapturedGNG(ocnet);
    }
    drawGridLine();
    
    glutSwapBuffers();
    glFlush();
}void display2()
{
    cameraSetting();
    
    if(tLoop > 1)
    DrawCapturedGNG(ocnet);
    setColor(0.5,0.5,0.5,1);
    drawGridLine();
    
    glutSwapBuffers();
    glFlush();
}
void display()
{
    cameraSetting();

    if(sensorEnable && tUpdate -- == 0){
        if(captureGNG_node){
            getGNGdata(ocnet,1);
            tUpdate = -100;
            testAngle = 0.1;
        }else{
            
            const dReal * Posiii = dBodyGetPosition(Sensor[0].body);
            for (int k=0; k<3; k++)
                SLAMpos[k] = Posiii[k];
            
            getGNGdata(ocnet,0);
            tUpdate = 20;
        }
        sensor(0);
    }
    if(captureGNG_node){
        if(ocnet->n_mapData[1] > 0){
            surfaceMatching(ocnet);
            if(stepMatch == 5){
                 stepMatch = 0;
                 tUpdate = 20;
                
                printf("rot %f\t%f\t%f\n", SLAMrot[0], SLAMrot[1], SLAMrot[2]);
                printf("Error pos");
                for (int k=0; k<3; k++) {
                    printf("%.5f\t", realPos[k] - SLAMpos[k]);
                    if(dSLAMrot[k]-SLAMrot[k] > 0.2 || dSLAMrot[k]-SLAMrot[k] < -0.2)
                    {
                       printf("Error Boss !!");
                    }
                    dSLAMrot[k] = SLAMrot[k];
                }
                printf("\n");

                const dReal * Posiii = dBodyGetPosition(Sensor[0].body);
                for (int k=0; k<3; k++) {
                    realPos[k] = Posiii[k];
                }
            }
            
            speed = 0;
            
        }
        if (tUpdate > 0){
            speed = 0.2;
        }
    }
//    DrawPointCloud();
    
    int n = 0;
    if (tLoop == 0) {
        ocnet = init_gng(); // step 1
        Cognet = init_gng(); // step 1
        tLoop = 1;
    }
    
    const clock_t begin_time = clock();
    
    for(int i = 0; i < 1; i++){
        n = gng_main(ocnet,  PointCloudActive, pointCloudActiveNum,  tLoop, 1, 2, Cfoc);
    }
    double map_input[10000][3];
    int a = 0;
    for (int j = 0; j<ocnet->n_mapData[0]; j++) {
        if(ocnet->mapData[0][j][6] == 4 || ocnet->mapData[0][j][6] == 4){
            for (int k = 0; k < 3; k++){
                map_input[a][k] = ocnet->mapData[0][j][k];
            }
//            glColor3f(1,1,0);
//            drawPoint(map_input[a], 6);
            a ++;
        }
    }
    if(a > 10){
        n = gng_main(Cognet,  map_input, a,  tLoop, 1, 3, Cfoc);
    }
    if(tLoop % 2 == 0 || tUpdate == 1){
        calc_node_normal_vector(ocnet, sensorPos, 1);
        //calc_node_normal_vector(Cognet, sensorPos, 1);
    }
    
//    for (int i = 0; i<Cognet->node_n; i++) {
//        for (int j = 0; j<i; j++) {
//            if (Cognet->edge[i][j] == 1 && Cognet->edge[j][i] == 1) { // membuat koneksi antar node GNG
//                drawLine(Cognet->node[i], Cognet->node[j], 4);
//            }
//        }
//        glColor3f(0.9,0.9,0.9);
//        drawPoint(Cognet->node[i], 12);
//    }
    printf("GNG cost: %.6f, %d, %d\n",float( clock () - begin_time ) /  CLOCKS_PER_SEC, ocnet->node_n, Cognet->node_n);
    DrawGNG(ocnet);
//    DrawCogMap(Cognet);
//    drawSuspectedObject(ocnet);
    drawGridLine();
//    DrawCapturedGNG(ocnet);
    tLoop++;
    
    
    glutSwapBuffers();
    glFlush();
    
}
