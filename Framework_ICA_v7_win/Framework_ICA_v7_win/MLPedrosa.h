//
//  ConfigMLPedrosa.h
//  OPF_xcode_v2
//
//  Created by Pedro Pedrosa Rebouças Filho on 06/05/15.
//  Copyright (c) 2015 Pedro Pedrosa Rebouças Filho. All rights reserved.
//  Contacts: 'pedrosarf@ifce.edu.br'
//  More informations: http://professorpedrosa.com

#ifndef _MLPedrosa_H_
#define _MLPedrosa_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <stdio.h>
#include <stdlib.h>

#include "OPF/OPF.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"    
#include "opencv2/ml/ml.hpp"
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <windows.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <stdint.h>

#define _CRT_SECURE_NO_WARNINGS

typedef struct timeval 
{
	long tv_sec;
	long tv_usec;
 } timeval;
 
using namespace cv;
using namespace std;

int applyML_PedrosaBayes(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );
int applyML_PedrosaOPF(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int distancia, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );
int applyML_PedrosaKNN(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );
int applyML_PedrosaMLP(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );
int applyML_PedrosaSVM(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, int parametrosAutomaticos, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

#endif
