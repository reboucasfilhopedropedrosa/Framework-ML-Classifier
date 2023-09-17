//
//  ConfigMLPedrosa.h
//  OPF_xcode_v2
//
//  Created by Pedro Pedrosa Rebouças Filho on 06/05/15.
//  Copyright (c) 2015 Pedro Pedrosa Rebouças Filho. All rights reserved.
//  Contacts: 'pedrosarf@ifce.edu.br'
//  More informations: http://professorpedrosa.com


#include <stdio.h>
#include <stdlib.h>

#include "OPF.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "opencv2/opencv.hpp"    // opencv general include file
#include "opencv2/ml/ml.hpp"
#include <vector>

//#define WIN32_LEAN_AND_MEAN
//#include <Windows.h>
#include <sys/timeb.h>
#include <sys/types.h>
//#include <winsock2.h>
#include <stdint.h>
#include <sys/time.h>
//#include <windows.h>

//#define _CRT_SECURE_NO_WARNINGS
// MSVC defines this in winsock2.h!?
/*typedef struct timeval {
 long tv_sec;
 long tv_usec;
 } timeval;
 */

#ifndef OPF_xcode_v2_ConfigMLPedrosa_h
#define OPF_xcode_v2_ConfigMLPedrosa_h

int applyML_PedrosaOPF(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int distancia, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

int applyML_PedrosaKNN(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

int applyML_PedrosaSVM(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, int parametrosAutomaticos, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

int applyML_PedrosaMLP(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

int applyML_PedrosaBayes(char *directorySrc, char *name_file_in_txt, char *name_file_out_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd );

#endif
