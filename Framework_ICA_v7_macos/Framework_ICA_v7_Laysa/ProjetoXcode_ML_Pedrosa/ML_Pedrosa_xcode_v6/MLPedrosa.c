//
//  MLPedrosa.c
//  OPF_xcode_v2
//
//  Created by Pedro Pedrosa Rebouças Filho on 06/05/15.
//  Copyright (c) 2015 Pedro Pedrosa Rebouças Filho. All rights reserved.
//  Contacts: 'pedrosarf@ifce.edu.br'
//  More informations: http://professorpedrosa.com

#include "MLPedrosa.h"

int CLASSES = 0;
char file_in_txt [500];
char file_opf_txt [500];
char file_opf_dat [500];
char file_training_dat [500];
char file_testing_dat [500];
char file_evaluating_dat [500];
char file_classifier_opf [500];
char file_distance_dat [500];
char file_results_txt [500];

int TRAINING_SAMPLES;
int TEST_SAMPLES;

float delay_time_rand = 100;

float ** specificityIteracao, **sensitivityIteracao, **ppvIteracao, **accpcIteracao, **fscoreIteracao;//accpc = acuracia por classe
float *specificityTotal,*sensitivityTotal,*accpcTotal,*fscoreTotal,*ppvTotal;

float *timeTrain;// = new float [iteracoes];
float *timeTest;// = new float [iteracoes];

/*int gettimeofday(struct timeval * tp, struct timezone * tzp)
 {
 // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
 static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);
 
 SYSTEMTIME  system_time;
 FILETIME    file_time;
 uint64_t    time;
 
 GetSystemTime( &system_time );
 SystemTimeToFileTime( &system_time, &file_time );
 time =  ((uint64_t)file_time.dwLowDateTime )      ;
 time += ((uint64_t)file_time.dwHighDateTime) << 32;
 
 tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
 tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
 return 0;
 }*/

int sleep_pedrosa(int microsec)
{
    //int milisec = 100; // length of time to sleep, in miliseconds
    struct timespec req = {0};
    req.tv_sec = 0;
    req.tv_nsec = microsec* 1L;//milisec * 1000000L;
    nanosleep(&req, (struct timespec *)NULL);
    return 0;
}

//1∞ passo  - split do database (../bin/opf_split  ../data/boat.dat 0.5 0 0.5 0)
void CheckInputData(float TrPercentage, float EvalPercentage, float TestPercentage){
    fprintf(stderr, "\nSummation of set percentages = %.1f ...",TrPercentage+EvalPercentage+TestPercentage);
    if((float)(TrPercentage+EvalPercentage+TestPercentage) != (float)1.0)
        Error("Percentage summation is not equal to 1","CheckInputData");
    fprintf(stderr, " OK");
    
    fprintf(stderr, "\nChecking set percentages ...");
    if(TrPercentage == 0.0f || TestPercentage == 0.0f)
        Error("Percentage of either training set or test set is equal to 0", "CheckInputData");
    printf(" OK");
}
int splitDatabase(int argc, char **argv){
    
    Subgraph *g = NULL, *gAux = NULL, *gTraining = NULL, *gEvaluating = NULL, *gTesting = NULL;
    float training_p = atof(argv[2]), evaluating_p = atof(argv[3]), testing_p = atof(argv[4]);
    int normalize = atoi(argv[5]);
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that generates training, evaluation and test sets for the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if(argc != 6){
        fprintf(stderr, "\nusage opf_split <P1> <P2> <P3> <P4> <P5>");
        fprintf(stderr, "\nP1: input dataset in the OPF file format");
        fprintf(stderr, "\nP2: percentage for the training set size [0,1]");
        fprintf(stderr, "\nP3: percentage for the evaluation set size [0,1] (leave 0 in the case of no learning)");
        fprintf(stderr, "\nP4: percentage for the test set size [0,1]");
        fprintf(stderr, "\nP5: normalize features? 1 - Yes  0 - No\n\n");
        exit(-1);
    }
    
    CheckInputData(training_p, evaluating_p, testing_p);
    
    fprintf(stdout, "\nReading data set\n %s...",argv[1]); fflush(stdout);
    g = ReadSubgraph(argv[1]);
    fprintf(stdout, " OK"); fflush(stdout);
    
    if(normalize) opf_NormalizeFeatures(g);
    
    fprintf(stdout, "\nSplitting data set ..."); fflush(stdout);
    opf_SplitSubgraph(g, &gAux, &gTesting, training_p+evaluating_p);
    
    if (evaluating_p > 0)
        opf_SplitSubgraph(gAux, &gTraining, &gEvaluating, training_p/(training_p+evaluating_p));
    else gTraining = CopySubgraph(gAux);
    
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting data sets to disk ..."); fflush(stdout);
    WriteSubgraph(gTraining, file_training_dat);
    if (evaluating_p > 0)
        WriteSubgraph(gEvaluating, file_evaluating_dat);
    WriteSubgraph(gTesting, file_testing_dat);
    fprintf(stdout, " OK"); fflush(stdout);
    
    cvWaitKey(1000);
    
    fprintf(stdout, "\nDeallocating memory ...");
    DestroySubgraph(&g);
    DestroySubgraph(&gAux);
    DestroySubgraph(&gTraining);
    DestroySubgraph(&gEvaluating);
    DestroySubgraph(&gTesting);
    fprintf(stdout, " OK\n");
    
    return 0;
}
//2∞ passo - train (training. dat)
float trainDatabase(int argc, char **argv){
    
    int n, i;
    char fileName[256];
    FILE *f = NULL;
    Subgraph *g = NULL;
    struct timeval tic, toc;
    float time;
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that executes the training phase of the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if((argc != 3) && (argc != 2)){
        fprintf(stderr, "\nusage opf_train <P1> <P2>");
        fprintf(stderr, "\nP1: training set in the OPF file format");
        fprintf(stderr, "\nP2: precomputed distance file (leave it in blank if you are not using this resource)\n");
        exit(-1);
    }
    
    
    if(argc == 3) opf_PrecomputedDistance = 1;
    
    fprintf(stdout, "\nReading data file ..."); fflush(stdout);
    g = ReadSubgraph(argv[1]);
    fprintf(stdout, " OK"); fflush(stdout);
    
    if(opf_PrecomputedDistance)
    {
        sprintf(argv[2],"%s", file_distance_dat);
        opf_DistanceValue = opf_ReadDistances(argv[2], &n);
    }
    
    fprintf(stdout, "\nTraining OPF classifier ..."); fflush(stdout);
    
    gettimeofday(&tic,NULL);
    opf_OPFTraining(g);
    gettimeofday(&toc,NULL);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting classifier's model file ..."); fflush(stdout);
    opf_WriteModelFile(g, file_classifier_opf);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"w");
    for (i = 0; i < g->nnodes; i++)
        fprintf(f,"%d\n",g->node[i].label);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nDeallocating memory ..."); fflush(stdout);
    DestroySubgraph(&g);
    if(opf_PrecomputedDistance){
        for (i = 0; i < n; i++)
            free(opf_DistanceValue[i]);
        free(opf_DistanceValue);
    }
    fprintf(stdout, " OK\n");
    
    time = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
    fprintf(stdout, "\nTraining time: %f seconds\n", time); fflush(stdout);
    
    sprintf(fileName,"%s.time",argv[1]);
    f = fopen(fileName,"a");
    fprintf(f,"%f\n",time);
    fclose(f);
    
    return time;
}

//3∞ passo - classifying the test set (testing. dat)
float classifyDatabase(int argc, char **argv){
    
    int n,i;
    float time;
    char fileName[256];
    FILE *f = NULL;
    struct timeval tic, toc;
    Subgraph *gTrain;
    Subgraph *gTest;
    
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that executes the test phase of the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if((argc != 3) && (argc != 2)){
        fprintf(stderr, "\nusage opf_classify <P1> <P2>");
        fprintf(stderr, "\nP1: test set in the OPF file format");
        fprintf(stderr, "\nP2: precomputed distance file (leave it in blank if you are not using this resource\n");
        exit(-1);
    }
    
    
    if(argc == 3) opf_PrecomputedDistance = 1;
    fprintf(stdout, "\nReading data files ..."); fflush(stdout);
    gTest = ReadSubgraph(argv[1]),
    gTrain = opf_ReadModelFile(file_classifier_opf);
    fprintf(stdout, " OK"); fflush(stdout);
    
    if(opf_PrecomputedDistance)
    {
        sprintf(argv[5],"%s", file_distance_dat);
        opf_DistanceValue = opf_ReadDistances(argv[5], &n);
    }
    
    fprintf(stdout, "\nClassifying test set ..."); fflush(stdout);
    gettimeofday(&tic,NULL);
    opf_OPFClassifying(gTrain, gTest); gettimeofday(&toc,NULL);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"w");
    for (i = 0; i < gTest->nnodes; i++)
        fprintf(f,"%d\n",gTest->node[i].label);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    n=gTest->nnodes;
    
    fprintf(stdout, "\nDeallocating memory ...");
    DestroySubgraph(&gTrain);
    DestroySubgraph(&gTest);
    if(opf_PrecomputedDistance){
        for (i = 0; i < n; i++)
            free(opf_DistanceValue[i]);
        free(opf_DistanceValue);
    }
    fprintf(stdout, " OK\n");
    
    time = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
    fprintf(stdout, "\nTesting time: %f seconds\n", time/n); fflush(stdout);
    
    sprintf(fileName,"%s.time",argv[1]);
    f = fopen(fileName,"a");
    fprintf(f,"%f\n",time);
    fclose(f);
    
    return time/n;
}
//4∞ passo - computing the accuracy over the test set (testing. dat)
float accuracyDatabase(int argc, char **argv)
{
    
    int i;
    float Acc;
    FILE *f = NULL;
    Subgraph *g = NULL;
    char fileName[256];
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that computes OPF accuracy of a given set\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if(argc != 2){
        fprintf(stderr, "\nusage opf_accuracy <P1>");
        fprintf(stderr, "\nP1: data set in the OPF file format");
        exit(-1);
    }
    
    fprintf(stdout, "\nReading data file ..."); fflush(stdout);
    g = ReadSubgraph(argv[1]);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nReading output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"r");
    if(!f){
        fprintf(stderr,"\nunable to open file %s", argv[2]);
        exit(-1);
    }
    for (i = 0; i < g->nnodes; i++)
        if (fscanf(f,"%d",&g->node[i].label) != 1) {
            fprintf(stderr,"\nError reading node label");
            exit(-1);
        }
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    //  fprintf(stdout, "\nComputing accuracy ..."); fflush(stdout);
    Acc = opf_Accuracy(g);
    //fprintf(stdout, "\nAccuracy: %.2f%%", Acc*100); fflush(stdout);
    
    
    float acc = 0.0f, countErr = 0.0f;
    
    for (i = 0; i < g->nnodes; i++){
        if(g->node[i].truelabel != g->node[i].label){
            countErr++;
        }
    }
    
    acc = (g->nnodes - countErr) / g->nnodes;
  //  fprintf(stdout, "\nAcuracia Global: %.2f%%", acc*100);
    fflush(stdout);
    
    fprintf(stdout, "\nWriting accuracy in output file ..."); fflush(stdout);
    sprintf(fileName,"%s.acc",argv[1]);
    f = fopen(fileName,"a");
    fprintf(f,"%f\n",acc*100);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nDeallocating memory ..."); fflush(stdout);
    DestroySubgraph(&g);
    fprintf(stdout, " OK\n");
    
    return acc*100;
}
// executing the OPF learning procedure (training. dat evaluating. dat)
int opfLearnopf_learn(int argc, char **argv){
    float Acc, time;
    char fileName[512];
    int i,n;
    timer tic, toc;
    FILE *f = NULL;
    Subgraph *gTrain = NULL;
    Subgraph *gEval  = NULL;
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that executes the learning phase for the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if((argc != 3) && (argc != 4)){
        fprintf(stderr, "\nusage opf_learn <P1> <P2> <P3>");
        fprintf(stderr, "\nP1: training set in the OPF file format");
        fprintf(stderr, "\nP2: evaluation set in the OPF file format");
        fprintf(stderr, "\nP3: precomputed distance file (leave it in blank if you are not using this resource\n");
        exit(-1);
    }
    
    
    if(argc == 4) opf_PrecomputedDistance = 1;
    fprintf(stdout, "\nReading data file ..."); fflush(stdout);
    gTrain = ReadSubgraph(argv[1]);
    gEval = ReadSubgraph(argv[2]);
    fprintf(stdout, " OK"); fflush(stdout);
    
    if(opf_PrecomputedDistance)
    {
        sprintf(argv[3],"%s", file_distance_dat);
        opf_DistanceValue = opf_ReadDistances(argv[3], &n);
    }
    
    fprintf(stdout, "\nLearning from errors in the evaluation set..."); fflush(stdout);
    gettimeofday(&tic,NULL); opf_OPFLearning(&gTrain, &gEval); gettimeofday(&toc,NULL);
    time = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
    Acc = opf_Accuracy(gTrain);
    fprintf(stdout, "\nFinal opf_Accuracy in the training set: %.2f%%", Acc*100); fflush(stdout);
    Acc = opf_Accuracy(gEval);
    fprintf(stdout, "\nFinal opf_Accuracy in the evaluation set: %.2f%%", Acc*100); fflush(stdout);
    
    fprintf(stdout, "\n\nWriting classifier's model file ..."); fflush(stdout);
    opf_WriteModelFile(gTrain, file_classifier_opf); fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nDeallocating memory ...");
    DestroySubgraph(&gTrain);
    DestroySubgraph(&gEval);
    if(opf_PrecomputedDistance){
        for (i = 0; i < n; i++)
            free(opf_DistanceValue[i]);
        free(opf_DistanceValue);
    }
    fprintf(stdout, " OK\n"); fflush(stdout);
    
    sprintf(fileName,"%s.time",argv[1]);
    f = fopen(fileName,"a");
    fprintf(f,"%f\n",time);
    fclose(f);
    
    return 0;
}
//distance normalization (parameter 0 at the final of the command line)
int opf_distance(int argc, char **argv){
    
    int i, j, distance = atoi(argv[2]), normalize = atoi(argv[3]);
    float **Distances = NULL, max = FLT_MIN;
    FILE *fp = NULL;
    Subgraph *sg = NULL;
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that generates the precomputed distance file for the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if(argc != 4){
        fprintf(stderr, "\nusage opf_distance <P1> <P2> <P3>");
        fprintf(stderr, "\nP1: Dataset in the OPF file format");
        fprintf(stderr, "\nP2: Distance ID\n");
        fprintf(stderr, "\n	1 - Euclidean");
        fprintf(stderr, "\n	2 - Chi-Square");
        fprintf(stderr, "\n	3 - Manhattan (L1)");
        fprintf(stderr, "\n	4 - Canberra");
        fprintf(stderr, "\n	5 - Squared Chord");
        fprintf(stderr,"\n	6 - Squared Chi-Squared");
        fprintf(stderr,"\n	7 - BrayCurtis");
        fprintf(stderr, "\nP3: Distance normalization? 1- yes 0 - no");
        exit(-1);
    }
    
    sg = ReadSubgraph(argv[1]);
    fp = fopen(file_distance_dat, "wb");
    
    fwrite(&sg->nnodes, sizeof(int), 1, fp);
    
    Distances  = (float **)malloc(sg->nnodes*sizeof(float *));
    for (i = 0; i < sg->nnodes; i++)
        Distances[i] = (float *)malloc(sg->nnodes*sizeof(int));
    
    switch(distance){
        case 1:
            fprintf(stdout, "\n	Computing euclidean distance ...");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_EuclDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 2:
            fprintf(stdout, "\n	Computing chi-square distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_ChiSquaredDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 3:
            fprintf(stdout, "\n	Computing Manhattan distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_ManhattanDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 4:
            fprintf(stdout, "\n	Computing Canberra distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_CanberraDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 5:
            fprintf(stdout, "\n	Computing Squared Chord distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_SquaredChordDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 6:
            fprintf(stdout, "\n	Computing Squared Chi-squared distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_SquaredChiSquaredDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        case 7:
            fprintf(stdout, "\n	Computing Bray Curtis distance ...\n");
            for (i = 0; i < sg->nnodes; i++){
                for (j = 0; j < sg->nnodes; j++){
                    if(i == j) Distances[i][j] = 0.0;
                    else Distances[sg->node[i].position][sg->node[j].position] = opf_BrayCurtisDist(sg->node[i].feat, sg->node[j].feat, sg->nfeats);
                    if(Distances[sg->node[i].position][sg->node[j].position] > max) max = Distances[sg->node[i].position][sg->node[j].position];
                }
            }
            break;
        default:
            fprintf(stderr, "\nInvalid distance ID ...\n");
    }
    
    if (!normalize) max = 1.0;
    for (i = 0; i < sg->nnodes; i++){
        for (j = 0; j < sg->nnodes; j++){
            Distances[i][j]/=max;
            fwrite(&Distances[i][j], sizeof(float), 1, fp);
        }
    }
    
    fprintf(stdout, "\n\nDistances generated ...\n"); fflush(stdout);
    fprintf(stdout, "\n\nDeallocating memory ...\n");
    for (i = 0; i < sg->nnodes; i++)
        free(Distances[i]);
    free(Distances);
    
    DestroySubgraph(&sg);
    fclose(fp);
    
    
    return 0;
}
//
int opf_cluster(int argc, char **argv){
    int i,n,op;
    float value;
    char fileName[256];
    FILE *f = NULL;
    Subgraph *g = NULL;
    float Hmax;
    double Vmax;
    
    fprintf(stdout, "\nProgram that computes clusters by OPF\n");
    fprintf(stdout, "\n");
    
    if((argc != 6) && (argc != 5)){
        fprintf(stderr, "\nusage opf_cluster <P1> <P2> <P3> <P4> <P5> <P6>");
        fprintf(stderr, "\nP1: unlabeled data set in the OPF file format");
        fprintf(stderr, "\nP2: kmax(maximum degree for the knn graph)");
        fprintf(stderr, "\nP3: P3 0 (height), 1(area) and 2(volume)");
        fprintf(stderr, "\nP4: value of parameter P3 in (0-1)");
        fprintf(stderr, "\nP5: precomputed distance file (leave it in blank if you are not using this resource");
        exit(-1);
    }
    
    if(argc == 6) opf_PrecomputedDistance = 1;
    fprintf(stdout, "\nReading data file ...");
    g = ReadSubgraph(argv[1]);
    
    if(opf_PrecomputedDistance){
        sprintf(argv[5],"%s", file_distance_dat);
        opf_DistanceValue = opf_ReadDistances(argv[5], &n);
    }
    
    op = atoi(argv[3]);
    
    opf_BestkMinCut(g,1,atoi(argv[2])); //default kmin = 1
    
    value = atof(argv[4]);
    if ((value < 1)&&(value>0)){
        fprintf(stdout, "\n\n Filtering clusters ... ");
        switch(op){
            case 0:
                fprintf(stdout, "\n by dome height ... ");
                Hmax=0;
                for (i=0; i < g->nnodes; i++)
                    if (g->node[i].dens > Hmax)
                        Hmax = g->node[i].dens;
                opf_ElimMaxBelowH(g, value*Hmax);
                break;
            case 1:
                fprintf(stdout, "\n by area ... ");
                opf_ElimMaxBelowArea(g, (int)(value*g->nnodes));
                break;
            case 2:
                fprintf(stdout, "\n by volume ... ");
                Vmax=0;
                for (i=0; i < g->nnodes; i++)
                    Vmax += g->node[i].dens;
                opf_ElimMaxBelowVolume(g, (int)(value*Vmax/g->nnodes));
                break;
            default:
                fprintf(stderr, "\nInvalid option for parameter P3 ... ");
                exit(-1);
                break;
        }
    }
    
    fprintf(stdout, "\n\nClustering by OPF ");
    opf_OPFClustering(g);
    printf("num of clusters %d\n",g->nlabels);
    
    /* If the training set has true labels, then create a
     classifier by propagating the true label of each root to
     the nodes of its tree (cluster). This classifier can be
     evaluated by running opf_knn_classify on the training set
     or on unseen testing set. Otherwise, copy the cluster
     labels to the true label of the training set and write a
     classifier, which essentially can propagate the cluster
     labels to new nodes in a testing set. */
    
    if (g->node[0].truelabel!=0){ // labeled training set
        g->nlabels = 0;
        for (i = 0; i < g->nnodes; i++){//propagating root labels
            if (g->node[i].root==i)
                g->node[i].label = g->node[i].truelabel;
            else
                g->node[i].label = g->node[g->node[i].root].truelabel;
        }
        
        for (i = 0; i < g->nnodes; i++){
            // retrieve the original number of true labels
            if (g->node[i].label > g->nlabels)
                g->nlabels = g->node[i].label;
        }
    }else{ // unlabeled training set
        for (i = 0; i < g->nnodes; i++)
            g->node[i].truelabel = g->node[i].label+1;
    }
    
    fprintf(stdout, "\nWriting classifier's model file ..."); fflush(stdout);
    opf_WriteModelFile(g, file_classifier_opf);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"w");
    for (i = 0; i < g->nnodes; i++)
        fprintf(f,"%d\n",g->node[i].label);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\n\nDeallocating memory ...\n");
    DestroySubgraph(&g);
    if(opf_PrecomputedDistance){
        for (i = 0; i < n; i++)
            free(opf_DistanceValue[i]);
        free(opf_DistanceValue);
    }
    
    return 0;
}
//
int opf_knn_classify(int argc, char **argv){
    
    int n,i;
    float time;
    char fileName[256];
    FILE *f = NULL;
    timer tic, toc;
    Subgraph *gTest = NULL;
    Subgraph *gTrain = NULL;
    
    
    fflush(stdout);
    fprintf(stdout, "\nProgram that executes the test phase of the OPF classifier\n");
    fprintf(stdout, "\n"); fflush(stdout);
    
    if((argc != 3) && (argc != 2)){
        fprintf(stderr, "\nusage opf_knn_classify <P1> <P2>");
        fprintf(stderr, "\nP1: test set in the OPF file format");
        fprintf(stderr, "\nP2: precomputed distance file (leave it in blank if you are not using this resource\n");
        exit(-1);
    }
    
    
    if(argc == 3) opf_PrecomputedDistance = 1;
    fprintf(stdout, "\nReading data files ..."); fflush(stdout);
    gTest = ReadSubgraph(argv[1]);
    gTrain = opf_ReadModelFile(file_classifier_opf);
    fprintf(stdout, " OK"); fflush(stdout);
    
    if(opf_PrecomputedDistance)
    {
        sprintf(argv[2],"%s", file_distance_dat);
        opf_DistanceValue = opf_ReadDistances(argv[2], &n);
    }
    
    fprintf(stdout, "\nClassifying test set ..."); fflush(stdout);
    gettimeofday(&tic,NULL);
    opf_OPFKNNClassify(gTrain, gTest); gettimeofday(&toc,NULL);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nWriting output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"w");
    for (i = 0; i < gTest->nnodes; i++)
        fprintf(f,"%d\n",gTest->node[i].label);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nDeallocating memory ...");
    DestroySubgraph(&gTrain);
    DestroySubgraph(&gTest);
    if(opf_PrecomputedDistance){
        for (i = 0; i < n; i++)
            free(opf_DistanceValue[i]);
        free(opf_DistanceValue);
    }
    fprintf(stdout, " OK\n");
    
    time = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
    fprintf(stdout, "\nTesting time: %f seconds\n", time); fflush(stdout);
    
    sprintf(fileName,"%s.time",argv[1]);
    f = fopen(fileName,"a");
    fprintf(f,"%f\n",time);
    fclose(f);
    
    return 0;
}

int opf2txt(int argc, char **argv){
    
    FILE *fpIn = NULL,*fpOut = NULL;
    int n, ndata, nclasses, label, i,j, id;
    float aux;
    
    /*
     100(numero de amostras) 3(qtd de classes) 2 (tam do vetor de atributos)
     0 (sample 0) 1 (classe 1) -1.528822 0.904446 (atributos)
     1 1 -2.036158 -0.042683
     
     */
    if (argc != 3) {
        fprintf(stderr,"\nusage: opf2txt <opf file name> <output file name> \n");
        exit(-1);
    }
    
    fprintf(stderr, "\nProgram to convert files written in the OPF binary format to the OPF ASCII format.");
    
    
    fpIn = fopen(argv[1],"rb");
    fpOut = fopen(argv[2],"w");
    
    /*gravando numero de objetos*/
    fread(&n,sizeof(int),1,fpIn);
    fprintf(fpOut,"%d ",n);
    
    /*gravando numero de classes*/
    fread(&nclasses,sizeof(int),1,fpIn);
    fprintf(fpOut,"%d ",nclasses);
    
    /*gravando tamanho vetor de caracteristicas*/
    fread(&ndata,sizeof(int),1,fpIn);
    fprintf(fpOut,"%d ",ndata);
    
    fprintf(fpOut,"\n");
    /*gravando vetor de caracteristicas*/
    for(i = 0; i < n; i++){
        fread(&id,sizeof(int),1,fpIn);
        fread(&label,sizeof(int),1,fpIn);
        fprintf(fpOut,"%d %d ",id,label);
        for(j = 0; j < ndata; j++)
        {
            fread(&aux,sizeof(float),1,fpIn);
            fprintf(fpOut,"%f ",aux);
        }
        fprintf(fpOut,"\n");
    }
    fclose(fpIn);
    fclose(fpOut);
    
    return 0;
}

int txt2opf(int argc, char **argv)
{
    FILE *fpIn = NULL,*fpOut = NULL;
    int n, ndata, nclasses, i,j, id,label;
    float aux;
    
    if (argc != 3)
    {
        fprintf(stderr,"\nusage txt2opf <P1> <P2>\n");
        fprintf(stderr,"\nP1: input file name in the OPF ASCII format");
        fprintf(stderr,"\nP2: output file name in the OPF binary format\n");
        exit(-1);
    }
    
    fprintf(stderr, "\nProgram to convert files written in the OPF ASCII format to the OPF binary format.");
    
    
    
    fpIn = fopen(argv[1],"r");
    fpOut = fopen(argv[2],"wb");
    
    /*writting the number of samples*/
    if (fscanf(fpIn,"%d",&n) != 1) {
        fprintf(stderr,"Could not read number of samples");
        exit(-1);
    }
    printf("\n number of samples: %d",n);
    fwrite(&n,sizeof(int),1,fpOut);
    
    /*writting the number of classes*/
    if (fscanf(fpIn,"%d",&nclasses) != 1) {
        fprintf(stderr,"Could not read number of classes");
        exit(-1);
    }
    
    printf("\n number of classes: %d",nclasses);
    fwrite(&nclasses,sizeof(int),1,fpOut);
    
    /*writting the number of features*/
    if (fscanf(fpIn,"%d",&ndata) != 1) {
        fprintf(stderr,"Could not read number of features");
        exit(-1);
    }
    
    printf("\n number of features: %d",ndata);
    fwrite(&ndata,sizeof(int),1,fpOut);
    
    /*writting data*/
    
    for(i = 0; i < n; i++)	{
        if (fscanf(fpIn,"%d",&id) != 1) {
            fprintf(stderr,"Could not read sample id");
            exit(-1);
        }
        fwrite(&id,sizeof(int),1,fpOut);
        
        if (fscanf(fpIn,"%d",&label) != 1) {
            fprintf(stderr,"Could not read sample label");
            exit(-1);
        }
        fwrite(&label,sizeof(int),1,fpOut);
        
        for(j = 0; j < ndata; j++)
        {
            if (fscanf(fpIn,"%f",&aux) != 1) {
                fprintf(stderr,"Could not read sample features");
                exit(-1);
            }
            
            fwrite(&aux,sizeof(float),1,fpOut);
        }
    }
    
    fclose(fpIn);
    fclose(fpOut);
    
    return 0;
}

int statistics(int argc, char **argv)
{
    FILE *fpIn = NULL;
    int i, it = atoi(argv[2]);
    float Std = 0.0f, MeanAcc = 0.0f, aux, *acc = NULL;
    
    if(argc != 4){
        fprintf(stderr,"\nusage statistics <file name> <running times> <message>\n");
        exit(-1);
    }
    
    
    
    /*Computing mean accuracy and standard deviation***/
    fpIn = fopen(argv[1],"r");
    if(!fpIn){
        fprintf(stderr,"\nunable to open file %s\n",argv[1]);
        exit(-1);
    }
    
    acc = (float *)malloc(it*sizeof(float));
    for (i = 1;i <= it ; i++){
        if (fscanf(fpIn,"%f",&aux) != 1) {
            fprintf(stderr,"\n Could not read accuracy");
            exit(-1);
        }
        acc[i-1] = aux;
        MeanAcc+=aux;
    }
    MeanAcc/=it;
    for (i = 0; i < it; i++)
        Std+=pow(acc[i]-MeanAcc,2);
    Std=sqrt(Std/it);
    
    fclose(fpIn);
    free(acc);
    
    fprintf(stderr,"\n%s %f with standard deviation: %f\n",argv[3],MeanAcc,Std);
    
    return 0;
}


int mxt2txtopf(int argc, char **argv,int quantidadeAtributos, int quantClasses)
{
    FILE *fpIn = NULL,*fpOut = NULL;
    int n, ndata, nclasses, i,j, id,label,cont=0;;
    float aux;
    
    if (argc != 7)
    {
        fprintf(stderr,"\nusage txt2opf <P1> <P2>\n");
        fprintf(stderr,"\nP1: input file name in the OPF ASCII format");
        fprintf(stderr,"\nP2: output file name in the OPF binary format\n");
        exit(-1);
    }
    
    fprintf(stderr, "\nProgram to convert files written in the any TXT to the OPF ASCII format.");
    
    fpIn = fopen(argv[1],"r");
    fpOut = fopen(argv[2],"wb");
    
    /*writting the number of samples*/
    if (!fpIn) {
        fprintf(stderr,"\n\n\nAtencao:Arquivo de entrada nao existe, ou endereco errado\n\n Verifique se o endereco e arquivo existem neste caminho:\n\n %s\n\n",argv[1]);
        exit(-1);
    }
    float *vet;
    int *vetCont;
    vet = new float[quantidadeAtributos];
    vetCont = new int[quantClasses];

    for (i=0; i<quantClasses; i++)
        vetCont[i]=0;
    
    do      /* Enquanto nao chegar ao final do arquivo */
    {
        for (i=0; i<quantidadeAtributos; i++)
        {
            fscanf(fpIn,"%f,", &vet[i]);
        }
        
        fscanf(fpIn,"%d\n", &id);
        vetCont[id]++;
        cont++;
    }while(getc(fpIn) != EOF);
    
    fclose(fpIn);
    
    for (i=0; i<quantClasses; i++)
        printf("\n%d - %d\n", i, vetCont[i]);
    
    fprintf(fpOut, "%d %d %d", cont, quantClasses, quantidadeAtributos);
    
    fpIn = fopen(argv[1],"r");
    cont = 0;
    for (i=0; i<quantClasses; i++)
        vetCont[i]=0;
    
    do      /* Enquanto nao chegar ao final do arquivo */
    {
        
        for (i=0; i<quantidadeAtributos; i++)
        {
            fscanf(fpIn,"%f,", &vet[i]);
        }
        fscanf(fpIn,"%d\n", &id);
        
        vetCont[id]++;
        
        fprintf(fpOut, "\n%d ", cont);
        
        fprintf(fpOut,"%d ", id+1);
        
        for (i=0; i<quantidadeAtributos; i++)
        {
            fprintf(fpOut, "%f ", vet[i]);
        }
        cont++;
    }while(getc(fpIn) != EOF);
    
    fclose(fpIn);
    
    
    fclose(fpOut);
    
    return 0;
}

int DireitosAutoriais()
{
    printf("\nEsta biblioteca foi desenvolvida por Pedro Pedrosa. \nTodos os direitos autorais deste framework eh reservado ao desenvolvedor.\nAtualmente ele eh professor do Instituto Federal do Ceara (IFCE), \ne trabalha com esta biblioteca com aplicacoes praticas e embarcadas de Machine Learning. \nAs areas principais de suas aplicacoes sao Visao Computacional, Robotica e Processamento de Sinais. \nCaso deseje usar esta biblioteca, peca autorizacao ao desenvolvedor. \nCaso tenha duvidas ou deseje realizar algum projeto de pesquisa em conjunto, estamos a disposicao.Para maiores informacoes seguem os contato abaixo. \nEmail: pedrosarf@ifce.edu.br\nSite: 'professorpedrosa.com'\n\n");
    int it =100000000000000000;
    
    while(it--);
    
    return 0;
}

int** opfmedidamodificadas(int argc, char **argv)
{
    int i,k;
    float Acc;
    size_t result;
    FILE *f = NULL;
    char fileName[256];
    float countErr = 0.0f;
    float auxf;
    float auxi;
    float acc = 0.0f, error = 0.0f;
    int *nclass = NULL, nlabels=0;
    int ** confMatrix;
    float **confMatrixPercentual;
    int j=0;
    
    Subgraph *g = NULL;
    
    if(argc != 2){
        fprintf(stderr, "\nusage opf_accuracy <P1>");
        fprintf(stderr, "\nP1: data set in the OPF file format");
        exit(-1);
    }
    
    fprintf(stdout, "\nReading data file ..."); fflush(stdout);
    g = ReadSubgraph(argv[1]);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nReading output file ..."); fflush(stdout);
    sprintf(fileName,"%s.out",argv[1]);
    f = fopen(fileName,"r");
    if(!f){
        fprintf(stderr,"\nunable to open file %s", argv[2]);
        exit(-1);
    }
    
    for (i = 0; i < g->nnodes; i++)
        result = fscanf(f,"%d",&g->node[i].label);
    fclose(f);
    fprintf(stdout, " OK"); fflush(stdout);
    
    fprintf(stdout, "\nComputing accuracy ..."); fflush(stdout);
    Acc = opf_Accuracy(g);
    // fprintf(stdout, "\nAccuracy: %.2f%%", Acc*100); fflush(stdout);
    
    //O campo truelabel eh carregado quando vocÍ le o arquivo, ou seja, ele corresponde ao rÛtulo original.
    //O campo label armazena o rÛtulo dado pelo classificador.
    
    for (i = 0; i < g->nnodes; i++){
        if(g->node[i].truelabel != g->node[i].label){
            countErr++;
        }
    }
    
    acc = (g->nnodes - countErr) / g->nnodes;
   // fprintf(stdout, "\nAcuracia Global: %.2f%%", acc*100);
    fflush(stdout);
    
    
    fprintf(stdout, "\nModificaÁ„o para reportar acur·cia global, Se e Sp"); fflush(stdout);
    fprintf(stdout, "\nComputing global accuracy, Se and Sp ..."); fflush(stdout);
    
    
    // int **confMatrix;
    
    
    confMatrix = new int*[g->nlabels];
    
    for(int f = 0; f < g->nlabels; f++)
    {
        confMatrix[f] = new int[g->nlabels];
    }
    
    for(i=0; i< g->nlabels; i++)
        for(j=0; j< g->nlabels; j++)
            confMatrix[i][j]=0;
    int res, res_ok;
    for (i = 0; i < g->nnodes; i++){
        //        printf("\ntrueL=%d, label=%d", g->node[i].truelabel, g->node[i].label); fflush(stdout);
        //
        //        if (g->node[i].truelabel != g->node[i].label)
        //            printf("********Errado*********");
        res = g->node[i].truelabel-1;
        res_ok = g->node[i].label-1;
        
        if(res<0) res =0;
         if(res_ok<0) res_ok =0;
        
        confMatrix[res][res_ok]++;
        
    }
    
    CLASSES = (int) (g->nlabels);
    
    return confMatrix;
}

int opfmedidamodificadas(int** confMatrix, int numClasses, char **argv, int indice)
{
    int i;
    int *nclass = NULL;
    //  float **confMatrixPercentual;
    int j=0;
    float * specificity, *sensitivity, *ppredict, *fposrate, *ppv,*accpc,*fscore;//accpc = acuracia por classe
    int TP, TN, FP, FN; //por classe
    int TP_total=0, TN_total=0, FP_total=0, FN_total=0; //por classe
    
    // sprintf(fileName, "%s.results.txt",argv[1]);
    
    // ----
    FILE *fileNameResults = fopen(argv[1],"a");
    fprintf(fileNameResults,"\nResults of Metrics Technics for Distance \n\n");
    
    specificity = new float[numClasses];
    sensitivity = new float[numClasses];
    ppredict = new float[numClasses];
    fposrate = new float[numClasses];
    ppv = new float[numClasses];
    accpc = new float[numClasses];
    fscore = new float[numClasses];
    
    float *numObj = new float[numClasses];//classification_matrix[verdadeiro][predict];
    
    fprintf(fileNameResults,"\n\nLegenda\n");
    
    fprintf(fileNameResults, "\nspe\t-\tspecificity\nsens\t-\tsensitivity\nppv\t-\tppv\naccpc\t-\taccpc\nfscore\t-\tfscore");
    
    fprintf(fileNameResults,"\n\nMedidas do por classe geral\n");
    
    //computa total da matriz
    fprintf(fileNameResults,"\n\n\n\tspe\tsens\tppv\taccpc\tfscore \n");
    
    int sum = 0;
    
    cv::Mat statistics = cv::Mat::zeros(numClasses,4,CV_32FC1);
    cv::Mat results    = cv::Mat::zeros(numClasses,5,CV_32FC1);
    
    int linesum,colsum;	sum = 0;
    printf("\n\t");
    for(int y=0;y<numClasses;y++)
        printf("%d\t",y);
    printf("\n");
    for(int y=0;y<numClasses;y++)
    {
        printf("%d\t",y);
        statistics.at<float>(y,0) = (int) confMatrix[y][y];//allconfmatrix.at(i).at<float>(y,y);
        for(int x=0;x<numClasses;x++)
        {
            sum+=(int)confMatrix[y][x];//(allconfmatrix.at(i).at<float>(y,x));
            printf("%d\t",(int)confMatrix[y][x]);
        }
        printf("\n");
    }
    printf("\n");
    
    for(int y=0;y<numClasses;y++)
    {
        numObj[y] = 0;
        for(int x=0;x<numClasses;x++)
        {
            numObj[y]+= (float) confMatrix[y][x];
        }
    }
    
    printf("\n");
    for(int y=0;y<numClasses;y++)
        printf("\n%d\t-\t%f", y, numObj[y]);
    printf("\n\n");
    for(int y=0;y<numClasses;y++)
    {
        linesum = 0;	colsum  = 0;
        for(int x=0;x<numClasses;x++)
        {
            linesum  = linesum  + (int)confMatrix[y][x];//allconfmatrix.at(i).at<float>(y,x);
            colsum   = colsum   + (int)confMatrix[x][y];//allconfmatrix.at(i).at<float>(x,y);
        }
        statistics.at<float>(y,2) = (float)colsum  - statistics.at<float>(y,0);
        statistics.at<float>(y,3) = (float)linesum - statistics.at<float>(y,0);
        statistics.at<float>(y,1) = (float)sum     - statistics.at<float>(y,0)-statistics.at<float>(y,2)-statistics.at<float>(y,3);
    }
    
    float precision,recall;
    for(int y=0;y<numClasses;y++)
    {
        TP = statistics.at<float>(y,0);
        
        TN = statistics.at<float>(y,1);
        
        FP = statistics.at<float>(y,2);
        
        FN = statistics.at<float>(y,3);
        
        //spe
        //results.at<float>(y,0)
        if((FP+TN) == 0)
            specificity[y] = 0;
        else
            specificity[y] = (float)TN/( FP+TN );
        //        printf("\nspe[%d]=%f\n", y, 100*specificity[y]);
        
        //sens
        //results.at<float>(y,1)
        if((TP+FN) == 0)
            sensitivity[y] = 0;
        else
            sensitivity[y] = (float)TP/( TP+FN );
        //ppv
        //results.at<float>(y,2)
        if((TP+FP) == 0)
            ppv[y] = 0;
        else
            ppv[y] = (float)TP/( TP+FP );
        //accpc
        //results.at<float>(y,3)
        if((TP+FP+FN+TN) == 0)
            accpc[y] = 0;
        else
            accpc[y] = (float)(TP+TN)/(TP+FP+FN+TN);
        
        if((TP+FP) == 0)
            precision = 0;
        else
            precision = (float)TP/(TP+FP);
        
        if((TP+FN) == 0)
            recall = 0;
        else
            recall    = (float)TP/(TP+FN);
        //fscore
        //results.at<float>(y,4)
        if((precision+recall) == 0)
            fscore[y] = 0;
        else
            fscore[y] = (float)(2*precision*recall)/(precision+recall);
        
        
        //        specificity[i] = (float)TN/(TN+FP); //ppv[i] = (float)TP/(TP+FP);//Precis„o da Classe C Taxa de Verdadeiros Negativos.
        fprintf(fileNameResults,"%d\t%.2f",y,specificity[y]);
        //        sensitivity[i] = (float)TP/(TP+FN);//(Taxa de Verdadeiros Positivos) Recall
        fprintf(fileNameResults,"\t%.2f",sensitivity[y]);
        //        ppv[i]         = (float)TP/(TP+FP);   //precision
        fprintf(fileNameResults,"\t%.2f",ppv[y]);
        //        accpc[i]       = (float)(TP+TN)/(TP+TN+FP+FN);//
        fprintf(fileNameResults,"\t%.2f",accpc[y]);
        //        fscore[i]      = (float)(2*(ppv[i]*sensitivity[i]))/(ppv[i]+sensitivity[i]);
        fprintf(fileNameResults,"\t%.2f",fscore[y]);
        fprintf(fileNameResults,"\n");
        
        
        specificityIteracao[y][indice]=specificity[y];
        sensitivityIteracao[y][indice]=sensitivity[y];
        ppvIteracao[y][indice]=ppv[y];
        accpcIteracao[y][indice]=accpc[y] ;
        fscoreIteracao[y][indice]=fscore[y];
    }
    
    fprintf(fileNameResults,"\n\nMedidas do classificador geral\n");
    
    float specificity_total = 0;
    float sensitivity_total = 0;
    float ppv_total         = 0;
    float accpc_total       = 0;
    float fscore_total      = 0;
    
    float totalObj = 0;
    
    for(int y=0;y<numClasses;y++)
    {
        specificity_total += (specificity[y]*numObj[y]);
        sensitivity_total += (sensitivity[y]*numObj[y]);
        ppv_total         += (ppv[y]*numObj[y]);
        accpc_total       += (accpc[y]*numObj[y]);
        fscore_total      += (fscore[y]*numObj[y]);
        
        totalObj +=((float)numObj[y]);
    }
    
    specificity_total /= totalObj;
    sensitivity_total /= totalObj;
    ppv_total         /= totalObj;
    accpc_total       /= totalObj;
    fscore_total      /= totalObj;
    
    fprintf(fileNameResults,"\n\nspe\t-\t%.2f\n",specificity_total);
    fprintf(fileNameResults,"sens\t-\t%.2f\n",sensitivity_total);
    fprintf(fileNameResults,"ppv\t-\t%.2f\n",ppv_total);
    fprintf(fileNameResults,"acc\t-\t%.2f\n",accpc_total);
    fprintf(fileNameResults,"fscore\t-\t%.2f\n",fscore_total);
    fprintf(fileNameResults,"\n");
    
    specificityTotal[indice]=specificity_total;
    sensitivityTotal[indice]=sensitivity_total;
    accpcTotal[indice]=accpc_total;
    fscoreTotal[indice]=fscore_total;
    ppvTotal[indice]=ppv_total;
    
    fprintf(stdout, "\nDeallocating memory ..."); fflush(stdout);
    
    delete [] specificity;
    delete [] sensitivity;
    delete [] ppredict;
    delete [] fposrate;
    delete [] ppv;
    delete [] accpc;
    delete [] fscore;
    delete [] numObj;
    
    
    fprintf(stdout, " OK\n");
    
    fclose(fileNameResults);
    
    statistics.release();
    results.release();
    
    return 0;
}

int printMetricasGerais(int numClasses, char **argv, int iterac,int method, int config)
{
    int i,j;
    
    float iteracoes = ((float) (iterac));
    
    FILE *fileNameResults = fopen(argv[1],"a");
    fprintf(fileNameResults,"\nResults of Metrics Technics for Distance \n\n");
    
    fprintf(fileNameResults,"\n\nLegenda\n");
    
    fprintf(fileNameResults, "\nspe\t-\tspecificity\nsens\t-\tsensitivity\nppv\t-\tppv\naccpc\t-\taccpc\nfscore\t-\tfscore");
    
    
    fprintf(fileNameResults,"\n\nMedidas do por classe geral\n");
    
    float AV[5], sd[5];
    
    //computa total da matriz
    fprintf(fileNameResults,"\n\nclass\tspe\t\tsens\t\tppv\t\tfscore\t\taccpc\n");
    for(i=0; i< numClasses; i++)
    {
        for(j=0; j< 5; j++)
        {
           AV[j] = 0;
           sd[j] = 0;
        }
        
        for(j=0; j< iteracoes; j++)
        {
            AV[0]+=specificityIteracao[i][j];
            AV[1]+=sensitivityIteracao[i][j];
            AV[2]+=ppvIteracao[i][j];
            AV[3]+=fscoreIteracao[i][j];
            AV[4]+=accpcIteracao[i][j];
        }
        
        for(j=0; j< 5; j++)
            AV[j]/=iteracoes;
        
        for(j=0; j< iteracoes; j++)
        {
            sd[0]+=((specificityIteracao[i][j]-AV[0])*(specificityIteracao[i][j]-AV[0]));
            sd[1]+=((sensitivityIteracao[i][j]-AV[1])*(sensitivityIteracao[i][j]-AV[1]));
            sd[2]+=((ppvIteracao[i][j]-AV[2])*(ppvIteracao[i][j]-AV[2]));
            sd[3]+=((fscoreIteracao[i][j]-AV[3])*(fscoreIteracao[i][j]-AV[3]));
            sd[4]+=((accpcIteracao[i][j]-AV[4])*(accpcIteracao[i][j]-AV[4]));
        }
        
        for(j=0; j< 5; j++)
            sd[j]/=iteracoes;
        
        for(j=0; j< 5; j++)
            sd[j]=sqrt(sd[j]);
        
        fprintf(fileNameResults,"%d\t",i);
        
        for(j=0; j< 5; j++)
            fprintf(fileNameResults,"%.2f/%.2f\t",100*AV[j],100*sd[j]);
        fprintf(fileNameResults,"\n");
    }
    
    //geral a partir deste ponto
    
    for(j=0; j< 5; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=specificityTotal[j];
        AV[1]+=sensitivityTotal[j];
        AV[2]+=ppvTotal[j];
        AV[3]+=fscoreTotal[j];
        AV[4]+=accpcTotal[j];
    }
    
    for(j=0; j< 5; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((specificityTotal[j]-AV[0])*(specificityTotal[j]-AV[0]));
        sd[1]+=((sensitivityTotal[j]-AV[1])*(sensitivityTotal[j]-AV[1]));
        sd[2]+=((ppvTotal[j]-AV[2])*(ppvTotal[j]-AV[2]));
        sd[3]+=((fscoreTotal[j]-AV[3])*(fscoreTotal[j]-AV[3]));
        sd[4]+=((accpcTotal[j]-AV[4])*(accpcTotal[j]-AV[4]));
    }
    
    for(j=0; j< 5; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 5; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"Geral\t");
    
    for(j=0; j< 5; j++)
        fprintf(fileNameResults,"%.2f/%.2f\t",100*AV[j],100*sd[j]);
    
    fprintf(fileNameResults,"\n\n\n\n");
    
    //-------------------------------------------------
    //---------------------tempo-----------------------
    //-------------------------------------------------
    
    for(j=0; j< 2; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=timeTrain[j];
        AV[1]+=timeTest[j];
    }
    
    for(j=0; j< 2; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((timeTrain[j]-AV[0])*(timeTrain[j]-AV[0]));
        sd[1]+=((timeTest[j]-AV[1])*(timeTest[j]-AV[1]));
    }
    
    for(j=0; j< 2; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 2; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"\nTempo de treino medio: %.6f/%.6f\n\n",AV[0],sd[0]);
    
    fprintf(fileNameResults,"Tempo de teste de 1 amostra medio: %.8f/%.8f\n\n\n",AV[1],sd[1]);
    
    
    fprintf(fileNameResults, "\n------------------------------------------------------------");
    fprintf(fileNameResults, "\n-----------------------TABELAS LATEX------------------------");
    fprintf(fileNameResults, "\n------------------------------------------------------------\n\n\n");
    
    fprintf(fileNameResults, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Escreva a legenda aqui.}\n\t\\label{tab::M%dC%d}\n\t\\begin{tabular}{cccccc}\n\\hline", method, config);
    fprintf(fileNameResults, "\nCLASS & SPE & SENS & PPV & MH & ACC\\\\ \\hline \n");
    for(i=0; i< numClasses; i++)
    {
        for(j=0; j< 5; j++)
        {
            AV[j] = 0;
            sd[j] = 0;
        }
        
        for(j=0; j< iteracoes; j++)
        {
            AV[0]+=specificityIteracao[i][j];
            AV[1]+=sensitivityIteracao[i][j];
            AV[2]+=ppvIteracao[i][j];
            AV[3]+=fscoreIteracao[i][j];
            AV[4]+=accpcIteracao[i][j];
        }
        
        for(j=0; j< 5; j++)
            AV[j]/=iteracoes;
        
        for(j=0; j< iteracoes; j++)
        {
            sd[0]+=((specificityIteracao[i][j]-AV[0])*(specificityIteracao[i][j]-AV[0]));
            sd[1]+=((sensitivityIteracao[i][j]-AV[1])*(sensitivityIteracao[i][j]-AV[1]));
            sd[2]+=((ppvIteracao[i][j]-AV[2])*(ppvIteracao[i][j]-AV[2]));
            sd[3]+=((fscoreIteracao[i][j]-AV[3])*(fscoreIteracao[i][j]-AV[3]));
            sd[4]+=((accpcIteracao[i][j]-AV[4])*(accpcIteracao[i][j]-AV[4]));
        }
        
        for(j=0; j< 5; j++)
            sd[j]/=iteracoes;
        
        for(j=0; j< 5; j++)
            sd[j]=sqrt(sd[j]);
        
        fprintf(fileNameResults,"%d\t&\t",i);
        
        for(j=0; j< 5; j++)
        {
            fprintf(fileNameResults,"%.2f$\\pm$%.2f\t",100*AV[j],100*sd[j]);
            
            if (j==4)
                fprintf(fileNameResults,"\\\\");
            else
                fprintf(fileNameResults,"&\t");
                
        }
        fprintf(fileNameResults,"\n");
    }
    
    //geral a partir deste ponto
    
    for(j=0; j< 5; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=specificityTotal[j];
        AV[1]+=sensitivityTotal[j];
        AV[2]+=ppvTotal[j];
        AV[3]+=fscoreTotal[j];
        AV[4]+=accpcTotal[j];
    }
    
    for(j=0; j< 5; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((specificityTotal[j]-AV[0])*(specificityTotal[j]-AV[0]));
        sd[1]+=((sensitivityTotal[j]-AV[1])*(sensitivityTotal[j]-AV[1]));
        sd[2]+=((ppvTotal[j]-AV[2])*(ppvTotal[j]-AV[2]));
        sd[3]+=((fscoreTotal[j]-AV[3])*(fscoreTotal[j]-AV[3]));
        sd[4]+=((accpcTotal[j]-AV[4])*(accpcTotal[j]-AV[4]));
    }
    
    for(j=0; j< 5; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 5; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"Geral\t&\t");
    
    for(j=0; j< 5; j++)
    {
        fprintf(fileNameResults,"%.2f$\\pm$%.2f\t",100*AV[j],100*sd[j]);
        
        if (j==4)
            fprintf(fileNameResults,"\\\\");
        else
            fprintf(fileNameResults,"&\t");
    }
    
    fprintf(fileNameResults,"\n\\hline \n\t \\end{tabular} \n \\end{table} ");
    
    fclose(fileNameResults);
    
    return 0;
}

void read_dataset_MLP(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples, int ATTRIBUTES)
{
    
    int label;
    float pixelvalue;
    //open the file
    FILE* inputfile = fopen( filename, "r" );
    
    //read each row of the csv file
    for(int row = 0; row < total_samples; row++)
    {
        //for each attribute in the row
        for(int col = 0; col <=ATTRIBUTES; col++)
        {
            //if its the pixel value.
            if (col < ATTRIBUTES){
                
                fscanf(inputfile, "%f,", &pixelvalue);
                data.at<float>(row,col) = pixelvalue;
                
            }//if its the label
            else if (col == ATTRIBUTES){
                //make the value of label column in that row as 1.
                fscanf(inputfile, "%i", &label);
                classes.at<float>(row,label) = 1.0;
                
            }
        }
    }
    
    fclose(inputfile);
    
}

void read_dataset(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples, int ATTRIBUTES)
{
    
    int label;
    float pixelvalue;
    //open the file
    FILE* inputfile = fopen( filename, "r" );
    
    //read each row of the csv file
    for(int row = 0; row < total_samples; row++)
    {
        //for each attribute in the row
        for(int col = 0; col <=ATTRIBUTES; col++)
        {
            //if its the pixel value.
            if (col < ATTRIBUTES){
                
                fscanf(inputfile, "%f,", &pixelvalue);
                data.at<float>(row,col) = pixelvalue;
                
            }//if its the label
            else if (col == ATTRIBUTES){
                //make the value of label column in that row as 1.
                fscanf(inputfile, "%i", &label);
                classes.at<float>(row,0) = label;
                
            }
        }
    }
    
    fclose(inputfile);
    
}

int printMetricasPrincipais(int numClasses, char **argv, int iterac, int method, int config, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd)
{
    int i,j;
    
    float iteracoes = ((float) (iterac));
    
    FILE *fileNameResults = fopen(argv[1],"a");
    fprintf(fileNameResults,"\nResults of Metrics Technics for Distance \n\n");
    
    fprintf(fileNameResults,"\n\nLegenda\n");
    
    fprintf(fileNameResults, "\nspe\t-\tspecificity\nsens\t-\tsensitivity\nppv\t-\tppv\naccpc\t-\taccpc\nfscore\t-\tfscore");
    
    
    fprintf(fileNameResults,"\n\nMedidas do por classe geral\n");
    
    float AV[4], sd[4];
    
    //computa total da matriz
    fprintf(fileNameResults,"\n\nclass\tspe\t\tsens\t\tPPV\t\taccpc\n");
    for(i=0; i< numClasses; i++)
    {
        for(j=0; j< 4; j++)
        {
            AV[j] = 0;
            sd[j] = 0;
        }
        
        for(j=0; j< iteracoes; j++)
        {
            AV[0]+=specificityIteracao[i][j];
            AV[1]+=sensitivityIteracao[i][j];
            AV[2]+=fscoreIteracao[i][j];
            AV[3]+=accpcIteracao[i][j];
        }
        
        for(j=0; j< 4; j++)
            AV[j]/=iteracoes;
        
        for(j=0; j< iteracoes; j++)
        {
            sd[0]+=((specificityIteracao[i][j]-AV[0])*(specificityIteracao[i][j]-AV[0]));
            sd[1]+=((sensitivityIteracao[i][j]-AV[1])*(sensitivityIteracao[i][j]-AV[1]));
            sd[2]+=((fscoreIteracao[i][j]-AV[2])*(fscoreIteracao[i][j]-AV[2]));
            sd[3]+=((accpcIteracao[i][j]-AV[3])*(accpcIteracao[i][j]-AV[3]));
        }
        
        for(j=0; j< 4; j++)
            sd[j]/=iteracoes;
        
        for(j=0; j< 4; j++)
            sd[j]=sqrt(sd[j]);
        
        fprintf(fileNameResults,"%d\t",i);
        
        for(j=0; j< 4; j++)
            fprintf(fileNameResults,"%.2f/%f\t",100*AV[j],100*sd[j]);
        fprintf(fileNameResults,"\n");
    }
    
    //geral a partir deste ponto
    
    for(j=0; j< 4; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=specificityTotal[j];
        AV[1]+=sensitivityTotal[j];
        AV[2]+=fscoreTotal[j];
        AV[3]+=accpcTotal[j];
    }
    
    for(j=0; j< 4; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((specificityTotal[j]-AV[0])*(specificityTotal[j]-AV[0]));
        sd[1]+=((sensitivityTotal[j]-AV[1])*(sensitivityTotal[j]-AV[1]));
        sd[2]+=((fscoreTotal[j]-AV[2])*(fscoreTotal[j]-AV[2]));
        sd[3]+=((accpcTotal[j]-AV[3])*(accpcTotal[j]-AV[3]));
    }
    
    for(j=0; j< 4; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 4; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"Geral\t");
    
    for(j=0; j< 4; j++)
        fprintf(fileNameResults,"%.2f/%f\t",100*AV[j],100*sd[j]);
    
    fprintf(fileNameResults,"\n\n\n\n");
    
    //-------------------------------------------------
    //---------------------tempo-----------------------
    //-------------------------------------------------
    
    for(j=0; j< 3; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=timeTrain[j];
        AV[1]+=timeTest[j];
        AV[2]+=(timeTest[j]*TEST_SAMPLES);
    }
    
    for(j=0; j< 3; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((timeTrain[j]-AV[0])*(timeTrain[j]-AV[0]));
        sd[1]+=((timeTest[j]-AV[1])*(timeTest[j]-AV[1]));
        sd[2]+=((timeTest[j]-AV[2])*(timeTest[j]-AV[2]));
    }
    
    
    for(j=0; j< 3; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 3; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"\nTempo de treino medio: %.6f/%.6f\n\n",AV[0],sd[0]);
    
    fprintf(fileNameResults,"\nTempo de teste medio: %.6f/%.6f\n\n",AV[2],sd[2]);
    
    fprintf(fileNameResults,"Tempo de teste de 1 amostra medio: %.8f/%.8f\n\n\n",AV[1],sd[1]);
    
    
    if(AV[0]<0.00000001)
        AV[0]=0.00000001;
    
    if(sd[0]<0.00000001)
        sd[0]=0.00000001;
    
    if(AV[1]<0.00000001)
        AV[1]=0.00000001;
    
    if(sd[1]<0.00000001)
        sd[1]=0.00000001;
    
    if(AV[2]<0.00000001)
        AV[2]=0.00000001;
    
    if(sd[2]<0.00000001)
        sd[2]=0.00000001;
    
    testAve = AV[2];
    testStd = sd[2];
    
    trainAve = AV[0];
    trainStd = sd[0];
    
    fprintf(fileNameResults, "\n%%------------------------------------------------------------");
    fprintf(fileNameResults, "\n%%-----------------------TABELAS LATEX------------------------");
    fprintf(fileNameResults, "\n%%------------------------------------------------------------\n\n\n");
    
    fprintf(fileNameResults, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Accuracy (Acc), sensitivity (Se), specificity (Sp) and PPV by class for database of ... obtained by ... method}\n\t\\label{tab::M%dC%d}\n\t\\begin{tabular}{cccccc}\n\\hline", method, config);
    fprintf(fileNameResults, "\nCLASS & Sp($\\%%$) & Se($\\%%$) & PPV ($\\%%$) & Acc($\\%%$)\\\\ \\hline \n");
    for(i=0; i< numClasses; i++)
    {
        for(j=0; j< 4; j++)
        {
            AV[j] = 0;
            sd[j] = 0;
        }
        
        for(j=0; j< iteracoes; j++)
        {
            AV[0]+=specificityIteracao[i][j];
            AV[1]+=sensitivityIteracao[i][j];
            AV[2]+=fscoreIteracao[i][j];
            AV[3]+=accpcIteracao[i][j];
        }
        
        for(j=0; j< 4; j++)
            AV[j]/=iteracoes;
        
        for(j=0; j< iteracoes; j++)
        {
            sd[0]+=((specificityIteracao[i][j]-AV[0])*(specificityIteracao[i][j]-AV[0]));
            sd[1]+=((sensitivityIteracao[i][j]-AV[1])*(sensitivityIteracao[i][j]-AV[1]));
            sd[2]+=((fscoreIteracao[i][j]-AV[2])*(fscoreIteracao[i][j]-AV[2]));
            sd[3]+=((accpcIteracao[i][j]-AV[3])*(accpcIteracao[i][j]-AV[3]));
        }
        
        for(j=0; j< 4; j++)
            sd[j]/=iteracoes;
        
        for(j=0; j< 4; j++)
            sd[j]=sqrt(sd[j]);
        
        fprintf(fileNameResults,"%d\t&\t",i);
        
        for(j=0; j< 4; j++)
        {
            fprintf(fileNameResults,"%.2f$\\pm$%f\t",100*AV[j],100*sd[j]);
            
            if (j==3)
                fprintf(fileNameResults,"\\\\");
            else
                fprintf(fileNameResults,"&\t");
            
        }
        fprintf(fileNameResults,"\n");
    }
    
    //geral a partir deste ponto
    
    for(j=0; j< 4; j++)
    {
        AV[j] = 0;
        sd[j] = 0;
    }
    
    for(j=0; j< iteracoes; j++)
    {
        AV[0]+=specificityTotal[j];
        AV[1]+=sensitivityTotal[j];
        AV[2]+=fscoreTotal[j];
        AV[3]+=accpcTotal[j];
    }
    
    for(j=0; j< 4; j++)
        AV[j]/=iteracoes;
    
    for(j=0; j< iteracoes; j++)
    {
        sd[0]+=((specificityTotal[j]-AV[0])*(specificityTotal[j]-AV[0]));
        sd[1]+=((sensitivityTotal[j]-AV[1])*(sensitivityTotal[j]-AV[1]));
        sd[2]+=((fscoreTotal[j]-AV[2])*(fscoreTotal[j]-AV[2]));
        sd[3]+=((accpcTotal[j]-AV[3])*(accpcTotal[j]-AV[3]));
    }
    
    for(j=0; j< 4; j++)
        sd[j]/=iteracoes;
    
    for(j=0; j< 4; j++)
        sd[j]=sqrt(sd[j]);
    
    fprintf(fileNameResults,"Geral\t&\t");
    
    for(j=0; j< 4; j++)
    {
        fprintf(fileNameResults,"%.2f$\\pm$%f\t",100*AV[j],100*sd[j]);
        
        if (j==3)
            fprintf(fileNameResults,"\\\\");
        else
            fprintf(fileNameResults,"&\t");
    }
    
    fprintf(fileNameResults,"\n\\hline \n\t \\end{tabular} \n \\end{table} ");
    
    fclose(fileNameResults);
    
    AV[3] = 100*AV[3];
    sd[3] = 100*sd[3];
    
    AV[2] = 100*AV[2];
    sd[2] = 100*sd[2];

    if(AV[3]<0.01)
        AV[3]=0.01;
    
    if(sd[3]<0.01)
        sd[3]=0.01;
    
    if(AV[2]<0.01)
        AV[2]=0.01;
    
    if(sd[2]<0.01)
        sd[2]=0.01;
    
    accAve = AV[3];
    accStd = sd[3];
    
    fscoreAve= AV[2];
    fscoreStd= sd[2];
    
    return 0;
}

int splitDataBaseOpenCvMethods(int argc, char **argv, float percentTraining,int quantidadeAtributos, int quantClasses, int normalize)
{
    srand(time(NULL));
    
    
    FILE *fpIn = NULL,*fpOut = NULL,*fpOutTest = NULL;
    int n, ndata, nclasses, i,j, id,label,cont=0;;
    float aux;
    
    if (argc != 7)
    {
        fprintf(stderr,"\nusage txt2opf <P1> <P2>\n");
        fprintf(stderr,"\nP1: input file name in the OPF ASCII format");
        fprintf(stderr,"\nP2: output file name in the OPF binary format\n");
        exit(-1);
    }
    
    fprintf(stderr, "\nProgram to convert files written in the any TXT to the OPF ASCII format.");
    
    
    fpIn = fopen(argv[1],"r");
    fpOut = fopen(argv[2],"wb");
    fpOutTest = fopen(argv[3],"wb");
    
    
    float *vetMax = new float[quantidadeAtributos];
    float *vetMin = new float[quantidadeAtributos];
    float *vet = new float[quantidadeAtributos];
    float *vetDif = new float[quantidadeAtributos];
    int *vetCont = new int[quantClasses];
    
    for (i=0; i<quantClasses; i++)
        vetCont[i]=0;
    
    std::vector <float*> DBdata;
    std::vector <int> DBclass;
    
    do      /* Enquanto nao chegar ao final do arquivo */
    {
        for (i=0; i<quantidadeAtributos; i++)
        {
            fscanf(fpIn,"%f,", &vet[i]);
            
            if (cont==0) {
                vetMax[i] = vet[i];
                vetMin[i] = vet[i];
            }
            else{
                if(vetMax[i] < vet[i])
                    vetMax[i] = vet[i];
                if(vetMin[i] > vet[i])
                    vetMin[i] = vet[i];
            }
        }
        fscanf(fpIn,"%d\n", &id);
        vetCont[id]++;
        //
        cont++;
    }while(getc(fpIn) != EOF);
    
    fclose(fpIn);
    
    for (i=0; i<quantidadeAtributos; i++)
    {
        vetDif[i] = vetMax[i] - vetMin[i];
        
        if (vetDif[i]==0)
            vetDif[i] = 1;
    }
    
    for (i=0; i<quantClasses; i++)
        printf("\n%d - %d\n", i, vetCont[i]);
    
    // fprintf(fpOut, "%d %d %d", cont+1 , 10, 1225);
    //fprintf(fpOut, "%d %d %d", 10000, 10, 1225);
    
    fpIn = fopen(argv[1],"r");
    cont = 0;
    for (i=0; i<quantClasses; i++)
        vetCont[i]=0;
    
    do      /* Enquanto nao chegar ao final do arquivo */
    {
        float* vt=new float[quantidadeAtributos];
        
        for (i=0; i<quantidadeAtributos; i++)
        {
            fscanf(fpIn,"%f,", &vt[i]);
            
            if (normalize)
                vt[i] = ((vt[i]-vetMin[i])/(vetDif[i]));
        }
        fscanf(fpIn,"%d\n", &id);
        //id = vet[1224];
        
        DBdata.push_back(vt);
        DBclass.push_back(id);
        
        vetCont[id]++;
        
        cont++;
        
    }while(getc(fpIn) != EOF);
    
    fclose(fpIn);
    
    
    //------------------------------------------------------------------//
    
    int contTraining, contTest;
    
    int totalTrainning = 0;
    int totalTest = 0;
    
    for (int classe = 0 ; classe < quantClasses ; classe++)
    {
        std::vector <int> vetTemp;
        
        for (int ex = 0 ; ex < cont ; ex++)
        {
            if (DBclass.at(ex)==classe)
            {
                vetTemp.push_back(ex);
            }
        }
        
        contTraining = (int)(percentTraining*((float)vetCont[classe]));
        
        totalTrainning+=contTraining;
        
        contTest = vetCont[classe] - contTraining;
        
        totalTest+=contTest;
        
        //printf("\n\n\nAMOSTRAS\n\n");
        
        
        for (int ex = 0 ; ex < contTraining ; ex++)
        {
            int x, y;
           // int contDelay = 2;
            
            do
            {
                x = rand()%vetCont[classe];
                sleep_pedrosa(1);
                //while(contDelay--);
                cvWaitKey(1);
                
            }while(vetTemp.at(x)==-1);
            
            //delay_time_rand = delay_time_rand*4/5;
            
            //printf("%d, ", x);
            
            y = vetTemp.at(x);
            
            vetTemp.at(x)=-1;
            
            
            for (i=0; i<quantidadeAtributos; i++)//for (i=0; i<quantidadeAtributos/25; i++)
            {
                fprintf(fpOut, "%f,", DBdata.at(y)[i]);
            }
            
            fprintf(fpOut,"%d\n", classe);
            
        }
        //printf("\n\n\n");
        
        
        for (int ex = 0 ; ex <  vetCont[classe] ; ex++)
        {
            if (vetTemp.at(ex)!=-1)
            {
                int x = vetTemp.at(ex);
                
                for (i=0; i<quantidadeAtributos; i++)//for (i=0; i<quantidadeAtributos/25; i++)
                {
                    fprintf(fpOutTest, "%f,", DBdata.at(x)[i]);
                }
                
                fprintf(fpOutTest,"%d\n", classe);
            }
        }
        
        
        vetTemp.clear();
    }
    
    fclose(fpOut);
    
    fclose(fpOutTest);
    
    printf("\n\nTotal de treino = %d \nTotal para teste = %d ", totalTrainning, totalTest);
    
    delete [] vetMax;
    delete [] vetMin;
    delete [] vet;
    delete [] vetCont;
    
    TEST_SAMPLES=totalTest;
    TRAINING_SAMPLES=totalTrainning;
    
    
    return 0;
}


int applyML_PedrosaOPF(char *directorySrc, char *name_file_in_txt, char *name_file_results_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int distancia, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd )
{
    timeTrain = new float [iteracoes];
    timeTest = new float [iteracoes];
    
    CLASSES=quantClasses;
    
    specificityTotal = new float[iteracoes];
    sensitivityTotal = new float[iteracoes];
    accpcTotal = new float[iteracoes];
    fscoreTotal = new float[iteracoes];
    ppvTotal = new float[iteracoes];
    
    specificityIteracao = new float*[quantClasses];
    sensitivityIteracao = new float*[quantClasses];
    ppvIteracao = new float*[quantClasses];
    accpcIteracao = new float*[quantClasses];
    fscoreIteracao = new float*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        specificityIteracao[f] = new float[iteracoes];
        sensitivityIteracao[f] = new float[iteracoes];
        ppvIteracao[f] = new float[iteracoes];
        accpcIteracao[f] = new float[iteracoes];
        fscoreIteracao[f] = new float[iteracoes];
    }
    
    int **confMatrixMedia;
    
    confMatrixMedia = new int*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        confMatrixMedia[f] = new int[quantClasses];
    }
    
    for(int i=0; i< quantClasses; i++)
        for(int j=0; j< quantClasses; j++)
            confMatrixMedia[i][j]=0;
    
    int objTotal = 0;
    
    //DireitosAutoriais();
    
    int argc;
    char **argv;
    
    for(int it = 0 ; it< iteracoes; it++)
    {
        char *name_file_opf_txt = "padronizado_opf.txt";
        char *name_file_opf_dat = "padronizado_opf.dat";
        
        char *name_file_training_dat= "training.dat";
        char *name_file_testing_dat= "testing.dat";
        char *name_file_evaluating_dat = "evaluating.dat";
        char *name_file_classifier_opf = "classifier.opf";
        char *name_file_distance_dat = "distances.dat";
        
        //char *name_file_results_txt = "results";//.txt eh colocado depois
        
        sprintf(file_results_txt, "%s/%s", directorySrc, name_file_results_txt);
        
        sprintf(file_in_txt, "%s/%s", directorySrc, name_file_in_txt);
        sprintf(file_opf_txt, "%s/%s", directorySrc, name_file_opf_txt);
        sprintf(file_opf_dat, "%s/%s", directorySrc, name_file_opf_dat);
        
        sprintf(file_training_dat, "%s/%s", directorySrc, name_file_training_dat);
        
        sprintf(file_testing_dat, "%s/%s", directorySrc, name_file_testing_dat);
        
        sprintf(file_evaluating_dat, "%s/%s", directorySrc, name_file_evaluating_dat);
        
        sprintf(file_classifier_opf, "%s/%s", directorySrc, name_file_classifier_opf);//distancia ,
        
        sprintf(file_distance_dat, "%s/%s", directorySrc, name_file_distance_dat);
        
        argv = new char*[7];
        
        for(int f = 0; f < 7; f++)
        {
            argv[f] = new char[500];
        }
    
        argc = 7;
        sprintf(argv[1],"%s",file_in_txt);
        sprintf(argv[2],"%s",file_opf_txt);
        
        mxt2txtopf(argc, argv, quantidadeAtributos, quantClasses);
        
        //-------------------------------------FERRAMENTAS - conversao de arquivos------------------------
        //opf2txt(argc, argv);
        //txt2opf(argc, argv);
        
        argc = 3;
        
        sprintf(argv[1],"%s",file_opf_txt);
        sprintf(argv[2],"%s",file_opf_dat);
        
        txt2opf(argc, argv);
        
        //-------------------------------------Distance-------------------------------------
        
        setDistanceOPFPedrosa(distancia);
        
//        if(it==0)
//        {
            argc = 4;
            sprintf(argv[1],"%s",file_opf_dat);
            sprintf(argv[2],"%d",distancia);
            sprintf(argv[3],"%d",normalize);
            
            opf_distance(argc, argv);
        //}/**/
        
        //-------------------------------------SPLIT DO BANCO DE DADOS------------------------------------
        
        //---This is an example of how to use the OPF classifier without learning procedure.---------------
        /*
         P1: input dataset in the OPF file format
         P2: percentage for the training set size [0,1]
         P3: percentage for the evaluation set size [0,1] (leave 0 in the case of no learning)
         P4: percentage for the test set size [0,1]
         P5: normalize features? 1 - Yes  0 - No
         */
        
        argc = 6;
        sprintf(argv[1],"%s",file_opf_dat);
        sprintf(argv[2],"%f",percentTreino);
        sprintf(argv[3],"0");
        sprintf(argv[4],"%f",1-percentTreino);
        sprintf(argv[5],"%d",normalize);
        splitDatabase(argc,argv);
        /**/
        
        //-------------------------------------TREINAMENTO ---------------------------------------------
        // p1 (training. dat)
        
        argc = 2; //3 para distancia pre-compilada
        sprintf(argv[1],"%s",file_training_dat);
        timeTrain[it] = trainDatabase(argc,argv);
        /**/
        
        //-------------------------------------CLASSIFICA«√O --------------------------------------------
        // p2 (testing. dat)
        
        argc = 2; //3 para distancia pre compilada
        sprintf(argv[1],"%s",file_testing_dat);
        timeTest[it] = classifyDatabase(argc,argv);
        /**/
        
        //-------------------------------------MEDIDAS ESTATISTICA---------------------------------------
        // p2 (testing. dat)
        
        argc = 2;
        sprintf(argv[1],"%s",file_testing_dat);
        float acc = accuracyDatabase(argc,argv);
        /**/
        
        
        //------------------------------------------------------------
        //---------------------------OPF------------------------------
        //------------------------------------------------------------
        
        
        argc = 2;
        sprintf(argv[1],"%s",file_testing_dat);
        sprintf(argv[2],"%s",file_results_txt);
        int ** classification_matrix = opfmedidamodificadas(argc,argv);
        
        
        printf( "\nImprimindo Matriz de consufsao...");
        FILE *fpMatrizConfusao = NULL;
        
        //sprintf(file_results_txt, "%s/%s%f_%d.txt", directorySrc, name_file_results_txt, acc,it);
        
        if (it==0)
            fpMatrizConfusao = fopen(file_results_txt,"w");
        else
            fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "Metodo - OPF \nBase - %s\n", name_file_in_txt);
        switch (distancia) {
            case 1:
                fprintf(fpMatrizConfusao, "Distancia - Euclidiana\n");
                break;
            case 2:
                fprintf(fpMatrizConfusao, "Distancia - Chi-Square\n");
                break;
            case 3:
                fprintf(fpMatrizConfusao, "Distancia - Manhattan (L1)\n");
                break;
            case 4:
                fprintf(fpMatrizConfusao, "Distancia - Canberra\n");
                break;
            case 5:
                fprintf(fpMatrizConfusao, "Distancia - Squared Chord\n");
                break;
            case 6:
                fprintf(fpMatrizConfusao, "Distancia - Squared Chi-Squared\n");
                break;
            case 7:
                fprintf(fpMatrizConfusao, "Distancia - BrayCurtis\n");
                break;
            case 8:
                fprintf(fpMatrizConfusao, "Distancia - Gaussian\n");
                break;
            case 9:
                fprintf(fpMatrizConfusao, "Distancia - Mahalanobis\n");
                break;
            default:
                break;
        }
        
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        fprintf(fpMatrizConfusao, "\n-------------------------Iteracao %d------------------------", it+1);
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        
        
        fprintf(fpMatrizConfusao, "\n\nTempos\n\nTrain\t-\t%f\nTeste 1 amostra\t-\t%f",timeTrain[it],timeTest[it]);
        
        fprintf(fpMatrizConfusao, "\n\n\nMatriz de confusao\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        int totalObj = 0;
        float totalObjetos;
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%d\t",classification_matrix[row][col]);
                totalObj+=classification_matrix[row][col];
                confMatrixMedia[row][col]+=classification_matrix[row][col];
                objTotal+=totalObj;
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        totalObjetos = (float)totalObj;
        printf( "Ok!");
        
        //----------------------------------------------
        //----------------------------------------------
        //----------------------------------------------
        
        /*printf( "\nImprimindo Matriz de confusao percentual...");
        
        fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%.2f\t", (float)classification_matrix[row][col]*100/totalObjetos);
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        
        printf( "Ok!");*/
        
        sprintf(argv[1],"%s",file_results_txt);
        
        opfmedidamodificadas(classification_matrix, CLASSES, argv, it);
    }
    
    
    FILE *fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------RESUMO GERAL------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    
    fprintf(fpMatrizConfusao, "\n\nMatriz de confusao Media\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    
   /* printf( "\nImprimindo Matriz de confusao percentual...");
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", (float)confMatrixMedia[row][col]*100/objTotal);
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    printf( "Ok!");*/
 
    sprintf(argv[1],"%s",file_results_txt);
    
    
    
    //---------------------------------------------------------
    //----------------PRINT METRICAS ACC, HM, SE,SP------------
    //---------------------------------------------------------
    
    printMetricasPrincipais(CLASSES, argv, iteracoes, 1, distancia, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
    //printMetricasGerais(CLASSES, argv, iteracoes, 1, distancia);
    
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%----------MATRIZ DE CONFUSAO LATEX MEDIA NUMERICA-----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Numerical average confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCNumericoM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 1, distancia);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    
    
   /* fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%---------MATRIZ DE CONFUSAO LATEX MEDIA PERCENTUAL----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Average percentage confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCPercentualM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 1, distancia);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", (float)confMatrixMedia[row][col]*100/objTotal);
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");*/

    
    fclose(fpMatrizConfusao);
    
    delete [] timeTrain;
    delete [] timeTest;
    
    delete [] specificityTotal;
    delete [] sensitivityTotal;
    delete [] accpcTotal;
    delete [] fscoreTotal;
    delete [] ppvTotal;
    
    for(int f = 0; f < quantClasses; f++)
    {
         delete [] specificityIteracao[f];
         delete [] sensitivityIteracao[f];
         delete [] ppvIteracao[f];
         delete [] accpcIteracao[f];
         delete [] fscoreIteracao[f];
    }
    
    delete [] specificityIteracao;
    delete [] sensitivityIteracao;
    delete [] ppvIteracao;
    delete [] accpcIteracao;
    delete [] fscoreIteracao;
    
    printf("\n\nAcabou a distancia %d\n\n", distancia);
    
    return 0;
}


int applyML_PedrosaKNN(char *directorySrc, char *name_file_in_txt, char *name_file_results_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int numK, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd )
{
    timeTrain = new float [iteracoes];
    timeTest = new float [iteracoes];
    
    CLASSES=quantClasses;
    
    specificityTotal = new float[iteracoes];
    sensitivityTotal = new float[iteracoes];
    accpcTotal = new float[iteracoes];
    fscoreTotal = new float[iteracoes];
    ppvTotal = new float[iteracoes];
    
    specificityIteracao = new float*[quantClasses];
    sensitivityIteracao = new float*[quantClasses];
    ppvIteracao = new float*[quantClasses];
    accpcIteracao = new float*[quantClasses];
    fscoreIteracao = new float*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        specificityIteracao[f] = new float[iteracoes];
        sensitivityIteracao[f] = new float[iteracoes];
        ppvIteracao[f] = new float[iteracoes];
        accpcIteracao[f] = new float[iteracoes];
        fscoreIteracao[f] = new float[iteracoes];
    }
    
    int **confMatrixMedia;
    
    confMatrixMedia = new int*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        confMatrixMedia[f] = new int[quantClasses];
    }
    
    for(int i=0; i< quantClasses; i++)
        for(int j=0; j< quantClasses; j++)
            confMatrixMedia[i][j]=0;
    
    int objTotal = 0;
    
    //DireitosAutoriais();
    
    int argc;
    char **argv;
    
    for(int it = 0 ; it< iteracoes; it++)
    {
        char *name_file_training_dat= "training.dat";
        char *name_file_testing_dat= "testing.dat";
        
        //char *name_file_results_txt = "results";//.txt eh colocado depois
        
        sprintf(file_results_txt, "%s/%s", directorySrc, name_file_results_txt);
        
        sprintf(file_in_txt, "%s/%s", directorySrc, name_file_in_txt);
        
        sprintf(file_training_dat, "%s/%s", directorySrc, name_file_training_dat);
        
        sprintf(file_testing_dat, "%s/%s", directorySrc, name_file_testing_dat);
        
       
        argv = new char*[7];
        
        for(int f = 0; f < 7; f++)
        {
            argv[f] = new char[500];
        }
        
        argc = 7;
        
        sprintf(argv[1],"%s",file_in_txt);
        sprintf(argv[2],"%s",file_training_dat);
        sprintf(argv[3],"%s",file_testing_dat);
        
        //-------------------------------------SPLIT DO BANCO DE DADOS------------------------------------
        splitDataBaseOpenCvMethods(argc, argv, percentTreino ,quantidadeAtributos, quantClasses, normalize);
        
        
        //-----------------------CARREGANDO BANCO DE DADOS PARA TREINO E TESTE----------------------------
        
        //matrix to hold the training sample
        cv::Mat training_set(TRAINING_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the labels of each taining sample
        cv::Mat training_set_classifications(TRAINING_SAMPLES, 1, CV_32FC1);
        //cv::Mat training_set_classifications(TRAINING_SAMPLES, quantClasses, CV_32F);
        //matric to hold the test samples
        cv::Mat test_set(TEST_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the test labels.
        cv::Mat test_set_classifications(TEST_SAMPLES,1,CV_32FC1);
        //cv::Mat test_set_classifications(TEST_SAMPLES,quantClasses,CV_32F);
        
        //
        cv::Mat classificationResult(1, 1, CV_32FC1);
        //cv::Mat classificationResult(1, CLASSES, CV_32FC1);
        //load the training and test data sets.
        read_dataset(argv[2], training_set, training_set_classifications, TRAINING_SAMPLES, quantidadeAtributos);
        read_dataset(argv[3], test_set, test_set_classifications, TEST_SAMPLES, quantidadeAtributos);
        
        
        
        //------------------------------------------------------------
        //-----------------KNN TREINAMENTO----------------------------
        //------------------------------------------------------------
        
        int K = numK;
        
        struct timeval tic, toc;
        
        printf("\nTreinando knn...");
        gettimeofday(&tic,NULL);
        CvKNearest knn(training_set, training_set_classifications, cv::Mat(), false, K);
        gettimeofday(&toc,NULL);
        printf("OK!\n\n");
        
        timeTrain[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        printf("\n\nTraining time: %f seconds\n\n", timeTrain[it]);
        
        //------------------------------------------------------------
        //-----------------KNN CLASSIFICACAO--------------------------
        //------------------------------------------------------------
        
        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;
        
        //classification matrix gives the count of classes to which the samples were classified.
        //int classification_matrix[CLASSES][CLASSES]={{}};
        int numClassses = (int)CLASSES;
        int ** classification_matrix;
        classification_matrix = new int*[numClassses];
        
        for(int f = 0; f < numClassses; f++)
        {
            classification_matrix[f] = new int[numClassses];
        }
        
        for(int row=0;row<CLASSES;row++)
            for(int col=0;col<CLASSES;col++)
                classification_matrix[row][col]=0;
        
        gettimeofday(&tic,NULL);
        // for each sample in the test set.
        for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {
            
            // extract the sample
            
            test_sample = test_set.row(tsample);
            
            //SVM.predict(test_sample, classificationResult);
            // float res3 = classificationResult.at<float>(0,0);
            int res =knn.find_nearest(test_sample, K);
            int test = (int) (test_set_classifications.at<float>(tsample,0));
            
            printf("Testing Sample %i -> class result (digit %d\t%d)\n", tsample, res, test);
            
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            
            classification_matrix[test][res]++;
            
            if (test!=res)
            {
                // if they differ more than floating point error => wrong class
                
                wrong_class++;
                
            } else {
                
                // otherwise correct
                
                correct_class++;
            }
            
        }
        
        gettimeofday(&toc,NULL);
        timeTest[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        timeTest[it]=timeTest[it]/((float)(TEST_SAMPLES));
        printf("\n\nTest time: %f seconds\n\n", timeTest[it]);
        
        //------------------------------------------------------------
        //---------------------------KNN------------------------------
        //------------------------------------------------------------
        
        printf( "\nImprimindo Matriz de consufsao...");
        FILE *fpMatrizConfusao = NULL;
        
        //sprintf(file_results_txt, "%s/%s%f_%d.txt", directorySrc, name_file_results_txt, acc,it);
        
        if (it==0)
            fpMatrizConfusao = fopen(file_results_txt,"w");
        else
            fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "Metodo - KNN \nBase - %s\n", name_file_in_txt);
        
        fprintf(fpMatrizConfusao, "K - %d\n", numK);
        
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        fprintf(fpMatrizConfusao, "\n-------------------------Iteracao %d------------------------", it+1);
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        
        
        fprintf(fpMatrizConfusao, "\n\nTempos\n\nTrain\t-\t%f\nTeste 1 amostra\t-\t%f",timeTrain[it],timeTest[it]);
        
        fprintf(fpMatrizConfusao, "\n\n\nMatriz de confusao\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        int totalObj = 0;
        float totalObjetos;
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%d\t",classification_matrix[row][col]);
                totalObj+=classification_matrix[row][col];
                confMatrixMedia[row][col]+=classification_matrix[row][col];
                objTotal+=totalObj;
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        totalObjetos = (float)totalObj;
        printf( "Ok!");
        
        //----------------------------------------------
        //----------------------------------------------
        //----------------------------------------------
        
        /*printf( "\nImprimindo Matriz de confusao percentual...");
        
        fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%.2f\t", (float)classification_matrix[row][col]*100/totalObjetos);
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        
        printf( "Ok!");*/
        
        sprintf(argv[1],"%s",file_results_txt);
        
        opfmedidamodificadas(classification_matrix, CLASSES, argv, it);
    }
    
    
    FILE *fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------RESUMO GERAL------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    
    fprintf(fpMatrizConfusao, "\n\nMatriz de confusao Media\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    
    /*printf( "\nImprimindo Matriz de confusao percentual...");
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", (float)confMatrixMedia[row][col]*100/objTotal);
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    printf( "Ok!");*/
    
    sprintf(argv[1],"%s",file_results_txt);
    
    
    
    //---------------------------------------------------------
    //----------------PRINT METRICAS ACC, HM, SE,SP------------
    //---------------------------------------------------------
    
    printMetricasPrincipais(CLASSES, argv, iteracoes, 2, numK, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
    //printMetricasGerais(CLASSES, argv, iteracoes, 2, numK);
    
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%----------MATRIZ DE CONFUSAO LATEX MEDIA NUMERICA-----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Numerical average confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCNumericoM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 2, numK);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    
    
    /*fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%---------MATRIZ DE CONFUSAO LATEX MEDIA PERCENTUAL----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Average percentage confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCPercentualM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 2, numK);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", (float)confMatrixMedia[row][col]*100/objTotal);
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");*/
    
    
    fclose(fpMatrizConfusao);
    
    delete [] timeTrain;
    delete [] timeTest;
    
    delete [] specificityTotal;
    delete [] sensitivityTotal;
    delete [] accpcTotal;
    delete [] fscoreTotal;
    delete [] ppvTotal;
    
    for(int f = 0; f < quantClasses; f++)
    {
        delete [] specificityIteracao[f];
        delete [] sensitivityIteracao[f];
        delete [] ppvIteracao[f];
        delete [] accpcIteracao[f];
        delete [] fscoreIteracao[f];
    }
    
    delete [] specificityIteracao;
    delete [] sensitivityIteracao;
    delete [] ppvIteracao;
    delete [] accpcIteracao;
    delete [] fscoreIteracao;
    
    printf("\n\nAcabou o K %d\n\n", numK);
    
    return 0;
}


int applyML_PedrosaSVM(char *directorySrc, char *name_file_in_txt, char *name_file_results_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int kernell, int normalize, int parametrosAutomaticos, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd)
{
    timeTrain = new float [iteracoes];
    timeTest = new float [iteracoes];
    
    CLASSES=quantClasses;
    
    specificityTotal = new float[iteracoes];
    sensitivityTotal = new float[iteracoes];
    accpcTotal = new float[iteracoes];
    fscoreTotal = new float[iteracoes];
    ppvTotal = new float[iteracoes];
    
    specificityIteracao = new float*[quantClasses];
    sensitivityIteracao = new float*[quantClasses];
    ppvIteracao = new float*[quantClasses];
    accpcIteracao = new float*[quantClasses];
    fscoreIteracao = new float*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        specificityIteracao[f] = new float[iteracoes];
        sensitivityIteracao[f] = new float[iteracoes];
        ppvIteracao[f] = new float[iteracoes];
        accpcIteracao[f] = new float[iteracoes];
        fscoreIteracao[f] = new float[iteracoes];
    }
    
    int **confMatrixMedia;
    
    confMatrixMedia = new int*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        confMatrixMedia[f] = new int[quantClasses];
    }
    
    for(int i=0; i< quantClasses; i++)
        for(int j=0; j< quantClasses; j++)
            confMatrixMedia[i][j]=0;
    
    int objTotal = 0;
    
    //DireitosAutoriais();
    
    int argc;
    char **argv;
    
    for(int it = 0 ; it< iteracoes; it++)
    {
        char *name_file_training_dat= "training.dat";
        char *name_file_testing_dat= "testing.dat";
        
        //char *name_file_results_txt = "results";//.txt eh colocado depois
        
        sprintf(file_results_txt, "%s/%s", directorySrc, name_file_results_txt);
        
        sprintf(file_in_txt, "%s/%s", directorySrc, name_file_in_txt);
        
        sprintf(file_training_dat, "%s/%s", directorySrc, name_file_training_dat);
        
        sprintf(file_testing_dat, "%s/%s", directorySrc, name_file_testing_dat);
        
        
        argv = new char*[7];
        
        for(int f = 0; f < 7; f++)
        {
            argv[f] = new char[500];
        }
        
        argc = 7;
        
        sprintf(argv[1],"%s",file_in_txt);
        sprintf(argv[2],"%s",file_training_dat);
        sprintf(argv[3],"%s",file_testing_dat);
        
        //-------------------------------------SPLIT DO BANCO DE DADOS------------------------------------
        splitDataBaseOpenCvMethods(argc, argv, percentTreino ,quantidadeAtributos, quantClasses, normalize);
        
        
        //-----------------------CARREGANDO BANCO DE DADOS PARA TREINO E TESTE----------------------------
        
        //matrix to hold the training sample
        cv::Mat training_set(TRAINING_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the labels of each taining sample
        cv::Mat training_set_classifications(TRAINING_SAMPLES, 1, CV_32FC1);
        //cv::Mat training_set_classifications(TRAINING_SAMPLES, quantClasses, CV_32F);
        //matric to hold the test samples
        cv::Mat test_set(TEST_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the test labels.
        cv::Mat test_set_classifications(TEST_SAMPLES,1,CV_32FC1);
        //cv::Mat test_set_classifications(TEST_SAMPLES,quantClasses,CV_32F);
        
        //
        cv::Mat classificationResult(1, 1, CV_32FC1);
        //cv::Mat classificationResult(1, CLASSES, CV_32FC1);
        //load the training and test data sets.
        read_dataset(argv[2], training_set, training_set_classifications, TRAINING_SAMPLES, quantidadeAtributos);
        read_dataset(argv[3], test_set, test_set_classifications, TEST_SAMPLES, quantidadeAtributos);
        
        
        
        //------------------------------------------------------------
        //-----------------SVM TREINAMENTO----------------------------
        //------------------------------------------------------------
        
        
        CvSVMParams param = CvSVMParams();
        
        param.svm_type = CvSVM::C_SVC;
        
        switch (kernell)
        {
            case 1:
                param.kernel_type = CvSVM::LINEAR;
                break;
                
            case 2:
                param.kernel_type = CvSVM::RBF;
                break;
                
            case 3:
                param.kernel_type = CvSVM::POLY;
                break;
                
            case 4:
                param.kernel_type = CvSVM::SIGMOID;
                break;
                
            default:
                param.kernel_type = CvSVM::LINEAR;
                break;
        }
        
        param.degree = 1; // for poly
        param.gamma  = 20; // for poly/rbf/sigmoid
        param.coef0  = 10; // for poly/sigmoid
        
        param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        param.p = 0.0; // for CV_SVM_EPS_SVR
        
        param.class_weights = NULL; // for CV_SVM_C_SVC
        param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
        param.term_crit.max_iter = 1000;
        param.term_crit.epsilon = 1e-6;
        
        
        // SVM training
        CvSVM SVM;
        
        struct timeval tic, toc;
        
        if(parametrosAutomaticos)
        {
        printf( "\nSVM training using dataset...");
        gettimeofday(&tic,NULL);
            
        int iteractions = SVM.train_auto(training_set, training_set_classifications, cv::Mat(), cv::Mat(), param);
            
        gettimeofday(&toc,NULL);
        printf( "OK\n\n%d Iteracoes", iteractions);
        }
        else
        {
            printf( "\nSVM training using dataset...");
            gettimeofday(&tic,NULL);
            int iteractions = SVM.train(training_set, training_set_classifications, cv::Mat(), cv::Mat(), param);
            gettimeofday(&toc,NULL);
            printf( "OK\n\n%d Iteracoes", iteractions);
        }
        timeTrain[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        printf("\n\nTraining time: %f seconds\n\n", timeTrain[it]);
        
        //------------------------------------------------------------
        //-----------------SVM CLASSIFICACAO--------------------------
        //------------------------------------------------------------
        
        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;
        
        //classification matrix gives the count of classes to which the samples were classified.
        //int classification_matrix[CLASSES][CLASSES]={{}};
        
        int numClassses = (int)CLASSES;
        int ** classification_matrix;
        classification_matrix = new int*[numClassses];
        
        for(int f = 0; f < numClassses; f++)
        {
            classification_matrix[f] = new int[numClassses];
        }
        
        for(int row=0;row<CLASSES;row++)
            for(int col=0;col<CLASSES;col++)
                classification_matrix[row][col]=0;
        
        gettimeofday(&tic,NULL);
        // for each sample in the test set.
        for (int tsample = 0; tsample < TEST_SAMPLES; tsample++)
        {
            
            // extract the sample
            
            test_sample = test_set.row(tsample);
            
            //SVM.predict(test_sample, classificationResult);
            // float res3 = classificationResult.at<float>(0,0);
            
            int res = (int) (SVM.predict(test_sample));
            int test = (int) (test_set_classifications.at<float>(tsample,0));
            
            //printf("Testing Sample %i -> class result (digit %d\t%d)\n", tsample, res, test);
            
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            
            classification_matrix[test][res]++;
            
            if (test!=res)
            {
                // if they differ more than floating point error => wrong class
                
                wrong_class++;
                
            } else {
                
                // otherwise correct
                
                correct_class++;
            }
            
        }
        
        gettimeofday(&toc,NULL);
        timeTest[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        timeTest[it]=timeTest[it]/((float)(TEST_SAMPLES));
        printf("\n\nTest time: %f seconds\n\n", timeTest[it]);
        
        //------------------------------------------------------------
        //---------------------------SVM------------------------------
        //------------------------------------------------------------
        
        printf( "\nImprimindo Matriz de consufsao...");
        FILE *fpMatrizConfusao = NULL;
        
        //sprintf(file_results_txt, "%s/%s%f_%d.txt", directorySrc, name_file_results_txt, acc,it);
        
        if (it==0)
            fpMatrizConfusao = fopen(file_results_txt,"w");
        else
            fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "Metodo - SVM \nBase - %s\n", name_file_in_txt);
        
        switch (kernell) {
            case 1:
                fprintf(fpMatrizConfusao, "kernell - Linear\n");
                break;
            case 2:
                fprintf(fpMatrizConfusao, "kernell - RBF\n");
                break;
            case 3:
                fprintf(fpMatrizConfusao, "kernell - Poly\n");
                break;
            case 4:
                fprintf(fpMatrizConfusao, "kernell - SIGMOID\n");
                break;
            default:
                break;
        }
        
        switch (parametrosAutomaticos) {
            case 0:
                fprintf(fpMatrizConfusao, "Parametros fixos\n");
                break;
            case 1:
                fprintf(fpMatrizConfusao, "Parametros automaticos (otimizados)\n");
                break;
            default:
                break;
        }
        
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        fprintf(fpMatrizConfusao, "\n-------------------------Iteracao %d------------------------", it+1);
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        
        
        fprintf(fpMatrizConfusao, "\n\nTempos\n\nTrain\t-\t%f\nTeste 1 amostra\t-\t%f",timeTrain[it],timeTest[it]);
        
        fprintf(fpMatrizConfusao, "\n\n\nMatriz de confusao\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        int totalObj = 0;
        float totalObjetos;
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%d\t",classification_matrix[row][col]);
                totalObj+=classification_matrix[row][col];
                confMatrixMedia[row][col]+=classification_matrix[row][col];
                objTotal+=totalObj;
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        totalObjetos = (float)totalObj;
        printf( "Ok!");
        
        //----------------------------------------------
        //----------------------------------------------
        //----------------------------------------------
        
        /*printf( "\nImprimindo Matriz de confusao percentual...");
        
        fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%.2f\t", (float)classification_matrix[row][col]*100/totalObjetos);
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        
        printf( "Ok!");*/
        
        sprintf(argv[1],"%s",file_results_txt);
        
        opfmedidamodificadas(classification_matrix, CLASSES, argv, it);
    }
    
    
    FILE *fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------RESUMO GERAL------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    
    fprintf(fpMatrizConfusao, "\n\nMatriz de confusao Media\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    
    /*printf( "\nImprimindo Matriz de confusao percentual...");
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", (float)confMatrixMedia[row][col]*100/objTotal);
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    printf( "Ok!");*/
    
    sprintf(argv[1],"%s",file_results_txt);
    
    
    
    //---------------------------------------------------------
    //----------------PRINT METRICAS ACC, HM, SE,SP------------
    //---------------------------------------------------------
    
    printMetricasPrincipais(CLASSES, argv, iteracoes, 3, kernell, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
    //printMetricasGerais(CLASSES, argv, iteracoes, 3, kernell);
    
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%----------MATRIZ DE CONFUSAO LATEX MEDIA NUMERICA-----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Numerical average confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCNumericoM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 3, kernell);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    
    
   /* fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%---------MATRIZ DE CONFUSAO LATEX MEDIA PERCENTUAL----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Average percentage confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCPercentualM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 3, kernell);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", (float)confMatrixMedia[row][col]*100/objTotal);
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");*/
    
    
    fclose(fpMatrizConfusao);
    
    delete [] timeTrain;
    delete [] timeTest;
    
    delete [] specificityTotal;
    delete [] sensitivityTotal;
    delete [] accpcTotal;
    delete [] fscoreTotal;
    delete [] ppvTotal;
    
    for(int f = 0; f < quantClasses; f++)
    {
        delete [] specificityIteracao[f];
        delete [] sensitivityIteracao[f];
        delete [] ppvIteracao[f];
        delete [] accpcIteracao[f];
        delete [] fscoreIteracao[f];
    }
    
    delete [] specificityIteracao;
    delete [] sensitivityIteracao;
    delete [] ppvIteracao;
    delete [] accpcIteracao;
    delete [] fscoreIteracao;
    
    printf("\n\nAcabou o SVM %d\n\n", kernell);
    
    return 0;
}

int applyML_PedrosaBayes(char *directorySrc, char *name_file_in_txt, char *name_file_results_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int config, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd )
{
    timeTrain = new float [iteracoes];
    timeTest = new float [iteracoes];
    
    CLASSES=quantClasses;
    
    specificityTotal = new float[iteracoes];
    sensitivityTotal = new float[iteracoes];
    accpcTotal = new float[iteracoes];
    fscoreTotal = new float[iteracoes];
    ppvTotal = new float[iteracoes];
    
    specificityIteracao = new float*[quantClasses];
    sensitivityIteracao = new float*[quantClasses];
    ppvIteracao = new float*[quantClasses];
    accpcIteracao = new float*[quantClasses];
    fscoreIteracao = new float*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        specificityIteracao[f] = new float[iteracoes];
        sensitivityIteracao[f] = new float[iteracoes];
        ppvIteracao[f] = new float[iteracoes];
        accpcIteracao[f] = new float[iteracoes];
        fscoreIteracao[f] = new float[iteracoes];
    }
    
    int **confMatrixMedia;
    
    confMatrixMedia = new int*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        confMatrixMedia[f] = new int[quantClasses];
    }
    
    for(int i=0; i< quantClasses; i++)
        for(int j=0; j< quantClasses; j++)
            confMatrixMedia[i][j]=0;
    
    int objTotal = 0;
    
    //DireitosAutoriais();
    
    int argc;
    char **argv;
    
    for(int it = 0 ; it< iteracoes; it++)
    {
        char *name_file_training_dat= "training.dat";
        char *name_file_testing_dat= "testing.dat";
        
        //char *name_file_results_txt = "results";//.txt eh colocado depois
        
        sprintf(file_results_txt, "%s/%s", directorySrc, name_file_results_txt);
        
        sprintf(file_in_txt, "%s/%s", directorySrc, name_file_in_txt);
        
        sprintf(file_training_dat, "%s/%s", directorySrc, name_file_training_dat);
        
        sprintf(file_testing_dat, "%s/%s", directorySrc, name_file_testing_dat);
        
        
        argv = new char*[7];
        
        for(int f = 0; f < 7; f++)
        {
            argv[f] = new char[500];
        }
        
        argc = 7;
        
        sprintf(argv[1],"%s",file_in_txt);
        sprintf(argv[2],"%s",file_training_dat);
        sprintf(argv[3],"%s",file_testing_dat);
        
        //int qa = quantidadeAtributos/25;
        
        //-------------------------------------SPLIT DO BANCO DE DADOS------------------------------------
        splitDataBaseOpenCvMethods(argc, argv, percentTreino ,quantidadeAtributos, quantClasses, normalize);
        
        //quantidadeAtributos=quantidadeAtributos/25;
        
        //-----------------------CARREGANDO BANCO DE DADOS PARA TREINO E TESTE----------------------------
        
        //matrix to hold the training sample
        cv::Mat training_set(TRAINING_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the labels of each taining sample
        cv::Mat training_set_classifications(TRAINING_SAMPLES, 1, CV_32FC1);
        //cv::Mat training_set_classifications(TRAINING_SAMPLES, quantClasses, CV_32F);
        //matric to hold the test samples
        cv::Mat test_set(TEST_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the test labels.
        cv::Mat test_set_classifications(TEST_SAMPLES,1,CV_32FC1);
        //cv::Mat test_set_classifications(TEST_SAMPLES,quantClasses,CV_32F);
        
        //
        cv::Mat classificationResult(1, 1, CV_32FC1);
        //cv::Mat classificationResult(1, CLASSES, CV_32FC1);
        //load the training and test data sets.
        read_dataset(argv[2], training_set, training_set_classifications, TRAINING_SAMPLES, quantidadeAtributos);
        read_dataset(argv[3], test_set, test_set_classifications, TEST_SAMPLES, quantidadeAtributos);
        
        
        
        //------------------------------------------------------------
        //-----------------Bayes TREINAMENTO----------------------------
        //------------------------------------------------------------
        
        
        CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier();
        
        struct timeval tic, toc;
        
        printf("\nTreinando Bayes...");
        gettimeofday(&tic,NULL);
        
        bayes->train(training_set, training_set_classifications);
        
    
        
        gettimeofday(&toc,NULL);
        printf("OK!\n\n");
        
        timeTrain[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        printf("\n\nTraining time: %f seconds\n\n", timeTrain[it]);
        
        //------------------------------------------------------------
        //-------------------Bayes CLASSIFICACAO----------------------
        //------------------------------------------------------------
        
        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;
        
        //classification matrix gives the count of classes to which the samples were classified.
        //int classification_matrix[CLASSES][CLASSES]={{}};
        
        int numClassses = (int)CLASSES;
        int ** classification_matrix;
        classification_matrix = new int*[numClassses];
        
        for(int f = 0; f < numClassses; f++)
        {
            classification_matrix[f] = new int[numClassses];
        }
        
        for(int row=0;row<CLASSES;row++)
            for(int col=0;col<CLASSES;col++)
                classification_matrix[row][col]=0;
        
        gettimeofday(&tic,NULL);
        // for each sample in the test set.
        // for each sample in the test set.
        for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {
            
            // extract the sample
            
            test_sample = test_set.row(tsample);
            
            //SVM.predict(test_sample, classificationResult);
            // float res3 = classificationResult.at<float>(0,0);
            
            int res = (int) (bayes->predict(test_sample));
            
            int test = (int) (test_set_classifications.at<float>(tsample,0));
            
            //printf("Testing Sample %i -> class result (digit %d\t%d)\n", tsample, res, test);
            
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            
            classification_matrix[test][res]++;
            
            if (test!=res)
            {
                // if they differ more than floating point error => wrong class
                
                wrong_class++;
                
            } else {
                
                // otherwise correct
                
                correct_class++;
            }
            
        }
        
        gettimeofday(&toc,NULL);
        
        timeTest[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        timeTest[it]=timeTest[it]/((float)(TEST_SAMPLES));
        printf("\n\nTest time: %f seconds\n\n", timeTest[it]);
        
        //------------------------------------------------------------
        //---------------------------SVM------------------------------
        //------------------------------------------------------------
        
        printf( "\nImprimindo Matriz de consufsao...");
        FILE *fpMatrizConfusao = NULL;
        
        //sprintf(file_results_txt, "%s/%s%f_%d.txt", directorySrc, name_file_results_txt, acc,it);
        
        if (it==0)
            fpMatrizConfusao = fopen(file_results_txt,"w");
        else
            fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "Metodo - Bayes \nBase - %s\n", name_file_in_txt);
        
        switch (config) {
            case 1:
                fprintf(fpMatrizConfusao, "Config - 1\n");
                break;
            case 2:
                fprintf(fpMatrizConfusao, "Config - 2\n");
                break;
            case 3:
                fprintf(fpMatrizConfusao, "Config - 3\n");
                break;
            case 4:
                fprintf(fpMatrizConfusao, "Config - 4\n");
                break;
            default:
                break;
        }
        
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        fprintf(fpMatrizConfusao, "\n-------------------------Iteracao %d------------------------", it+1);
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        
        
        fprintf(fpMatrizConfusao, "\n\nTempos\n\nTrain\t-\t%f\nTeste 1 amostra\t-\t%f",timeTrain[it],timeTest[it]);
        
        fprintf(fpMatrizConfusao, "\n\n\nMatriz de confusao\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        int totalObj = 0;
        float totalObjetos;
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%d\t",classification_matrix[row][col]);
                totalObj+=classification_matrix[row][col];
                confMatrixMedia[row][col]+=classification_matrix[row][col];
                objTotal+=totalObj;
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        totalObjetos = (float)totalObj;
        printf( "Ok!");
        
        //----------------------------------------------
        //----------------------------------------------
        //----------------------------------------------
        
        /*printf( "\nImprimindo Matriz de confusao percentual...");
        
        fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%.2f\t", (float)classification_matrix[row][col]*100/totalObjetos);
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        
        printf( "Ok!");*/
        
        sprintf(argv[1],"%s",file_results_txt);
        
        opfmedidamodificadas(classification_matrix, CLASSES, argv, it);
    }
    
    
    FILE *fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------RESUMO GERAL------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    
    fprintf(fpMatrizConfusao, "\n\nMatriz de confusao Media\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    
    /*printf( "\nImprimindo Matriz de confusao percentual...");
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", (float)confMatrixMedia[row][col]*100/objTotal);
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    printf( "Ok!");*/
    
    sprintf(argv[1],"%s",file_results_txt);
    
    
    
    //---------------------------------------------------------
    //----------------PRINT METRICAS ACC, HM, SE,SP------------
    //---------------------------------------------------------
    
    printMetricasPrincipais(CLASSES, argv, iteracoes, 3, config, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
    //printMetricasGerais(CLASSES, argv, iteracoes, 3, kernell);
    
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%----------MATRIZ DE CONFUSAO LATEX MEDIA NUMERICA-----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Numerical average confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCNumericoM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 5, config);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    
    
    /*fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%---------MATRIZ DE CONFUSAO LATEX MEDIA PERCENTUAL----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Average percentage confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCPercentualM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 5, config);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", (float)confMatrixMedia[row][col]*100/objTotal);
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");*/
    
    
    fclose(fpMatrizConfusao);
    
    delete [] timeTrain;
    delete [] timeTest;
    
    delete [] specificityTotal;
    delete [] sensitivityTotal;
    delete [] accpcTotal;
    delete [] fscoreTotal;
    delete [] ppvTotal;
    
    for(int f = 0; f < quantClasses; f++)
    {
        delete [] specificityIteracao[f];
        delete [] sensitivityIteracao[f];
        delete [] ppvIteracao[f];
        delete [] accpcIteracao[f];
        delete [] fscoreIteracao[f];
    }
    
    delete [] specificityIteracao;
    delete [] sensitivityIteracao;
    delete [] ppvIteracao;
    delete [] accpcIteracao;
    delete [] fscoreIteracao;
    
    printf("\n\nAcabou o Bayes %d\n\n", config);
    
    return 0;
}

int applyML_PedrosaMLP(char *directorySrc, char *name_file_in_txt, char *name_file_results_txt, float percentTreino, int quantidadeAtributos, int quantClasses, int iteracoes, int config, int normalize, float &accAve, float &accStd, float &fscoreAve, float &fscoreStd, float &trainAve, float &trainStd, float &testAve, float &testStd )
{
    timeTrain = new float [iteracoes];
    timeTest = new float [iteracoes];
    
    CLASSES=quantClasses;
    
    specificityTotal = new float[iteracoes];
    sensitivityTotal = new float[iteracoes];
    accpcTotal = new float[iteracoes];
    fscoreTotal = new float[iteracoes];
    ppvTotal = new float[iteracoes];
    
    specificityIteracao = new float*[quantClasses];
    sensitivityIteracao = new float*[quantClasses];
    ppvIteracao = new float*[quantClasses];
    accpcIteracao = new float*[quantClasses];
    fscoreIteracao = new float*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        specificityIteracao[f] = new float[iteracoes];
        sensitivityIteracao[f] = new float[iteracoes];
        ppvIteracao[f] = new float[iteracoes];
        accpcIteracao[f] = new float[iteracoes];
        fscoreIteracao[f] = new float[iteracoes];
    }
    
    int **confMatrixMedia;
    
    confMatrixMedia = new int*[quantClasses];
    
    for(int f = 0; f < quantClasses; f++)
    {
        confMatrixMedia[f] = new int[quantClasses];
    }
    
    for(int i=0; i< quantClasses; i++)
        for(int j=0; j< quantClasses; j++)
            confMatrixMedia[i][j]=0;
    
    int objTotal = 0;
    
    //DireitosAutoriais();
    
    int argc;
    char **argv;
    
    for(int it = 0 ; it< iteracoes; it++)
    {
        char *name_file_training_dat= "training.dat";
        char *name_file_testing_dat= "testing.dat";
        
        //char *name_file_results_txt = "param.xml";//.txt eh colocado depois
        
        sprintf(file_results_txt, "%s/%s", directorySrc, name_file_results_txt);
        
        sprintf(file_in_txt, "%s/%s", directorySrc, name_file_in_txt);
        
        sprintf(file_training_dat, "%s/%s", directorySrc, name_file_training_dat);
        
        sprintf(file_testing_dat, "%s/%s", directorySrc, name_file_testing_dat);
        
        sprintf(file_classifier_opf, "%s/param.xml", directorySrc);
        
        
        
        argv = new char*[7];
        
        for(int f = 0; f < 7; f++)
        {
            argv[f] = new char[500];
        }
        
        argc = 7;
        
        sprintf(argv[1],"%s",file_in_txt);
        sprintf(argv[2],"%s",file_training_dat);
        sprintf(argv[3],"%s",file_testing_dat);
        
        //-------------------------------------SPLIT DO BANCO DE DADOS------------------------------------
        splitDataBaseOpenCvMethods(argc, argv, percentTreino ,quantidadeAtributos, quantClasses, normalize);
        
        
        //-----------------------CARREGANDO BANCO DE DADOS PARA TREINO E TESTE----------------------------
        
        //matrix to hold the training sample
        cv::Mat training_set(TRAINING_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the labels of each taining sample
        //cv::Mat training_set_classifications(TRAINING_SAMPLES, 1, CV_32FC1);
        cv::Mat training_set_classifications(TRAINING_SAMPLES, quantClasses, CV_32F);
        //matric to hold the test samples
        cv::Mat test_set(TEST_SAMPLES,quantidadeAtributos,CV_32FC1);
        //matrix to hold the test labels.
        //cv::Mat test_set_classifications(TEST_SAMPLES,1,CV_32FC1);
        cv::Mat test_set_classifications(TEST_SAMPLES,quantClasses,CV_32F);
        
        //
        //cv::Mat classificationResult(1, 1, CV_32FC1);
        cv::Mat classificationResult(1, CLASSES, CV_32FC1);
        //load the training and test data sets.
        read_dataset_MLP(argv[2], training_set, training_set_classifications, TRAINING_SAMPLES, quantidadeAtributos);
        read_dataset_MLP(argv[3], test_set, test_set_classifications, TEST_SAMPLES, quantidadeAtributos);
        
        
        
        //------------------------------------------------------------
        //-----------------MLP TREINAMENTO----------------------------
        //------------------------------------------------------------
        cv::Mat layers(3,1,CV_32S);
        
        int numHiddenLayer;
        
        switch (config)
        {
            case 1:
                numHiddenLayer = (quantClasses+quantidadeAtributos)/5;
                layers.at<int>(0,0) = quantidadeAtributos;//input layer
                layers.at<int>(1,0) = numHiddenLayer;//hidden layer
                layers.at<int>(2,0) = quantClasses;//output layer
                break;
            case 2:
                numHiddenLayer = (quantClasses+quantidadeAtributos)/4;
                layers.at<int>(0,0) = quantidadeAtributos;//input layer
                layers.at<int>(1,0) = numHiddenLayer;//hidden layer
                layers.at<int>(2,0) = quantClasses;//output layer
                break;
            default:
                numHiddenLayer = (quantClasses+quantidadeAtributos)/2;
                layers.at<int>(0,0) = quantidadeAtributos;//input layer
                layers.at<int>(1,0) = numHiddenLayer;//hidden layer
                layers.at<int>(2,0) = quantClasses;//output layer
                break;
        }
        
       
        
        //create the neural network.
        //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
        CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1); //3 conf
        
        CvANN_MLP_TrainParams params(
                                     
                                     // terminate the training after either 1000
                                     // iterations or a very small change in the
                                     // network wieghts below the specified value
                                     cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
                                     // use backpropogation for training
                                     CvANN_MLP_TrainParams::BACKPROP, //4 conf
                                     // co-efficents for backpropogation training
                                     // recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
                                     0.1,
                                     0.1);
        
        // train the neural network (using training data)
        struct timeval tic, toc;
        
        printf( "\nUsing training dataset\n");
        gettimeofday(&tic,NULL);
        int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
        gettimeofday(&toc,NULL);
        printf( "Training iterations: %i\n\n", iterations);
        
        
        // Save the model generated into an xml file.
        CvFileStorage* storage = cvOpenFileStorage(file_classifier_opf, 0, CV_STORAGE_WRITE );
        nnetwork.write(storage,"DigitOCR");
        cvReleaseFileStorage(&storage);
        
        
        timeTrain[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        printf("\n\nTraining time: %f seconds\n\n", timeTrain[it]);
        
        //------------------------------------------------------------
        //-----------------MLP CLASSIFICACAO--------------------------
        //------------------------------------------------------------
        
        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;
        
        //classification matrix gives the count of classes to which the samples were classified.
        //int classification_matrix[CLASSES][CLASSES]={{}};
        
        int numClassses = (int)CLASSES;
        int ** classification_matrix;
        classification_matrix = new int*[numClassses];
        
        for(int f = 0; f < numClassses; f++)
        {
            classification_matrix[f] = new int[numClassses];
        }
        
        for(int row=0;row<CLASSES;row++)
            for(int col=0;col<CLASSES;col++)
                classification_matrix[row][col]=0;
        
        
        // for each sample in the test set.
        for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {
            
            // extract the sample
            
            test_sample = test_set.row(tsample);
            
            //try to predict its class
            
            nnetwork.predict(test_sample, classificationResult);
            /*The classification result matrix holds weightage  of each class.
             we take the class with the highest weightage as the resultant class */
            
            // find the class with maximum weightage.
            int maxIndex = 0;
            float value=0.0f;
            float maxValue=classificationResult.at<float>(0,0);
            for(int index=1;index<CLASSES;index++)
            {
                value = classificationResult.at<float>(0,index);
                if(value>maxValue)
                {   maxValue = value;
                    maxIndex=index;
                    
                }
            }
            
            
            printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
            
            
            
            gettimeofday(&tic,NULL);
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            if (test_set_classifications.at<float>(tsample, maxIndex)!=1.0f)
            {
                // if they differ more than floating point error => wrong class
                
                wrong_class++;
                
                //find the actual label 'class_index'
                for(int class_index=0;class_index<CLASSES;class_index++)
                {
                    if(test_set_classifications.at<float>(tsample, class_index)==1.0f)
                    {
                        
                        classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
                        break;
                    }
                }
                
            } else {
                
                // otherwise correct
                
                correct_class++;
                classification_matrix[maxIndex][maxIndex]++;
            }
        }
        
        gettimeofday(&toc,NULL);
        timeTest[it] = ((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001)/1000.0;
        timeTest[it]=timeTest[it]/((float)(TEST_SAMPLES));
        printf("\n\nTest time: %f seconds\n\n", timeTest[it]);
        
        //------------------------------------------------------------
        //---------------------------MLP------------------------------
        //------------------------------------------------------------
        
        printf( "\nImprimindo Matriz de consufsao...");
        FILE *fpMatrizConfusao = NULL;
        
        //sprintf(file_results_txt, "%s/%s%f_%d.txt", directorySrc, name_file_results_txt, acc,it);
        
        if (it==0)
            fpMatrizConfusao = fopen(file_results_txt,"w");
        else
            fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "Metodo - MLP \nBase - %s\n", name_file_in_txt);
        
        fprintf(fpMatrizConfusao, "Configuracao %d- %d/%d/%d\n", config, quantidadeAtributos,numHiddenLayer,quantClasses);
        
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        fprintf(fpMatrizConfusao, "\n-------------------------Iteracao %d------------------------", it+1);
        fprintf(fpMatrizConfusao, "\n------------------------------------------------------------");
        
        
        fprintf(fpMatrizConfusao, "\n\nTempos\n\nTrain\t-\t%f\nTeste 1 amostra\t-\t%f",timeTrain[it],timeTest[it]);
        
        fprintf(fpMatrizConfusao, "\n\n\nMatriz de confusao\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        int totalObj = 0;
        float totalObjetos;
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%d\t",classification_matrix[row][col]);
                totalObj+=classification_matrix[row][col];
                confMatrixMedia[row][col]+=classification_matrix[row][col];
                objTotal+=totalObj;
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        totalObjetos = (float)totalObj;
        printf( "Ok!");
        
        //----------------------------------------------
        //----------------------------------------------
        //----------------------------------------------
        
        /*printf( "\nImprimindo Matriz de confusao percentual...");
        
        fpMatrizConfusao = fopen(file_results_txt,"a");
        
        fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
        
        fprintf(fpMatrizConfusao,"\t");
        for (int i = 0; i < CLASSES; i++)
        {
            fprintf(fpMatrizConfusao, "%d\t", i);
        }
        fprintf(fpMatrizConfusao,"\n");
        
        
        for(int row=0;row<CLASSES;row++)
        {
            fprintf(fpMatrizConfusao,"%d\t",row);
            for(int col=0;col<CLASSES;col++)
            {
                fprintf(fpMatrizConfusao,"%.2f\t", (float)classification_matrix[row][col]*100/totalObjetos);
            }
            fprintf(fpMatrizConfusao,"\n");
        }
        fprintf(fpMatrizConfusao,"\n");
        fprintf(fpMatrizConfusao,"\n");
        fclose(fpMatrizConfusao);
        
        printf( "Ok!");*/
        
        sprintf(argv[1],"%s",file_results_txt);
        
        opfmedidamodificadas(classification_matrix, CLASSES, argv, it);
    }
    
    
    FILE *fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------RESUMO GERAL------------------------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    
    fprintf(fpMatrizConfusao, "\n\nMatriz de confusao Media\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    
   /* printf( "\nImprimindo Matriz de confusao percentual...");
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\nMatriz de confusao percentual\n\n");
    
    fprintf(fpMatrizConfusao,"\t");
    for (int i = 0; i < CLASSES; i++)
    {
        fprintf(fpMatrizConfusao, "%d\t", i);
    }
    fprintf(fpMatrizConfusao,"\n");
    
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.2f\t", (float)confMatrixMedia[row][col]*100/objTotal);
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    fclose(fpMatrizConfusao);
    
    printf( "Ok!");*/
    
    sprintf(argv[1],"%s",file_results_txt);
    
    
    
    //---------------------------------------------------------
    //----------------PRINT METRICAS ACC, HM, SE,SP------------
    //---------------------------------------------------------
    
    printMetricasPrincipais(CLASSES, argv, iteracoes, 4, config, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
    //printMetricasGerais(CLASSES, argv, iteracoes, 4, config);
    
    //---------------------------------------------------------
    //---------------------------------------------------------
    //---------------------------------------------------------
    
    fpMatrizConfusao = fopen(file_results_txt,"a");
    
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%----------MATRIZ DE CONFUSAO LATEX MEDIA NUMERICA-----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Numerical average confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCNumericoM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 4, config);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", ((float)confMatrixMedia[row][col])/((float)iteracoes));
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");
    
    
    /*fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------");
    fprintf(fpMatrizConfusao, "\n%%---------MATRIZ DE CONFUSAO LATEX MEDIA PERCENTUAL----------");
    fprintf(fpMatrizConfusao, "\n%%------------------------------------------------------------\n\n\n\n");
    
    fprintf(fpMatrizConfusao, "\\begin{table}[!hbt] \n\t \\centering \n\t\t \\caption{\\it Average percentage confusion matrix for database of ... obtained by ... method.}\n\t\\label{tabMCPercentualM%dC%d}\t\n{\\normalsize\n\\begin{tabular}{", 4, config);
    
    for (int i = 0; i < CLASSES; i++)
        fprintf(fpMatrizConfusao, "c");
    fprintf(fpMatrizConfusao, "c");
    
    fprintf(fpMatrizConfusao, "}\n\\hline\nCLASS\t&\t");
    
    for (int i = 0; i < CLASSES; i++)
    {
        if (i==CLASSES-1)
            fprintf(fpMatrizConfusao, "%d\t\\\\", i+1);
        else
            fprintf(fpMatrizConfusao, "%d\t&\t", i+1);
    }
    
    fprintf(fpMatrizConfusao, " \\hline \n");
    
    for(int row=0;row<CLASSES;row++)
    {
        fprintf(fpMatrizConfusao,"%d\t&\t",row);
        for(int col=0;col<CLASSES;col++)
        {
            fprintf(fpMatrizConfusao,"%.1f\t", (float)confMatrixMedia[row][col]*100/objTotal);
            if (col==CLASSES-1)
                fprintf(fpMatrizConfusao,"\\\\");
            else
                fprintf(fpMatrizConfusao,"&\t");
        }
        fprintf(fpMatrizConfusao,"\n");
    }
    
    fprintf(fpMatrizConfusao,"\\hline \n\t \\end{tabular}} \n \\end{table} ");
    
    fprintf(fpMatrizConfusao,"\n");
    fprintf(fpMatrizConfusao,"\n");*/
    
    
    fclose(fpMatrizConfusao);
    
    delete [] timeTrain;
    delete [] timeTest;
    
    delete [] specificityTotal;
    delete [] sensitivityTotal;
    delete [] accpcTotal;
    delete [] fscoreTotal;
    delete [] ppvTotal;
    
    for(int f = 0; f < quantClasses; f++)
    {
        delete [] specificityIteracao[f];
        delete [] sensitivityIteracao[f];
        delete [] ppvIteracao[f];
        delete [] accpcIteracao[f];
        delete [] fscoreIteracao[f];
    }
    
    delete [] specificityIteracao;
    delete [] sensitivityIteracao;
    delete [] ppvIteracao;
    delete [] accpcIteracao;
    delete [] fscoreIteracao;
    
    printf("\n\nAcabou o MLP %d\n\n", config);
    
    return 0;
}
