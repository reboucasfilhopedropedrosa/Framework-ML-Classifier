#//
//  MLPedrosa.c
//  OPF_xcode_v2
//
//  Created by Pedro Pedrosa Rebouças Filho on 06/05/15.
//  Copyright (c) 2015 Pedro Pedrosa Rebouças Filho. All rights reserved.
//  Contacts: 'pedrosarf@ifce.edu.br'
//  More informations: http://professorpedrosa.com

#include "MLPedrosa.h"

using namespace std;

char directory [500];

char name_file_in_txt [500];

float percentualTreino = 0.5; //teste = 1 - treino

int quantidadeAtributos = 8;

int quantidadeClasses = 10;

int quantidadeIteracoes = 5;

int OPF_method(int method, int normalizacao)
{
    int distancia = method;
    
    int inicio, termino;
    
    if (distancia == 0)
    {
        inicio = 1;
        termino = 9;
    }
    else
    {
        inicio = distancia;
        termino = distancia;
    }

    
    for ( distancia = inicio; distancia <= termino; distancia++)
    {
        float percent = percentualTreino;
//        for(float percent = 0.1; percent<0.95; percent+=0.1)
//        {
            char name_file_out_txt[500];
            
            char name_file_stat_txt[500];
            //int distancia = 1;
            
            switch (distancia) {
                case 1:
                    sprintf(name_file_out_txt, "resultados_OPF_Euclidean_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_OPF_Euclidean_stat.txt",directory);
                    break;
                case 2:
                    sprintf(name_file_out_txt, "fora_resultados_OPF_Chi-Square_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/fora_resultados_OPF_Chi-Square_stat.txt",directory);
                    break;
                case 3:
                    sprintf(name_file_out_txt, "resultados_OPF_Manhattan_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_OPF_Manhattan_stat.txt",directory);
                    break;
                case 4:
                    sprintf(name_file_out_txt, "resultados_OPF_Canberra_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_OPF_Canberra_stat.txt",directory);
                    break;
                case 5:
                    sprintf(name_file_out_txt, "fora_resultados_OPF_Squared_Chord_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/fora_resultados_OPF_Squared_Chord_stat.txt",directory);
                    break;
                case 6:
                    sprintf(name_file_out_txt, "resultados_OPF_Squared_Chi-Squared_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_OPF_Squared_Chi-Squared_stat.txt",directory);
                    break;
                case 7:
                    sprintf(name_file_out_txt, "fora_resultados_OPF_BrayCurtis_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/fora_resultados_OPF_BrayCurtis_stat.txt",directory);
                    break;
                case 8:
                    sprintf(name_file_out_txt, "resultados_OPF_Gaussian_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_OPF_Gaussian_stat.txt",directory);
                    break;
                case 9:
                    sprintf(name_file_out_txt, "fora_resultados_OPF_Mahalanobis_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/fora_resultados_OPF_Mahalanobis_stat.txt",directory);
                    break;
                default:
                    break;
            }
            
            int normalize = normalizacao; //Distance normalization? 1- yes 0 - no
            
            float accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd;
            
            applyML_PedrosaOPF(directory, name_file_in_txt, name_file_out_txt, percent, quantidadeAtributos, quantidadeClasses, quantidadeIteracoes, distancia, normalize, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            
            
            FILE *stat;
            
            if(trainAve/4<0.00000001)
                trainAve=0.00000004;
            
            if(trainStd/4<0.00000001)
                trainStd=0.00000004;
            
            if(testAve/4<0.00000001)
                testAve=0.00000004;
            
            if(testStd/4<0.00000001)
                testStd=0.00000004;
            
            if(percent > 0.05 && percent <0.15)
            {
                stat= fopen(name_file_stat_txt, "w");
                fprintf(stat,"PE\tAccAv\tAccSD\tFsAv\tFsSD\ttrAv\t\ttrSD\t\tteAv\t\tteSD\n");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve/4, trainStd/4, testAve/4, testStd/4);
            }
            else
            {
                stat= fopen(name_file_stat_txt, "a");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve/4, trainStd/4, testAve/4, testStd/4);
            }
            fclose(stat);
            
        }
//    }
   
    // system("pause");
    return 0;
}

int KNN_method(int method, int normalizacao)
{
    int k = method;
    
    int inicio, termino;
    
    if (k == 0)
    {
        inicio = 1;
        termino = 5;
    }
    else
    {
        inicio = k;
        termino = k;
    }
    
    for (k = inicio; k <= termino; k+=2)
    {
        float percent = percentualTreino;
        
//        for(float percent = 0.1; percent<0.95; percent+=0.1)
//        {
        
            char name_file_out_txt[500];
            
            char name_file_stat_txt[500];
            
            sprintf(name_file_out_txt, "resultados_KNN%d_%d.txt",k,((int)(100*percent)));
            
            sprintf(name_file_stat_txt, "%s/resultados_KNN_stat.txt",directory);
            
            int normalize = normalizacao; //Distance normalization? 1- yes 0 - no
            
            float accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd;
            
            applyML_PedrosaKNN(directory, name_file_in_txt, name_file_out_txt, percent, quantidadeAtributos, quantidadeClasses, quantidadeIteracoes, k, normalize, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            
            FILE *stat;
            
            if(percent > 0.05 && percent <0.15)
            {
                stat= fopen(name_file_stat_txt, "w");
                fprintf(stat,"PE\tAccAv\tAccSD\tFsAv\tFsSD\ttrAv\t\ttrSD\t\tteAv\t\tteSD\n");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            else
            {
                stat= fopen(name_file_stat_txt, "a");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            fclose(stat);
            
            
        }
    //}

    // system("pause");
    return 0;
}

int SVM_method(int method, int SVM_auto, int normalizacao)
{
    int kernell = method;
    
    int inicio, termino;
    
    if (kernell == 0)
    {
        inicio = 1;
        termino = 4;
    }
    else
    {
        inicio = kernell;
        termino = kernell;
    }
    for (kernell = inicio; kernell <= termino; kernell++)
    {
        float percent = percentualTreino;
//        for(float percent = 0.1; percent<0.95; percent+=0.1)
//        {
            char name_file_out_txt[500];
            char name_file_stat_txt[500];
            
            switch (kernell) {
                case 1:
                    sprintf(name_file_out_txt, "resultados_SVM_Linear_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_SVM_Linear_stat.txt",directory);
                    break;
                case 2:
                    sprintf(name_file_out_txt, "resultados_SVM_RBF_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_SVM_RBF_stat.txt",directory);
                    break;
                case 3:
                    sprintf(name_file_out_txt, "resultados_SVM_Poly_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_SVM_Poly_stat.txt",directory);
                    break;
                case 4:
                    sprintf(name_file_out_txt, "resultados_SVM_Sigmoid_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_SVM_Sigmoid_stat.txt",directory);
                    break;
                default:
                    break;
            }
            
            int normalize = normalizacao; //Distance normalization? 1- yes 0 - no
            
            int parametrosAutomaticos = SVM_auto; //1 - automaticos, 0 fixo
            
            float accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd;
            
            applyML_PedrosaSVM(directory, name_file_in_txt, name_file_out_txt, percent, quantidadeAtributos, quantidadeClasses, quantidadeIteracoes, kernell, normalize, parametrosAutomaticos, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            
            FILE *stat;
            
            if(percent > 0.05 && percent <0.15)
            {
                stat= fopen(name_file_stat_txt, "w");
                fprintf(stat,"PE\tAccAv\tAccSD\tFsAv\tFsSD\ttrAv\t\ttrSD\t\tteAv\t\tteSD\n");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            else
            {
                stat= fopen(name_file_stat_txt, "a");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            fclose(stat);
        }
    //}
    
    // system("pause");
    return 0;
}

int MLP_method(int method, int normalizacao)
{
    int config = method;
    
    int inicio, termino;
    
    if (config == 0)
    {
        inicio = 1;
        termino = 2;
    }
    else
    {
        inicio = config;
        termino = config;
    }
    
    for (config = inicio; config <= termino; config++)
    {
        float percent = percentualTreino;
//            for(float percent = 0.1; percent<0.95; percent+=0.1)
//            {
        
            char name_file_out_txt[500];
            char name_file_stat_txt[500];
                
            switch (config) {
                case 1:
                    sprintf(name_file_out_txt, "resultados_MLP_config1_%d.txt",((int)(100*percent)));
                    
                    sprintf(name_file_stat_txt, "%s/resultados_ANN_stat.txt",directory);
                    break;
                case 2:
                    sprintf(name_file_out_txt, "resultados_MLP_config2_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_ANN_stat.txt",directory);
                    break;
                default:
                    sprintf(name_file_out_txt, "resultados_MLP_config1_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_ANN_stat.txt",directory);
                    break;
            }
            
            
            
            int normalize = normalizacao; //Distance normalization? 1- yes 0 - no
            
                float accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd;
            
            applyML_PedrosaMLP(directory, name_file_in_txt, name_file_out_txt, percent, quantidadeAtributos, quantidadeClasses, quantidadeIteracoes, config, normalize, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            
                FILE *stat;
                
                if(percent > 0.05 && percent <0.15)
                {
                    stat= fopen(name_file_stat_txt, "w");
                    fprintf(stat,"PE\tAccAv\tAccSD\tFsAv\tFsSD\ttrAv\t\ttrSD\t\tteAv\t\tteSD\n");
                    fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
                }
                else
                {
                    stat= fopen(name_file_stat_txt, "a");
                    fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
                }
                fclose(stat);
        }
    //}

    // system("pause");
    return 0;
}

int bayes_normal_method(int normalizacao)
{
    int config = 1;
    
    int inicio, termino;
    
    if (config == 0)
    {
        inicio = 1;
        termino = 1;
    }
    else
    {
        inicio = config;
        termino = config;
    }
    

    for (config = inicio; config <= termino; config++)
    {
        float percent = percentualTreino;
        
//        for(float percent = 0.1; percent<0.95; percent+=0.1)
//        {
                char name_file_out_txt[500];
            
            char name_file_stat_txt[500];
            
            switch (config) {
                case 1:
                    sprintf(name_file_out_txt, "resultados_bayes_Normal_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_bayes_Normal_stat.txt",directory);
                    break;
                default:
                    sprintf(name_file_out_txt, "resultados_bayes_Normal_%d.txt",((int)(100*percent)));
                    sprintf(name_file_stat_txt, "%s/resultados_bayes_Normal_stat.txt",directory);
                    break;
            }
            
            
            
            int normalize = normalizacao; //Distance normalization? 1- yes 0 - no
            
            float accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd;
            
            applyML_PedrosaBayes(directory, name_file_in_txt, name_file_out_txt, percent, quantidadeAtributos, quantidadeClasses, quantidadeIteracoes, config, normalize, accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            
            FILE *stat;
            
            if(percent > 0.05 && percent <0.15)
            {
                stat= fopen(name_file_stat_txt, "w");
                fprintf(stat,"PE\tAccAv\tAccSD\tFsAv\tFsSD\ttrAv\t\ttrSD\t\tteAv\t\tteSD\n");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            else
            {
                stat= fopen(name_file_stat_txt, "a");
                fprintf(stat,"%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.8fe\t%.8f\t%.8f\t%.8f\n", percent,accAve, accStd, fscoreAve, fscoreStd, trainAve, trainStd, testAve, testStd);
            }
            fclose(stat);
        }
    //}
    // system("pause");
    return 0;
}

int help_FormatoDoArquivo()
{
    printf("\n\nO arquivo de entrada fornecido na variavel 'name_file_in_txt', deve obedecer um formato especifico:\n-Cada linha deve ser um objeto supervisionado de exemplo do seu banco de dados. \n-Na linha deve estar em sequencia os atributos, separados por virgula.\n-No final da linha, deve estar o rotulo (label) da classe que objeto representa.\n\nObs.1: Se nao estiver neste formato, o codigo apresentara erros.\nObs.2:Lembre que as casas decimais de cada atributos sao com ponto, pois a linguacem C obedece o sistema ingles.\nObs.3:A classe deve iniciar com número 0, entao se existir 5 classes, entao existirao as classes 0, 1, 2, 3 e 4.");
    
    printf("\n\nExemplo: Banco de Dados com 5 atributos e 3 classes, deve estar assim:\n\n");
    
    printf("0.2, 1.4, 3.5, 3.2, 0.1, 0\n");
    printf("1.2, 0.4, 2.5, 4.2, 0.1, 1\n");
    printf("0.2, 2.4, 2, 4.2, 0.4, 1\n");
    printf("2.2, 5, 3.5, 4.2, 0.1, 0\n");
    printf("1, 1.4, 3.5, 4, 0.2, 2\n");
    printf("0.8, 1.4, 3.5, 4.2, 0.1, 2\n");
    printf("0.5, 1.4, 2.5, 3.2, 2, 0\n");
    
    //printf("\n\n\n\nQuaisquer duvidas, enviar email para 'pedrosarf@ifce.edu.br'.\n\n");
    
    printf("\nCreated by Pedro Pedrosa Rebouças Filho, Version 3.0 on 15/05/15.");
    printf("\nCopyright (c) 2015 Pedro Pedrosa Rebouças Filho. All rights reserved.");
    printf("\nContacts: 'pedrosarf@ifce.edu.br'");
    printf("\nMore informations: http://professorpedrosa.com\n\n\n");
    
    getchar();
    
    return 1;
}

int changeFormatFile()
{
    srand(time(NULL));
    
    char input[500];
    char output1[500];
    char output2[500];
    int quantClasses = 5;
    int quantidadeAtributos = 52;
    
    int quantidadeLinhas=0;
    
    FILE *fpIn = NULL,*fpOut1 = NULL;
    int n, ndata, nclasses, i,j, id,label,cont=0, numLine;
    float aux;
    
    //sprintf(output1, "/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_Expert_System/DB_Edgard/db_raw.txt");
    
    sprintf(output1, "/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_Expert_System/DB_Edgard/db_estatistica.txt");
    
    fpOut1 = fopen(output1,"w");
    
    vector <float*> vetAtt;
    
    int *vetCont = new int[quantClasses];
    
    for (i=0; i<quantClasses; i++)
        vetCont[i]=0;
    
    for (int name = 0; name < quantClasses; name++)
    {
        //sprintf(input, "/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_Expert_System/DB_Edgard/classe%d.txt",name);
        
        sprintf(input, "/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_Expert_System/DB_Edgard/estatistica%d.txt",name);
        
        fpIn = fopen(input,"r");
        id = name;
        float *vet = new float[quantidadeAtributos];
        
        printf("\nArquivo %d\n\n", name);
        
        quantidadeLinhas = 0;
        
        do      /* Enquanto nao chegar ao final do arquivo */
        {
            //fscanf(fpIn,"%d", &numLine);
            
            //fscanf(fpIn,"%d", &id);
            
            for (i=0; i<quantidadeAtributos; i++)
            {
                fscanf(fpIn,"%f", &vet[i]);
                //fprintf(fpOut,"%f,", vet[i]);
                printf("%f, ", vet[i]);
            }
            //fprintf(fpOut,"%d\n", id);
            
            printf("\n\n");
            
            vetAtt.push_back(vet);
            
            //vetCont[id]++;
            
            quantidadeLinhas++;
            
            //cont++;
        }while(getc(fpIn) != EOF);
        
        fclose(fpIn);
        
        printf("Linha - %d\n\n", quantidadeLinhas);
        
        //--------------------------------------------------------
        cont = 0;
        for (j=0; j<quantidadeAtributos; j++)
        {
            for (i=0; i<quantidadeLinhas; i++)
            {
                //vetAtt.at(i)[j];
                
                fprintf(fpOut1,"%f,", vetAtt.at(i)[j]);
                printf("%f, ", vetAtt.at(i)[j]);
            }
            
            fprintf(fpOut1,"%d\n", name);
            printf("%d\n", name);
            cont++;
        }
        
        vetCont[name]=cont;
        
        vetAtt.clear();
        
    }
    
    for (i=0; i<quantClasses; i++)
        printf("\n%d - %d",i,vetCont[i]);
    
    fclose(fpOut1);
    
    //delete [] vet;
    //delete [] vetCont;
    
    
    return 0;
}



int main(int argc, char **argv){
    //-----------------------------------------------------
    //------------------Change File In---------------------
    //-----------------------------------------------------
    // help_FormatoDoArquivo();
    /*changeFormatFile();
    
    return 1; */
    
//-----------------------------------------------------
//----------------------Setup In-----------------------
//-----------------------------------------------------
//    for( int pt = 0; pt<=50; pt+=5)//20
//    {
        //  raw
        
//         if(pt<10)
//         {
//          sprintf(directory,"/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_ESWA/XCode/pca_db_raw_0%d",pt);
//          sprintf(name_file_in_txt, "pca_db_raw_0%d.txt",pt);
//         }
//         else
//         {
//             sprintf(directory,"/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_ESWA/XCode/pca_db_raw_%d",pt);
//             sprintf(name_file_in_txt, "pca_db_raw_%d.txt",pt);
//         }
    
//    sprintf(directory,"/Users/pedro/Documents/Laysa_base/base - Moodle_Forum_Status_class");
//    sprintf(name_file_in_txt, "base - Moodle_Forum_Status_class.csv");
//    quantidadeAtributos = 23;
    
	sprintf(directory,"C:\\Users\\well\\Desktop\\base - Status_class");
    sprintf(name_file_in_txt, "base - Status_class.csv");
    quantidadeAtributos = 96;
            
    
        //  estatisticos
        //  sprintf(directory,"/Users/pedropedrosa/Dropbox/Producao Cientifica/Artigos/Artigos em elaboração/2017_Revista_Edgard_Expert_System/db_estatistica");
        //  sprintf(name_file_in_txt, "db_estatistica.txt");
        //  quantidadeAtributos = 3;
    

    
    
        
        //-----------------------------------------------------
        //------------------Configuracao-----------------------
        //-----------------------------------------------------
        
        int normalize = 1; //Distance normalization? 1- yes 0 - no
        
        percentualTreino = 0.5; //teste = 1 - treino
        
        quantidadeClasses = 4;
        
        quantidadeIteracoes = 5;
        
        //-----------------------------------------------------
        //-----------------------Bayes-------------------------
        //-----------------------------------------------------
        
        //Passo unico: Descomente a funcao bayes_normal_method para rodar o classificador bayes normal
        
        bayes_normal_method(normalize);
        
        //-----------------------------------------------------
        //------------------------OPF--------------------------
        //-----------------------------------------------------
        
        //Passo 1: Escolha UMA das opcoes abaixo descomentando method_OPF
        //Passo 2: Descomente a funcao OPF_method
        
       // int method_OPF = 0; //calcula todas as distancias
        int method_OPF = 1; //Calcula a distancia Euclidean
        //int method_OPF = 2; //Calcula a distancia Chi-Square
        //int method_OPF = 3; //Calcula a distancia Manhattan
        //int method_OPF = 4; //Calcula a distancia Canberra
        //int method_OPF = 5; //Calcula a distancia Squared_Chord
        //int method_OPF = 6; //Calcula a distancia Squared_Chi-Squared
        //int method_OPF = 7; //Calcula a distancia BrayCurtis
        //int method_OPF = 8; //Calcula a distancia Gaussiana
        //int method_OPF = 9; //Calcula a distancia Mahalanobis
    
        OPF_method(method_OPF, normalize);
        
        //-----------------------------------------------------
        //------------------------KNN--------------------------
        //-----------------------------------------------------
        
        //Passo 1: Escolha UMA das opcoes abaixo descomentando method_KNN
        //Passo 2: Descomente a funcao KNN_method
        
        //int method_KNN = 0;//calcula KNN com K 1, 3 e 5
        //int method_KNN = 1; //Calcula KNN com K = 1
        //int method_KNN = 3; //Calcula KNN com K = 3
        int method_KNN = 5; //Calcula KNN com K = 5
        
        KNN_method(method_KNN, normalize);
        
        //-----------------------------------------------------
        //------------------------MLP--------------------------
        //-----------------------------------------------------
        
        //Passo 1: Escolha UMA das opcoes abaixo descomentando method_MLP
        //Passo 2: Descomente a funcao MLP_method
        
        //int method_MLP = 0;//Roda as configuracoes 1 e 2 da MLP
        int method_MLP = 1; //Roda a configuracao 1 da MLP
        //int method_MLP = 2; //Roda a configuracao 2 da MLP
        MLP_method(method_MLP, normalize);
        
        //-----------------------------------------------------
        //------------------------SVM--------------------------
        //-----------------------------------------------------
        
        //Passo 1: Escolha UMA das opcoes abaixo descomentando method_SVM
        //Passo 2: Escolha se quer a configuracao automatica ou manual em SVM_auto
        //Passo 3: Descomente a funcao SVM_method
        
        //int method_SVM = 0;//Roda a SVM com todos os kernells
        int method_SVM = 1; //Roda a SVM com kernell Linear
        //int method_SVM = 2; //Roda a SVM com kernell RBF
        //int method_SVM = 3; //Roda a SVM com kernell Poly
        //int method_SVM = 4; //Roda a SVM com kernell Sigmoid
        
        int SVM_auto = 0; //1 - parametros automaticos, 0 - parametros fixos
        
         /*Se o classificador SVM nao terminar nenhuma iteracao,
         significa que nao encontrou os parametros automaticos.
         Entao, rode o classificador com parametros fixos, 
         mas de preferencia, so nestes casos.
         */
        
        SVM_method(method_SVM, SVM_auto, normalize);
        
        //-----------------------------------------------------
        //-----------------------------------------------------
        //-----------------------------------------------------
//    }
    printf("\n\nAcabou o main.\n\n");
    
    return 0;
}
