#include <stdio.h>
#include <stdbool.h>
#include <math.h>

int main(){    
printf("Hello C Language How? \n");    
return 0;   
}

int test(){
    printf("You did it! \n");
    return 0;
}

float * broadXAS(int CalcSXSCase, int BroadSXSCount[3][40], float BroadSXS[7][3500][3][40], float disord){
    float Gauss[3500][3500]; //This stores the Gauss broadening matrix for each spectrum
    float Lorentz[3500][3500]; //This stores the Lorentz broadening matrix for each spectrum
    float Disorder[3500][3500]; //This stores the disorder broadening matrix for each spectrum
    int c1,c2,c3,c4;

    float width; //This is a dynamic variable the captures the width used in the distribution function
    float position; //This a dynamic variable used to store the centre of the distribution function
    float Pi = 3.14159265; //The Pi constant used for the distribution functions.

    for(c1=0; c1< CalcSXSCase; c1++)
    {
        for(c3 = 0; c3 < BroadSXSCount[0][c1]; c3++) //This cycles through the matrix rows
        {
            width = BroadSXS[4][c3][0][c1]/2.3548; //We extract the variance for the Gaussian Distribution
            position = BroadSXS[0][c3][0][c1]; //We extract the centroid of the Gaussian Distribution
            for(c4=0; c4< BroadSXSCount[0][c1];c4++)
            {
                Gauss[c3][c4] = 1/sqrt(2*Pi*width*width)*exp(-(BroadSXS[0][c4][0][c1]-position)*(BroadSXS[0][c4][0][c1]-position)/2/width/width);
            }

            width = disord/2.3548; //We extract the variance for the Gaussian Distribution
            position = BroadSXS[0][c3][0][c1]; //We extract the centroid of the Gaussian Distribution
            for(c4=0; c4< BroadSXSCount[0][c1];c4++)
            {
                Disorder[c3][c4] = 1/sqrt(2*Pi*width*width)*exp(-(BroadSXS[0][c4][0][c1]-position)*(BroadSXS[0][c4][0][c1]-position)/2/width/width);
            }

            width = BroadSXS[2][c3][0][c1]/2; //We extract the variance for the Lorentz Distribution
            position = BroadSXS[0][c3][0][c1]; //We extract the centroid of the Lorentz Distribution
            for(c4=0; c4< BroadSXSCount[0][c1];c4++)
            {
                Lorentz[c3][c4] = 1/Pi*(width/((BroadSXS[0][c4][0][c1]-position)*(BroadSXS[0][c4][0][c1]-position)+(width*width)));
            }
        }
        for(c3=0; c3<BroadSXSCount[0][c1]; c3++)
        {
            BroadSXS[3][c3][0][c1]=0;
        }
        for(c2=0; c2<BroadSXSCount[0][c1]; c2++)
        {
            for(c3=0; c3<BroadSXSCount[0][c1]; c3++)
            {
                BroadSXS[3][c2][0][c1]=BroadSXS[3][c2][0][c1]+(Lorentz[c3][c2]*BroadSXS[1][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]));
            }
        }
        for(c3=0; c3<BroadSXSCount[0][c1]; c3++)
        {
            BroadSXS[6][c3][0][c1]=0;
        }
        for(c2=0; c2<BroadSXSCount[0][c1]; c2++)
        {
            for(c3=0; c3<BroadSXSCount[0][c1]; c3++)
            {
                BroadSXS[6][c2][0][c1]=BroadSXS[6][c2][0][c1]+(Gauss[c3][c2]*BroadSXS[3][c3][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]));
            }
        }

        /*for(c3=0; c3<BroadSXSCount[0][c1];c3++)
        {
            BroadSXS[6][c3][0][c1]=0;
        }
        for(c3=0; c3<BroadSXSCount[0][c1]; c3++)
        {
            for(c4=0; c4<BroadSXSCount[0][c1]; c4++)
            {
                BroadSXS[6][c3][0][c1]=BroadSXS[6][c3][0][c1]+(Disorder[c4][c3]*BroadSXS[5][c4][0][c1]*(BroadSXS[0][1][0][c1]-BroadSXS[0][0][0][c1]));
            }
        }*/
    }
    
    for(c1=0; c1< CalcSXSCase; c1++)
    {
        for(c2=1; c2<3; c2++)
        {
            for(c3 = 0; c3 < BroadSXSCount[c2][c1]; c3++) //This cycles through the matrix rows
            {
                width = BroadSXS[4][c3][c2][c1]/2.3548; //We extract the variance for the Gaussian Distribution
                position = BroadSXS[0][c3][c2][c1]; //We extract the centroid of the Gaussian Distribution
                for(c4=0; c4< BroadSXSCount[c2][c1];c4++)
                {
                    Gauss[c3][c4] = 1/sqrt(2*Pi*width*width)*exp(-(BroadSXS[0][c4][c2][c1]-position)*(BroadSXS[0][c4][c2][c1]-position)/2/width/width);
                }

                width = disord/2.3548; //We extract the variance for the Gaussian Distribution
                position = BroadSXS[0][c3][c2][c1]; //We extract the centroid of the Gaussian Distribution
                for(c4=0; c4< BroadSXSCount[c2][c1];c4++)
                {
                    Disorder[c3][c4] = 1/sqrt(2*Pi*width*width)*exp(-(BroadSXS[0][c4][c2][c1]-position)*(BroadSXS[0][c4][c2][c1]-position)/2/width/width);
                }

                width = BroadSXS[2][c3][c2][c1]/2; //We extract the variance for the Lorentz Distribution
                position = BroadSXS[0][c3][c2][c1]; //We extract the centroid of the Lorentz Distribution
                for(c4=0; c4< BroadSXSCount[c2][c1];c4++)
                {
                    Lorentz[c3][c4] = 1/Pi*(width/((BroadSXS[0][c4][c2][c1]-position)*(BroadSXS[0][c4][c2][c1]-position)+(width*width)));
                }
            }
            for(c3=0; c3<BroadSXSCount[c2][c1]; c3++)
            {
                BroadSXS[3][c3][c2][c1]=0;
            }
            for(c4=0; c4<BroadSXSCount[c2][c1]; c4++)
            {
                for(c3=0; c3<BroadSXSCount[c2][c1]; c3++)
                {
                    BroadSXS[3][c4][c2][c1]=BroadSXS[3][c4][c2][c1]+(Lorentz[c4][c3]*BroadSXS[1][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]));
                }
            }
            for(c3=0; c3<BroadSXSCount[c2][c1]; c3++)
            {
                BroadSXS[5][c3][c2][c1]=0;
            }
            for(c4=0; c4<BroadSXSCount[c2][c1]; c4++)
            {
                for(c3=0; c3<BroadSXSCount[c2][c1]; c3++)
                {
                    BroadSXS[5][c4][c2][c1]=BroadSXS[5][c4][c2][c1]+(Gauss[c4][c3]*BroadSXS[3][c3][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]));
                }
            }

            for(c3=0; c3<BroadSXSCount[c2][c1];c3++)
            {
                BroadSXS[6][c3][c2][c1]=0;
            }
            for(c3=0; c3<BroadSXSCount[c2][c1]; c3++)
            {
                for(c4=0; c4<BroadSXSCount[c2][c1]; c4++)
                {
                    BroadSXS[6][c3][c2][c1]=BroadSXS[6][c3][c2][c1]+(Disorder[c4][c3]*BroadSXS[5][c4][c2][c1]*(BroadSXS[0][1][c2][c1]-BroadSXS[0][0][c2][c1]));
                }
            }
        }
    }
    return BroadSXS;
}