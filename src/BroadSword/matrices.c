#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

float Gauss[3500][3500]; //This stores the Gauss broadening matrix for each spectrum
float Lorentz[3500][3500]; //This stores the Lorentz broadening matrix for each spectrum
float Disorder[3500][3500]; //This stores the disorder broadening matrix for each spectrum

int main(){    
    printf("Well done \n");
    return 0;   
}

int broadXAS(int CalcSXSCase, int BroadSXSCount[3][40], float BroadSXS[7][3500][3][40], float disord){ 
    // BroadSXS may have to be allocated outside.
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
    return 0;
}

int add(int CalcSXSCase, float scalar[3][40], int Edge[40], float Site[40], float BroadSXS[7][3500][3][40], 
int BroadSXSCount[3][40], float SumSXS[2][3500][3], int SumSXSCount[3]){
    // Might have some issues here with the character, since it's not strings. AHHHHH. Added a [4] to the end to get array of strings.

    //Variable to check edges
    // char Edge_check[5][4] = {"K", "L2", "L3", "M4", "M5"}; // This is what we compare the edge against the get the proportionality
    int Edge_check[5] = {1, 2, 3, 4, 5};
    float Edge_scale[5] = {1, 0.333333, 0.66666667, 0.4, 0.6};
    float x1, x2, y1, y2;
    float slope;
    int c1, c2, c3, c4;
    int first; // This is the first spectra added, all others will be added to it.
    float value; // This is a dummy variable to compare values
    int max = 0;

    //Here we determine the relative addition scale factor XES
    for(c1 = 0; c1 < CalcSXSCase; c1++)
    {
        for(c2=0; c2< 3; c2++)

        {
            scalar[c2][c1]=1;
        }
    }

    //Now we apply the scaling to the running scaler
    for(c1 = 0; c1 < CalcSXSCase; c1++)//This counts through the different Calculated spectra
    {
        for (c2 = 0; c2 < 5; c2++) //This counts through the types of edges
        {
            if(Edge[c1] == Edge_check[c2])
            {
                for(c3=0; c3<3; c3++)
                {
                    scalar[c3][c1] = scalar[c3][c1]*Site[c1]*Edge_scale[c2];
                }
            }
        }
    }


    for(c1=0; c1<3; c1++)
    {
        first=0;
        value=BroadSXS[0][0][c1][0];
        for(c2=1; c2<CalcSXSCase; c2++)
        {
            if(BroadSXS[0][0][c1][c2]>=value)
            {
                first=c2;
            }
        }

        for(c3=0;c3<BroadSXSCount[c1][first];c3++)
        {
            SumSXS[0][c3][c1]=BroadSXS[0][c3][c1][first];
            SumSXS[1][c3][c1]=scalar[c1][first]*BroadSXS[6][c3][c1][first];
        }

        SumSXSCount[c1]=c3;

        for(c2=0; c2<CalcSXSCase; c2++)
        {
            if(c2!=first)
            {
                for(c3=0; c3<SumSXSCount[c1];c3++)
                {
                    for(c4=0;c4<BroadSXSCount[c1][c2];c4++)
                    {
                        if(BroadSXS[0][c4][c1][c2]>SumSXS[0][c3][c1])
                        {
                            x1=BroadSXS[0][c4-1][c1][c2];
                            x2=BroadSXS[0][c4][c1][c2];
                            y1=BroadSXS[6][c4-1][c1][c2];
                            y2=BroadSXS[6][c4][c1][c2];
                            slope = (y2-y1)/(x2-x1);
                            SumSXS[1][c3][c1] = SumSXS[1][c3][c1] + scalar[c1][c2]*(slope*(SumSXS[0][c3][c1]-x1)+y1);
                            c4=9999999;
                            max=c3;
                        }
                    }
                }
                SumSXSCount[c1]=max;
            }

        }

    }
    return 0;
}
