#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <math.h> // for exp()

using namespace cv;

//Value of number of gaussians in n mixture and dimension of variables
#define K 5
#define dim 3

float TOL = 1e-5;
int i_rows = 0, i_cols = 0;
int fg_start_row = 0, fg_start_col = 0;
int fg_end_row = 0, fg_end_col = 0;
int log_fg_old = 0, log_bg_old = 0;
int log_fg_new = 0, log_bg_new = 0;


//Mean of 5 Gaussian distributions of foreground and background
float mean_fg_new[dim][K] = {0};
float mean_bg_new[dim][K] = {0};
float mean_fg_old[dim][K] = {0};
float mean_bg_old[dim][K] = {0};

float resp[K] = {0};
float pi_fg_new[K] = {0};
float pi_bg_new[K] = {0};
float pi_fg_old[K] = {0};
float pi_bg_old[K] = {0};

//Variance matrix of 5 Gaussian distributions of foreground and background
float sigma_fg_old[dim][dim][K] = {0};
float sigma_bg_old[dim][dim][K] = {0};
float sigma_fg_new[dim][dim][K] = {0};
float sigma_bg_new[dim][dim][K] = {0};

void initialize_mixtures()
{
	for (int i=0; i<K; i++)
	{
		for (int j=0; j<dim; j++)
		{
			mean_fg_old[dim][i] = getRandomNumber(0.1,0.9)*255;
			mean_bg_old[dim][i] = getRandomNumber(0.1,0.9)*255;
		}
	}

	for (int i=0; i<K; i++)
	{
		pi_fg_old[i] = 1.0/K;
		pi_bg_old[i] = 1.0/K;
	}

	for (int k=0; k<K; k++)
	{
		for (int i=0; i<dim; i++)
		{
			for (int j=i; j<dim; j++)
			{
				if (i==j)
				{
					sigma_fg_old[i][j][k] = getRandomNumber(0.1, 0.5)*255;
					sigma_bg_old[i][j][k] = getRandomNumber(0.1, 0.5)*255;
				}
				else
				{
					sigma_fg_old[i][j][k] = getRandomNumber(0.1, 0.5)*255;
					sigma_fg_old[j][i][k] = getRandomNumber(0.1, 0.5)*255;
					sigma_bg_old[i][j][k] = getRandomNumber(0.1, 0.5)*255;
					sigma_bg_old[j][i][k] = getRandomNumber(0.1, 0.5)*255;
				}
			}
		}
	}
}

float gaussian_fg(unsigned int x[dim], int k , int flag_old)
{
	float inv_sigma[dim][dim];
	float  x_mu[dim];
	float result = 0;
	float exponent = 0;
	float determinant = 0;
	// Inverse of covariance matrix
	if(flag_old = 0)
	{
		for( i=0; i<3; i++)
		{
  			determinant = determinant + (sigma_fg_old[0][i][k]*(sigma_fg_old[1][(i+1)%3][k]*
  			sigma_fg_old[2][(i+2)%3][k] - sigma_fg_old[1][(i+2)%3][k]*sigma_fg_old[2][(i+1)%3][k]));
		}
	 	if(determinant==0)
	 		cout<<"Inverse does not exist (Determinant=0).\n";
		else
		{
		 	for(i=0;i<3;i++)
		 	{
		  		for(j=0;j<3;j++)
		  		{
		   			inv_sigma[i][j] = (sigma_fg_old[(i+1)%3][(j+1)%3][k] *sigma_fg_old[(i+2)%3][(j+2)%3][k]) - (sigma_fg_old[(i+1)%3][(j+2)%3][k]*
		 			 sigma_fg_old[(i+2)%3][(j+1)%3][k]))/ determinant<<"\t";
		  		}
		  	}
		 }
	}
	else
	{
		for( i=0; i<3; i++)
		{
  			determinant = determinant + (sigma_fg_new[0][i][k]*(sigma_fg_new[1][(i+1)%3][k]*
  			sigma_fg_new[2][(i+2)%3][k] - sigma_fg_new[1][(i+2)%3][k]*sigma_fg_new[2][(i+1)%3][k]));
		}
	 	if(determinant==0)
	 		cout<<"Inverse does not exist (Determinant=0).\n";
		else
		{
		 	for(i=0;i<3;i++)
		 	{
		  		for(j=0;j<3;j++)
		  		{
		   			inv_sigma[i][j] = (sigma_fg_new[(i+1)%3][(j+1)%3][k] *sigma_fg_new[(i+2)%3][(j+2)%3][k]) - (sigma_fg_new[(i+1)%3][(j+2)%3][k]*
		 			 sigma_fg_new[(i+2)%3][(j+1)%3][k]))/ determinant<<"\t";
		  		}
		  	}
		 }
	}

	// calculating exponent
	if( flag_old = 0)
	{
		for(int i =0; i<dim; i++)
			x_mu[i] = x[i] - mean_fg_old[i][k];
	}
	else
	{
		for(int i =0; i<dim; i++)
			x_mu[i] = x[i] - mean_fg_new[i][k];
	}

	 float temp[dim] = {0}
	for(int i =0; i<dim; i++)
	{
		for(int j=0;j<dim ; j++)
			temp[i] + = x_mu[j]*inv_sigma[i][j];
		exponent  += temp[i]*x_mu[i];
	}
	result = 1/sqrt(2*pi*determinant)*exp(- 0.5*exponent);
	return result;
}

float gaussian_bg(unsigned int x[dim], int k , int flag_old)
{
	float inv_sigma[dim][dim];
	float  x_mu[dim];
	float result = 0;
	float exponent = 0;
	float determinant = 0;
	// Inverse of covariance matrix
	if(flag_old = 0)
	{
		for( i=0; i<3; i++)
		{
  			determinant = determinant + (sigma_bg_old[0][i][k]*(sigma_bg_old[1][(i+1)%3][k]*
  			sigma_bg_old[2][(i+2)%3][k] - sigma_bg_old[1][(i+2)%3][k]*sigma_bg_old[2][(i+1)%3][k]));
		}
	 	if(determinant==0)
	 		cout<<"Inverse does not exist (Determinant=0).\n";
		else
		{
		 	for(i=0;i<3;i++)
		 	{
		  		for(j=0;j<3;j++)
		  		{
		   			inv_sigma[i][j] = (sigma_bg_old[(i+1)%3][(j+1)%3][k] *sigma_bg_old[(i+2)%3][(j+2)%3][k]) - (sigma_bg_old[(i+1)%3][(j+2)%3][k]*
		 			 sigma_bg_old[(i+2)%3][(j+1)%3][k]))/ determinant<<"\t";
		  		}
		  	}
		 }
	}
	else
	{
		for( i=0; i<3; i++)
		{
  			determinant = determinant + (sigma_bg_new[0][i][k]*(sigma_bg_new[1][(i+1)%3][k]*
  			sigma_bg_new[2][(i+2)%3][k] - sigma_bg_new[1][(i+2)%3][k]*sigma_bg_new[2][(i+1)%3][k]));
		}
	 	if(determinant==0)
	 		cout<<"Inverse does not exist (Determinant=0).\n";
		else
		{
		 	for(i=0;i<3;i++)
		 	{
		  		for(j=0;j<3;j++)
		  		{
		   			inv_sigma[i][j] = (sigma_bg_new[(i+1)%3][(j+1)%3][k] *sigma_bg_new[(i+2)%3][(j+2)%3][k]) - (sigma_bg_new[(i+1)%3][(j+2)%3][k]*
		 			 sigma_bg_new[(i+2)%3][(j+1)%3][k]))/ determinant<<"\t";
		  		}
		  	}
		 }
	}

	// calculating exponent
	if( flag_old = 0)
	{
		for(int i =0; i<dim; i++)
			x_mu[i] = x[i] - mean_bg_old[i][k];
	}
	else
	{
		for(int i =0; i<dim; i++)
			x_mu[i] = x[i] - mean_bg_new[i][k];
	}

	 float temp[dim] = {0}
	for(int i =0; i<dim; i++)
	{
		for(int j=0;j<dim ; j++)
			temp[i] + = x_mu[j]*inv_sigma[i][j];
		exponent  += temp[i]*x_mu[i];
	}
	result = 1/sqrt(2*pi*determinant)*exp(- 0.5*exponent);
	return result;
}

void responsibility_calculate(unsigned int x[dim], int flag_fg_bg )
{
	float sum = 0;
	if(flag_fg_bg==0)
	{
		for(int k=0; k<K; k++)
		{
			for(i=0; i<dim; i++)
			{
				for(j=0; j<dim; j++)
				{
					sum += pi_fg_old[k]*gaussian_fg(x,k,0);
				}
			}
		}
		for(int k=0;k<K; k++)
		{
			resp[k] = pi_fg_old[k]*gaussian_fg(x,k,0)/ sum;
		}
	}
	else
	{
		for(int k=0; k<K; k++)
		{
			for(i=0; i<dim; i++)
			{
				for(j=0; j<dim; j++)
				{
					sum += pi_bg[k]*gaussian_bg(x,k,0);
				}
			}
		}
		for(int k=0;k<K; k++)
		{
			resp[k] = pi_bg[k]*gaussian_bg(x,k,0)/ sum;
		}
	}

}

void mean_calculate(const Mat& img)
{
	float resp_sum_fg[K] = {0};
	float resp_sum_bg[K] = {0};
	float mean_sum_fg[dim][K] = {0};
	float mean_sum_bg[dim][K] = {0};

	for (int r=0; r<i_rows; r++)
	{
		for (int c=0; c<i_cols; c++)
		{
			Vec3b bgr = img.at<Vec3b>(r,c);
			unsigned int x[dim] = {bgr[0], bgr[1], bgr[2]};
			if ((fg_start_row<r<fg_end_row) && (fg_start_col<c<fg_end_col))
			{
				responsibility_calculate(x,0);
				for (int i=0; i<K; i++)
				{
					resp_sum_fg[i] += resp[i];
					for (int j=0; j<dim; j++)
					{
						mean_sum_fg[j][i] += resp[i]*x[j];
					}
				}
			}
			else
			{
				responsibility_calculate(x,1);
				for (int i=0; i<K; i++)
				{
					resp_sum_bg[i] += resp[i];
					for (int j=0; j<dim; j++)
					{
						mean_sum_bg[j][i] += resp[i]*x[j];
					}
				}
			}
		}
	}

	for (int i=0; i<K; i++)
	{
		for (int j=0; j<dim; j++)
		{
			mean_fg_new[j][i] = mean_sum_fg[j][i]/resp_sum_fg[i];
			mean_bg_new[j][i] = mean_sum_bg[j][i]/resp_sum_bg[i];
		}
	}
}

void sigma_calculate(const Mat& img)
{
	float resp_sum_fg[K] = {0};
	float resp_sum_bg[K] = {0};
	float sigma_temp_bg[dim][dim][K] = {0};
	float sigma_temp_fg[dim][dim][K] = {0};

	for (int r=0; r<i_rows; r++)
	{
		for (int c=0; c<i_cols; c++)
		{
			Vec3b bgr = img.at<Vec3b>(r,c);
			unsigned int x[dim] = {bgr[0], bgr[1], bgr[2]};
			if ((fg_start_row<r<fg_end_row) && (fg_start_col<c<fg_end_col))
			{
				responsibility_calculate(x,0);
				for(int k=0; k<K; k++)
				{
					resp_sum_fg[k] +=resp[k];
					for(i=0; i<dim; i++)
					{
						for(j=0; j<dim; j++)
						{
						 	sigma_temp_fg[i][j][k] +=  resp[k]*(x[i]-mean_fg_new[i][k])*(x[j]-mean_fg_new[j][k]);
						}
					}
				}
			}
			else
			{
				responsibility_calculate(x,1);
				for(int k=0; k<K; k++)
				{
					resp_sum_bg[k] +=resp[k];
					for(i=0; i<dim; i++)
					{
						for(j=0; j<dim; j++)
						{
						 	sigma_temp_bg[i][j][k] +=  resp[k]*(x[i]-mean_bg_new[i][k])*(x[j]-mean_bg_new[j][k]);
						}
					}
				}
			}
		}
	}
	for(int k=0; k<K; k++)
	{
		for(i=0; i<dim; i++)
		{
			for(j=0; j<dim; j++)
			{
				sigma_fg_new[i][j][k] = sigma_temp_fg[i][j][k]/resp_sum_fg[k];
				sigma_bg_new[i][j][k] = sigma_temp_bg[i][j][k]/resp_sum_bg[k];
			}
		}
	}
	// updating pi_fg, pi_bg
	N = i_rows*i_cols;
	for(int k=0; k<K; k++)
	{
		pi_fg_new = resp_sum_fg[k]/N;
		pi_bg_new = resp_sum_bg[k]/N;
	}
}

int getRandomNumber(int min, int max)
{
	// static used for efficiency, so we only calculate this value once
	static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);

    // evenly distribute the random number across our range
	return static_cast<int>(rand() * fraction * (max - min + 1) + min);
}

int log_likelihood(const Mat& img)
{
	ll_fg = 0;
	ll_bg = 0;
	for (int r=0; r<i_rows; r++)
	{
		for (int c=0; c<i_cols; c++)
		{
			Vec3b bgr = img.at<Vec3b>(r,c);
			unsigned int x[dim] = {bgr[0], bgr[1], bgr[2]};
			if ((fg_start_row<r<fg_end_row) && (fg_start_col<c<fg_end_col))
			{
				for (int k=0; k<K; k++)
				{
					ll_fg += pi_fg_new[k]*gaussian_fg(x,k,1);
				}
			}
			else
			{
				for (int k=0; k<K; k++)
				{
					ll_bg += pi_bg_new[k]*gaussian_bg(x,k,1);
				}
			}
		}
	}
	log_fg_new = log(ll_fg);
	log_bg_new = log(ll_bg);
}
bool convergence()
{
	float pi_fg_norm = 0;
	float mu_fg_norm = 0;
	float sigma_fg_norm = 0;

	for(int i=0;i<K; i++)
		pi_fg_norm += (pi_fg_new[i] - pi_fg_old[i])^2;
	pi_fg_norm = pi_fg_norm^0.5;

	for(int i=0; i<dim ; i++)
	{
		for(int k=0;k<K;k++)
			mu_fg_norm += (mu_fg_new[i][k] - mu_fg_old[i][k])^2;


	}
	mu_fg_norm = mu_fg_norm^0.5;

	for(int k = 0; k< K ; k++)
	{
		for(int i=0; i<dim ; i++)
		{
			for(int j=0; j<dim; j++)
				sigma_fg_norm += (sigma_fg_new[i][j][k] - sigma_fg_old[i][j][k])^2;


		}
	}
	sigma_fg_norm = sigma_fg_norm^0.5;






}



