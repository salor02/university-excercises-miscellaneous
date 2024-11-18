#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle/particle_filter.h"

//0: wheel, 1: stratified, 2: systematic
#define RESAMPLE_TYPE 0
//#define WEIGHTED_EUCLIDEAN

using namespace std;

static  default_random_engine gen;


/*
* This function initialize randomly the particles
* Input:
*  std - noise that might be added to the position
*  nParticles - number of particles
*/
void ParticleFilter::init_random(int nParticles, std::pair<float, float> min_pt, std::pair<float, float> max_pt) {
    num_particles = nParticles;
    uniform_real_distribution<double> dist_x(min_pt.first, max_pt.first);
    uniform_real_distribution<double> dist_y(min_pt.second, max_pt.second);
    normal_distribution<double> dist_theta(0, 2 * 3.14159);

    for(int i=0; i < num_particles; i++){
        particles.push_back(Particle(dist_x(gen), dist_y(gen), dist_theta(gen))); 
    }
    is_initialized=true;
}

/*
* This function initialize the particles using an initial guess
* Input:
*  x,y,theta - position and orientation
*  std - noise that might be added to the position
*  nParticles - number of particles
*/ 
void ParticleFilter::init(double x, double y, double theta, double std[],int nParticles) {
    num_particles = nParticles;
    normal_distribution<double> dist_x(-std[0], std[0]); //random value between [-noise.x,+noise.x]
    normal_distribution<double> dist_y(-std[1], std[1]);
    normal_distribution<double> dist_theta(-std[2], std[2]);

    for(int i=0; i < num_particles; i++){
        particles.push_back(Particle(x + dist_x(gen), y + dist_y(gen), theta + dist_theta(gen))); 
    }
    is_initialized=true;
}

/*
* The predict phase uses the state estimate from the previous timestep to produce an estimate of the state at the current timestep
* Input:
*  delta_t  - time elapsed beetween measurements
*  std_pos  - noise that might be added to the position
*  velocity - velocity of the vehicle
*  yaw_rate - current orientation
* Output:
*  Updated x,y,theta position
*/
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //for each particle
    for(Particle& particle : particles){
        double x,y,theta;
        if (fabs(yaw_rate) < 0.00001) {
            x = particle.x + velocity * delta_t *std::cos(particle.theta);
            y = particle.y + velocity * delta_t * std::sin(particle.theta);
            theta = particle.theta;
        }else{ 
            x = particle.x + (velocity/yaw_rate) * (std::sin(particle.theta + yaw_rate * delta_t) - std::sin(particle.theta));
            y = particle.y + (velocity/yaw_rate) * (std::cos(particle.theta) - std::cos(particle.theta + yaw_rate * delta_t));
            theta = particle.theta + yaw_rate * delta_t;
        }   
        normal_distribution<double> dist_x(0, std_pos[0]); //the random noise cannot be negative in this case
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);

        //add the computed noise to the current particles position (x,y,theta)
        particle.x = x + dist_x(gen);
        particle.y = y + dist_y(gen);
        particle.theta = theta + dist_theta(gen);
	}
}

/*
* This function associates the landmarks from the MAP to the landmarks from the OBSERVATIONS
* Input:
*  mapLandmark   - landmarks of the map
*  observations  - observations of the car
* Output:
*  Associated observations to mapLandmarks (perform the association using the ids)
*/
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> mapLandmark, std::vector<LandmarkObs>& observations, double std_pos[]) {
   //TIP: Assign to observations[i].id the id of the landmark with the smallest euclidean distance

    #ifdef WEIGHTED_EUCLIDEAN
    double dim_weight[2];
    dim_weight[0] = 1 / (std::pow(std_pos[0], 2));
    dim_weight[1] = 1 / (std::pow(std_pos[1], 2));
    #endif

    for(LandmarkObs& observation : observations){
        double min_dist = std::numeric_limits<int>::max();
        int min_dist_id;
        for(LandmarkObs landmark : mapLandmark){
            #ifdef WEIGHTED_EUCLIDEAN
            double dist = std::sqrt(dim_weight[0] * std::pow((landmark.x - observation.x), 2) + dim_weight[1] * std::pow((landmark.y - observation.y), 2));
            #else
            double dist = std::sqrt(std::pow((landmark.x - observation.x), 2) + std::pow((landmark.y - observation.y), 2));
            #endif
            if(dist < min_dist){
                min_dist = dist;
                min_dist_id = landmark.id;
            }
        }
        observation.id = min_dist_id;
    }
}

/*
* This function transform a local (vehicle) observation into a global (map) coordinates
* Input:
*  observation   - A single landmark observation
*  p             - A single particle
* Output:
*  local         - transformation of the observation from local coordinates to global
*/
LandmarkObs transformation(LandmarkObs observation, Particle p){
    LandmarkObs global;
    
    global.id = observation.id;
    global.x = observation.x * std::cos(p.theta) - observation.y * std::sin(p.theta) + p.x;
    global.y = observation.x * std::sin(p.theta) + observation.y* std::cos(p.theta) + p.y;

    return global;
}

/*
* TODO
* This function updates the weights of each particle
* Input:
*  std_landmark   - Sensor noise
*  observations   - Sensor measurements
*  map_landmarks  - Map with the landmarks
* Output:
*  Updated particle's weight (particles[i].weight *= w)
*/
void ParticleFilter::updateWeights(double std_landmark[], 
	std::vector<LandmarkObs> observations, std::vector<LandmarkObs> mapLandmark, double std_pos[]) {

    //Creates a vector that stores tha map (this part can be improved)
    
    for(int i=0;i<particles.size();i++){

        // Before applying the association we have to transform the observations in the global coordinates
        std::vector<LandmarkObs> transformed_observations;

        //TODO: for each observation transform it (transformation function)
        for(LandmarkObs observation : observations){
            transformed_observations.push_back(transformation(observation, particles[i]));
        }

        //TODO: perform the data association (associate the landmarks to the observations)
        dataAssociation(mapLandmark, transformed_observations, std_pos);

        particles[i].weight = 1.0;
        // Compute the probability
		//The particles final weight can be represented as the product of each measurement’s Multivariate-Gaussian probability density
		//We compute basically the distance between the observed landmarks and the landmarks in range from the position of the particle
        for(int k=0;k<transformed_observations.size();k++){
            double obs_x,obs_y,l_x,l_y;
            obs_x = transformed_observations[k].x;
            obs_y = transformed_observations[k].y;
            //get the associated landmark 
            for (int p = 0; p < mapLandmark.size(); p++) {
                if (transformed_observations[k].id == mapLandmark[p].id) {
                    l_x = mapLandmark[p].x;
                    l_y = mapLandmark[p].y;
                }
            }	
			// How likely a set of landmarks measurements are, given a prediction state of the car 
            double w = exp( -( pow(l_x-obs_x,2)/(2*pow(std_landmark[0],2)) + pow(l_y-obs_y,2)/(2*pow(std_landmark[1],2)) ) ) / ( 2*M_PI*std_landmark[0]*std_landmark[1] );
            particles[i].weight *= w;
        }

    }    
}

/*
* TODO
* This function resamples the set of particles by repopulating the particles using the weight as metric
*/
void ParticleFilter::resample() {
    
    uniform_int_distribution<int> dist_distribution(0,num_particles-1);
    vector<double> weights;
    vector<Particle> new_particles;

    double total_weight = 0;

    for(int i=0;i<num_particles;i++){
        weights.push_back(particles[i].weight);
        
        //per lo stratified resampling
        total_weight += particles[i].weight;
    }
																
    //wheel resampling
    #if RESAMPLE_TYPE == 0
    int index = dist_distribution(gen);
    double beta  = 0.0;
    float max_w = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> uni_dist(0.0, max_w);

    for(int i=0; i < num_particles; i++){
        beta += uni_dist(gen) * 2;
        while(weights[index] < beta){
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    //stratified resampling
    #else
    //calcolo della distribuzione cumulativa dei pesi
    std::vector<double> cumulative_weights(num_particles);
    cumulative_weights[0] = particles[0].weight / total_weight;
    for (int i = 1; i < num_particles; ++i) {
        cumulative_weights[i] = cumulative_weights[i - 1] + particles[i].weight / total_weight;
    }

    //divisione dell'intervallo [0,1] in num_particles sezioni
    double section = 1.0 / num_particles;

    //generazione di un numero casuale all'interno di ogni sezione e calcolo della particella corrispondente
    std::uniform_real_distribution<double> dist_section(0.0, section);
    
    //utile al systematic resample
    double start = dist_section(gen);

    int index = 0;
    for(int i = 0; i < num_particles; i++){
    #if RESAMPLE_TYPE == 1
        double section_random = dist_section(gen) + section * i;
    #elif RESAMPLE_TYPE == 2
        double section_random = start + section * i;
    #endif
        while(section_random > cumulative_weights[index] && index < num_particles-1){
            index++;
        }

        new_particles.push_back(particles[index]);
    }

    #endif

    particles.swap(new_particles);
}


