/*
 *  Created on: May 30, 2017
 *      Author: Neeraj Dixit
 */
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>
#include <map>
#include <math.h>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

vector<double> GetNoiseVector(double x, double y, double theta, double std[]) {
	return { 
		normal_distribution<double> (x, std[0])(gen), 
		normal_distribution<double> (y, std[1])(gen), 
		normal_distribution<double> (theta, std[2])(gen)
	};
}

Particle ParticleGenerator(int id, double x, double y, double theta, double std[]) {
	vector<double> noiseVector = GetNoiseVector(x, y, theta, std);
    return {id, noiseVector[0], noiseVector[1], noiseVector[2], 1};
}

vector<LandmarkObs> Predict(vector<Map::single_landmark_s>& lm_list, Particle& p, double sensor_range) {
	vector<LandmarkObs> predicted;
	for_each(lm_list.begin(), lm_list.end(), [&](Map::single_landmark_s& lm) {
		double diff = dist(lm.x_f, lm.y_f, p.x, p.y);
		if(sensor_range >= diff) {
	        predicted.push_back({lm.id_i , lm.x_f , lm.y_f});
	    }
	});
	return predicted;
}

vector<LandmarkObs> TransformObservations(vector<LandmarkObs>& observations, Particle p) {
	vector<LandmarkObs> transformed(observations.size());
	for(int i = 0; i < observations.size(); i++) {
	    transformed[i].x = observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta) + p.x ;
	    transformed[i].y = observations[i].x * sin(p.theta) + observations[i].y * cos(p.theta) + p.y ;
	}
	return transformed;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 20;
	int id = -1;
	particles = vector<Particle>(num_particles);
	weights = vector<double>(num_particles);
	fill(weights.begin(), weights.end(), 1);
	fill(particles.begin(), particles.end(), ParticleGenerator(id++, x, y, theta, std));
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for_each (particles.begin(), particles.end(), [&](Particle& p) {
		vector<double> noiseVector = GetNoiseVector(0, 0, 0, std_pos);
		double delta_x = 0;
		double delta_y = 0; 
		double delta_yaw = 0;

		if(fabs(yaw_rate) < 1e-4) {
		    delta_x = velocity * delta_t * cos(p.theta);
		    delta_y = velocity * delta_t * sin(p.theta);
		} else {
		    double c = velocity/yaw_rate;
		    delta_x = c * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
		    delta_y = c * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		    delta_yaw = yaw_rate * delta_t;
		}
		//Add control noise
		delta_x += noiseVector[0];
		delta_y += noiseVector[1];
		delta_yaw += noiseVector[2];
		//Add predcition
		p.x += delta_x;
		p.y += delta_y;
		p.theta += delta_yaw;
    });
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
	for_each (observations.begin(), observations.end(), [&](LandmarkObs& o) {
		int min = 1e6;
	    int id = -1;
		for_each (predicted.begin(), predicted.end(), [&](LandmarkObs& p) {
			double distance = dist(p.x, p.y, o.x, o.y);
	        if(distance < min) {
	            min = distance;
	            id = p.id;
	        }
		});
		o.id = id;
	});
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {
	double total_weight = 0;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
	vector<Map::single_landmark_s> lm_list = map_landmarks.landmark_list;
    
    for(int i=0; i < num_particles; i++) {

        double prob = 1.0;
        double c = 1/(2*M_PI * std_x * std_y);

		vector<LandmarkObs> predicted = Predict(lm_list, particles[i], sensor_range);
		vector<LandmarkObs> transformed = TransformObservations(observations, particles[i]);
        dataAssociation(predicted , transformed);
        	
        for_each(transformed.begin(), transformed.end(), [&](LandmarkObs& o) {
			vector<Map::single_landmark_s>::iterator lm = find_if (lm_list.begin(), lm_list.end(),
				[&](Map::single_landmark_s& p){
				return p.id_i == o.id;
			});
            double x_diff = pow((o.x - lm->x_f)/std_x, 2.0);
            double y_diff = pow((o.y - lm->y_f)/std_y, 2.0);
            prob *= c*exp(-( x_diff + y_diff )/2.0);
        });
        
        weights[i] = prob;
        particles[i].weight = prob;
        total_weight += prob;
    }
    //normalize
    if(total_weight > 0) {
		for_each (weights.begin(), weights.end(), [&](double& w) {
			w /= total_weight;
		});
    }
}

void ParticleFilter::resample() {
	discrete_distribution<int> dd(weights.begin(), weights.end());
    vector<Particle> resampled(num_particles);
	fill(resampled.begin(), resampled.end(), particles[dd(gen)]);
    particles = resampled;
}

void ParticleFilter::write(string filename) {
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
