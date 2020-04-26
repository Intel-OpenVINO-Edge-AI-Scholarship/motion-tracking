#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math>

using namespace std;

/**
** estimated using exponential functions
**/
float estimate_distance(float camera_known_proximity, float camera_known_distant, 
	float estimated_dimension, float wall_reference_center_dimension, std::vector<float> reference_dimensions,
	float alpha=1e-5) {
	float coefficient = 0.0f;
	if(estimated_dimension < wall_reference_center_dimension) {
		coefficient = (wall_reference_center_dimension - estimated_dimension) / (wall_reference_center_dimension - reference_dimensions[1]);

		return (exp(alpha * -coefficient) * camera_known_proximity + exp(alpha * coefficient) * camera_known_distant) / (exp(alpha * coefficient)+exp(alpha * -coefficient));
	} else if(estimated_dimension > wall_reference_center_dimension) {
		coefficient = (estimated_dimension - wall_reference_center_dimension) / (reference_dimensions[0] - wall_reference_center_dimension);

		return (exp(alpha * coefficient) * camera_known_proximity + exp(alpha * -coefficient) * camera_known_distant) / (exp(alpha * coefficient)+exp(alpha * -coefficient));
	} else {
		return (camera_known_proximity + camera_known_distant) / 2
	}
}

/**
** plugged in dimensions from estimate_dimension, regular_length, regular_height
**/
float camera_center_wall_reference_dimension(float camera_proximity_dimension, float camera_wall_dimension) {
	return (camera_proximity_dimension + camera_wall_dimension) / 2
}

/**
** distance estimated using weighted sum
**/
float estimate_dimension(float regular_length, float regular_height, float hfov, float vfov) {
	return (regular_length/hfov + regular_height/vfov) / (1/hfov + 1/vfov) * 180/PI;
}

/**
** regular length measured for face only
**/
float regular_length(float measured_length, float camera_reference_length, float measured_reference_length) {
	return camera_reference_length * measured_length / measured_reference_length;
}

/**
** regular height measured for face and body
**/
float regular_height(float measured_height, float camera_reference_height, float measured_reference_height) {
	return camera_reference_length * measured_height / measured_reference_height;
}

