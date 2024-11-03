#include "tracker/Tracker.h"
#include <iostream>

Tracker::Tracker()
{
    cur_id_ = 0;
    distance_threshold_ = 1.2; // meters
    covariance_threshold = 0.5; 
    loss_threshold = 15; //number of frames the track has not been seen
}
Tracker::~Tracker()
{
}

/*
    This function removes tracks based on any strategy
*/
void Tracker::removeTracks()
{
    std::vector<Tracklet> tracks_to_keep;

    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        // TODO
        // Implement logic to discard old tracklets
        if(tracks_[i].getXCovariance() < covariance_threshold && tracks_[i].getYCovariance() < covariance_threshold){
            if (tracks_[i].getLossCount() < loss_threshold){
                tracks_to_keep.push_back(tracks_[i]);
            }
        }
        
    }

    tracks_.swap(tracks_to_keep);
}

/*
    This function add new tracks to the set of tracks ("tracks_" is the object that contains this)
*/
void Tracker::addTracks(const std::vector<bool> &associated_detections, const std::vector<double> &centroids_x, const std::vector<double> &centroids_y)
{
    // Adding not associated detections
    for (size_t i = 0; i < associated_detections.size(); ++i)
        if (!associated_detections[i])
            tracks_.push_back(Tracklet(cur_id_++, centroids_x[i], centroids_y[i]));
}

/*
    This function associates detections (centroids_x,centroids_y) with the tracks (tracks_)
    Input:
        associated_detection an empty vector to host the associated detection
        centroids_x & centroids_y measurements representing the detected objects
*/
void Tracker::dataAssociation(std::vector<bool> &associated_detections, const std::vector<double> &centroids_x, const std::vector<double> &centroids_y)
{

    //Remind this vector contains a pair of tracks and its corresponding
    associated_track_det_ids_.clear();

    for (size_t i = 0; i < tracks_.size(); ++i)
    {

        int closest_point_id = -1;
        double min_dist = std::numeric_limits<double>::max();
        double track_x = tracks_[i].getX();
        double track_y = tracks_[i].getY();

        for (size_t j = 0; j < associated_detections.size(); ++j)
        {
            // TODO
            // Implement logic to find the closest detection (centroids_x,centroids_y) 
            // to the current track (tracks_)
            if(!associated_detections[j]){
                double dist = sqrt(pow(centroids_x[j]-track_x, 2) + pow(centroids_y[j]-track_y, 2));
                if(dist < min_dist){
                    min_dist = dist;
                    closest_point_id = j;
                }
            }
        }

        // Associate the closest detection to a tracklet
        if (min_dist < distance_threshold_ && !associated_detections[closest_point_id])
        {
            associated_track_det_ids_.push_back(std::make_pair(closest_point_id, i));
            associated_detections[closest_point_id] = true;
        }
    }
}

void Tracker::track(const std::vector<double> &centroids_x,
                    const std::vector<double> &centroids_y,
                    bool lidarStatus)
{

    std::vector<bool> associated_detections(centroids_x.size(), false);

    // TODO: Predict the position
    //For each track --> Predict the position of the tracklets
    for (size_t i = 0; i < tracks_.size(); ++i){
        tracks_[i].predict();
    }
    
    // TODO: Associate the predictions with the detections
    dataAssociation(associated_detections, centroids_x, centroids_y);

    // Update tracklets with the new detections
    for (int i = 0; i < associated_track_det_ids_.size(); ++i)
    {
        auto det_id = associated_track_det_ids_[i].first;
        auto track_id = associated_track_det_ids_[i].second;
        tracks_[track_id].update(centroids_x[det_id], centroids_y[det_id], lidarStatus);
    }

    // TODO: Remove dead tracklets
    removeTracks();
    // TODO: Add new tracklets
    addTracks(associated_detections, centroids_x, centroids_y);
}
