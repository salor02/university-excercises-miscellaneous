#include "tracker/Tracker.h"
#include "viewer/Renderer.h"
#include <iostream>

Tracker::Tracker()
{
    cur_id_ = 0;
    distance_threshold_ = 1.2; // meters
    covariance_threshold = 0.5; 
    loss_threshold = 15; //number of frames the track has not been seen

    //definizione area per tenere traccia dei tracciati che entrano
    area_.x_min=-1.5;
    area_.y_min=-1.5;
    area_.z_min=0;
    area_.x_max=1.5;
    area_.y_max=1.5;
    area_.z_max=0;
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
        if (!associated_detections[i]){
            tracks_.push_back(Tracklet(cur_id_++, centroids_x[i], centroids_y[i]));
            area_tracks_frames_.push_back(0); //ogni volta che si crea un nuovo tracciato si inizializza il numero di frame nell'area a 0
        }
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

/*  Un tracciato viene considerato come "entrato nell'area" se il numero di frame corrispondente
    contenuto nel vettore dedicato è maggiore di zero. */
int Tracker::getAreaCount(){
    return area_tracks_frames_.size() - std::count(area_tracks_frames_.begin(), area_tracks_frames_.end(), 0);
}

/*  Ricerca del massimo numero di frame per cui un tracciato è rimasto nell'area */
std::pair<int,int> Tracker::getLongestTrackInArea(){
    auto max_frames_it = std::max_element(area_tracks_frames_.begin(), area_tracks_frames_.end());
    int track_id = std::distance(area_tracks_frames_.begin(), max_frames_it);
    int max_value = *max_frames_it;

    return std::pair<int,int>(track_id, max_value);
}

void Tracker::track(const std::vector<double> &centroids_x,
                    const std::vector<double> &centroids_y,
                    bool lidarStatus)
{

    std::vector<bool> associated_detections(centroids_x.size(), false);

    //For each track --> Predict the position of the tracklets
    for (size_t i = 0; i < tracks_.size(); ++i){
        printf("Tracklet ID: %d\n", tracks_[i].getId());
        printf("\tStarting position: [%.5f,%.5f]\n", tracks_[i].getX(), tracks_[i].getY());
        tracks_[i].predict();
        printf("\tAfter prediction : [%.5f,%.5f]\n", tracks_[i].getX(), tracks_[i].getY());
    }
    
    // Associate the predictions with the detections
    dataAssociation(associated_detections, centroids_x, centroids_y);

    // Update tracklets with the new detections
    for (int i = 0; i < associated_track_det_ids_.size(); ++i)
    {
        auto det_id = associated_track_det_ids_[i].first;
        auto track_id = associated_track_det_ids_[i].second;
        tracks_[track_id].update(centroids_x[det_id], centroids_y[det_id], lidarStatus);

        printf("Tracklet ID: %d\n", tracks_[track_id].getId());
        printf("\tAfter update : [%.5f,%.5f]\n", tracks_[track_id].getX(), tracks_[track_id].getY());
        printf("\tTracklet length: %.5f\n", tracks_[track_id].getLength());

        //aggiornamento del tracciato più lungo
        if(tracks_[track_id].getLength() > longest_path_.second){
            longest_path_.first = tracks_[track_id].getId();
            longest_path_.second = tracks_[track_id].getLength();
        }

        //se il tracciato aggiornato è nell'area allora si aggiunge un frame al totale
        if(tracks_[track_id].getX() >= area_.x_min && tracks_[track_id].getX() <= area_.x_max &&
            tracks_[track_id].getY() >= area_.y_min && tracks_[track_id].getY() <= area_.y_max){
                area_tracks_frames_[tracks_[track_id].getId()] += 1;
            }
        printf("\tFrames in area: %d\n", area_tracks_frames_[tracks_[track_id].getId()]);
    }

    // Remove dead tracklets
    removeTracks();
    // Add new tracklets
    addTracks(associated_detections, centroids_x, centroids_y);
}
