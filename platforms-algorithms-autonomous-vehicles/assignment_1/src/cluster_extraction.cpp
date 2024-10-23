#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include "../include/Renderer.hpp"
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <chrono>
#include <unordered_set>
#include "../include/tree_utilities.hpp"
#include <boost/filesystem.hpp>

#define USE_PCL_LIBRARY
using namespace lidar_obstacle_detection;

typedef std::unordered_set<int> my_visited_set_t;
using namespace std;

//This function sets up the custom kdtree using the point cloud
void setupKdtree(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, int dimension)
{
    //insert point cloud points into tree
    for (int i = 0; i < cloud->size(); ++i)
    {
        tree->insert({cloud->at(i).x, cloud->at(i).y, cloud->at(i).z}, i);
    }
}

/*
OPTIONAL
This function computes the nearest neighbors and builds the clusters
    - Input:
        + cloud: Point cloud to be explored
        + target_ndx: i-th point to visit
        + tree: kd tree for searching neighbors
        + distanceTol: Distance tolerance to build the clusters 
        + visited: Visited points --> typedef std::unordered_set<int> my_visited_set_t;
        + cluster: Here we add points that will represent the cluster
        + max: Max cluster size
    - Output:
        + visited: already visited points
        + cluster: at the end of this function we will have one cluster
*/
void proximity(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int target_ndx, my_pcl::KdTree* tree, float distanceTol, my_visited_set_t& visited, std::vector<int>& cluster, int max)
{
	if (cluster.size() < max)
    {
        cluster.push_back(target_ndx);
        visited.insert(target_ndx);

        std::vector<float> point {cloud->at(target_ndx).x, cloud->at(target_ndx).y, cloud->at(target_ndx).z};
    
        // get all neighboring indices of point
        std::vector<int> neighborNdxs = tree->search(point, distanceTol);

        for (int neighborNdx : neighborNdxs)
        {
            // if point was not visited
            if (visited.find(neighborNdx) == visited.end())
            {
                proximity(cloud, neighborNdx, tree, distanceTol, visited, cluster, max);
            }

            if (cluster.size() >= max)
            {
                return;
            }
        }
    }
}

/*
OPTIONAL
This function builds the clusters following a euclidean clustering approach
    - Input:
        + cloud: Point cloud to be explored
        + tree: kd tree for searching neighbors
        + distanceTol: Distance tolerance to build the clusters 
        + setMinClusterSize: Minimum cluster size
        + setMaxClusterSize: Max cluster size
    - Output:
        + cluster: at the end of this function we will have a set of clusters
TODO: Complete the function
*/
std::vector<pcl::PointIndices> euclideanCluster(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, my_pcl::KdTree* tree, float distanceTol, int setMinClusterSize, int setMaxClusterSize)
{
	my_visited_set_t visited{};                                                          //already visited points
	std::vector<pcl::PointIndices> clusters;                                             //vector of PointIndices that will contain all the clusters
    std::vector<int> cluster;                                                            //vector of int that is used to store the points that the function proximity will give me back
	
    //for every point of the cloud
    for(int i = 0; i < cloud->size(); ++i){
        //  if the point has not been visited (use the function called "find")
        if(visited.find(i) == visited.end()){
            cluster.clear();

            // find clusters using the proximity function
            proximity(cloud, i, tree, distanceTol, visited, cluster, setMaxClusterSize);

            //if we have more clusters than the minimum
            if(cluster.size() >= setMinClusterSize){
                //Create the cluster and insert it in the vector of clusters. You can extract the indices from the cluster returned by the proximity funciton (use pcl::PointIndices)
                pcl::PointIndices cluster_indices;
                cluster_indices.indices = cluster;
                clusters.push_back(cluster_indices); 
            }
        }
    }

	return clusters;	
}

void ProcessAndRenderPointCloud (Renderer& renderer, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    std::vector<Color> colors = {Color(1,0,0), Color(1,1,0), Color(0,0,1), Color(1,0,1), Color(0,1,1), Color(0,1,0)};

    // TODO: 1) Downsample the dataset 
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.13f, 0.13f, 0.13f); //this value defines how much the PC is filtered
    sor.filter (*cloud_filtered);
    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
    << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;
    //DEBUG
    //renderer.RenderPointCloud(cloud,"cloudFiltered",colors[1]);

    // 2) here we crop the points that are far away from us, in which we are not interested
    pcl::CropBox<pcl::PointXYZ> cb(true);
    cb.setInputCloud(cloud_filtered);
    cb.setMin(Eigen::Vector4f (-20, -6, -2, 1));
    cb.setMax(Eigen::Vector4f ( 30, 7, 5, 1));
    cb.filter(*cloud_filtered); 

    // TODO: 3) Segmentation and apply RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.3); // determines how close a point must be to the model in order to be considered an inlier
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size () == 0){
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        return;
    }
    

    // TODO: 4) iterate over the filtered cloud, segment and remove the planar inliers 
    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    // Extract the inliers (here we extract the points of the plane moving the indices representing the plane to cloud_segmented)
    extract.setInputCloud (cloud_filtered); 
    
    //PCL defines a way to define a region of interest / list of point indices that the algorithm should operate on, rather than the entire cloud, via setIndices.
    extract.setIndices (inliers);
    extract.setNegative (false); // Retrieve indices to all points in cloud_filtered but only those referenced by inliers:
    extract.filter (*cloud_plane);   // We effectively retrieve JUST the plane
    renderer.RenderPointCloud(cloud_plane,"cloudPlane",colors[2]);

    extract.setNegative(true); //si mantengono tutti punti non appartenenti al piano
    extract.filter(*cloud_filtered); //rimozione del piano dalla scena

    // TODO: 5) Create the KDTree and the vector of PointIndices
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered); 

    // TODO: 6) Set the spatial tolerance for new cluster candidates (pay attention to the tolerance!!!)
    std::vector<pcl::PointIndices> cluster_indices;

    #ifdef USE_PCL_LIBRARY
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        //Set the spatial tolerance for new cluster candidates
        //If you take a very small value, it can happen that an actual object can be seen as multiple clusters. On the other hand, if you set the value too high, it could happen, that multiple objects are seen as one cluster
        ec.setClusterTolerance (0.4); 

        //We impose that the clusters found must have at least setMinClusterSize() points and maximum setMaxClusterSize() points
        ec.setMinClusterSize (30);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered);
        ec.extract (cluster_indices);
    #else
        // Optional assignment
        my_pcl::KdTree treeM;
        treeM.set_dimension(3);
        setupKdtree(cloud_filtered, &treeM, 3);
        cluster_indices = euclideanCluster(cloud_filtered, &treeM, 0.3, 30, 25000);
    #endif

    /**Now we extracted the clusters out of our point cloud and saved the indices in cluster_indices. 

    To separate each cluster out of the vector<PointIndices> we have to iterate through cluster_indices, create a new PointCloud for each entry and write all points of the current cluster in the PointCloud.
    Compute euclidean distance
    **/
    int j = 0;
    int clusterId = 0;
    
    //posizione del veicolo all'interno della scena
    pcl::PointXYZ egoPos(0.0, 0.0, 0.0);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            cloud_cluster->push_back ((*cloud_filtered)[*pit]); 
        }
        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // TODO: 7) render the cluster and plane without rendering the original cloud 
        //Here we create the bounding box on the detected clusters
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);

        //TODO: 8) Here you can plot the distance of each cluster w.r.t ego vehicle
        Box box{minPt.x, minPt.y, minPt.z,
        maxPt.x, maxPt.y, maxPt.z};

        //calcolo dell'angolo del box più vicino al mio veicolo che si trova in posizione 0,0,0
        float closestX, closestY, closestZ;
        if(abs(minPt.x) > abs(maxPt.x)) closestX = maxPt.x;
        else closestX = minPt.x;
        if(abs(minPt.y) > abs(maxPt.y)) closestY = maxPt.y;
        else closestY = minPt.y;
        if(abs(minPt.z) > abs(maxPt.z)) closestZ = maxPt.z;
        else closestZ = minPt.z;
        pcl::PointXYZ closestAngle(closestX, closestY, closestZ);
        float distance = sqrt(pow(egoPos.x - closestAngle.x, 2) + pow(egoPos.y - closestAngle.y, 2) + pow(egoPos.z - closestAngle.z, 2));

        //stampa testo distanza sul box del cluster
        stringstream ss;
        ss<<fixed<<setprecision(2)<<distance;
        string distanceText = ss.str()+"m";
        pcl::PointXYZ posText;
        posText.x = (minPt.x + maxPt.x)/2;
        posText.y = (minPt.y + maxPt.y)/2; 
        posText.z = maxPt.z;
        renderer.addText(posText.x, posText.y, posText.z, distanceText);
        
        //DEBUG: ho modificato anche Renderer.cpp e Renderer.hpp
        //renderer.addLine(closestAngle, egoPos, j);

        //TODO: 9) Here you can color the vehicles that are both in front and 5 meters away from the ego vehicle
        //please take a look at the function RenderBox to see how to color the box
        if(distance < 5 && (minPt.x > 0 || maxPt.x > 0)){
            renderer.RenderBox(box, j, colors[0]);
        }
        else
            renderer.RenderBox(box, j, colors[5]);

        ++clusterId;
        j++;
    }  

}


int main(int argc, char* argv[])
{
    Renderer renderer;
    renderer.InitCamera(CameraAngle::XY);
    // Clear viewer
    renderer.ClearViewer();

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    std::vector<boost::filesystem::path> stream(boost::filesystem::directory_iterator{"../dataset_1"},
                                                boost::filesystem::directory_iterator{});

    // sort files in ascending (chronological) order
    std::sort(stream.begin(), stream.end());

    auto streamIterator = stream.begin();

    while (not renderer.WasViewerStopped())
    {
        renderer.ClearViewer();

        pcl::PCDReader reader;
        reader.read (streamIterator->string(), *input_cloud);
        auto startTime = std::chrono::steady_clock::now();

        ProcessAndRenderPointCloud(renderer,input_cloud);
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "[PointCloudProcessor<PointT>::ReadPcdFile] Loaded "
        << input_cloud->points.size() << " data points from " << streamIterator->string() <<  "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

        streamIterator++;
        if(streamIterator == stream.end())
            streamIterator = stream.begin();

        renderer.SpinViewerOnce();
    }
}
