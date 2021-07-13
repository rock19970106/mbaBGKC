
#include <iostream>
#include <math.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>


#include <Eigen/Core>

#include <MBA.h>
#include <UCButils.h>
#include <PointAccessUtils.h>

using Point = pcl::PointXYZ;
using namespace std;
using namespace Eigen;

// // 全局变量

pcl::PointCloud<Point>::Ptr laserCloudTrain;
pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudTrainDS;
pcl::PointCloud<Point>::Ptr laserCloudTest;
pcl::PointCloud<Point>::Ptr laserCloudTestValid;

float grid_size = 0.1;

float predictionKernalSize = 0.2; // predict elevation within x meters

string fileDirectory;

// for downsample
float **minHeight;
float **maxHeight;

bool **initFlag;

int row, column;
int cIndexMin, cIndexMax, rIndexMin, rIndexMax;


void readCloud(int countScan, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    std::string pcd_file = to_string(countScan) + ".pcd";

    printf("  loading %s\n", pcd_file.c_str());

    pcl::PCLPointCloud2 cloud2;

    if (pcl::io::loadPCDFile(fileDirectory + pcd_file, cloud2) == -1)
        throw std::runtime_error("  PCD file not found.");

    fromPCLPointCloud2(cloud2, *cloud);

}

void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) {
    d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
    for (int i = 0; i < xStar.rows(); ++i) {
        d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
    }
}

void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) {
    dist(xStar / (predictionKernalSize + 0.1), xTrain / (predictionKernalSize + 0.1), Kxz);
    Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
           (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * 1.0f;
    // Clean up for values with distance outside length scale, possible because Kxz <= 0 when dist >= predictionKernalSize
    for (int i = 0; i < Kxz.rows(); ++i)
        for (int j = 0; j < Kxz.cols(); ++j)
            if (Kxz(i, j) < 0) Kxz(i, j) = 0;
}

void predictCloudBGK(float &error1) {

    int kernelGridLength = int(predictionKernalSize / grid_size);

    int count1 = 0;
    int valid1 = 0;
    error1 = 0.0;

    for (int i = 0; i < laserCloudTest->points.size(); i++) {

        Point testPoint = laserCloudTest->points[i];

        // Training data
        vector<float> xTrainVec; // training data x and y coordinates
        vector<float> yTrainVecElev; // training data elevation

        int idy = (int) floor(testPoint.y / grid_size) - rIndexMin;
        int idx = (int) floor(testPoint.x / grid_size) - cIndexMin;

        // index out of boundary
        if (idy < 0 || idx < 0 || idy >= row || idx >= column)
            continue;


        // Fill training data (vector)
        for (int m = -kernelGridLength; m <= kernelGridLength; ++m) {
            for (int n = -kernelGridLength; n <= kernelGridLength; ++n) {
                // skip grids too far
                if (std::sqrt(float(m * m + n * n)) * grid_size > predictionKernalSize)
                    continue;
                int idr = idy + m;
                int idc = idx + n;
                // index out of boundary
                if (idr < 0 || idc < 0 || idr >= row || idc >= column)
                    continue;
                // save only observed grid in this scan
                if (initFlag[idr][idc]) {
                    xTrainVec.push_back((cIndexMin + idc) * grid_size + grid_size / 2.0);
                    xTrainVec.push_back((rIndexMin + idr) * grid_size + grid_size / 2.0);
                    yTrainVecElev.push_back(maxHeight[idr][idc]);
                }
            }
        }

        // no training data available, continue
        if (xTrainVec.size() == 0)
            continue;
        // convert from vector to eigen
        Eigen::MatrixXf xTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVec.data(),
                                                                                                 xTrainVec.size() /
                                                                                                 2, 2);
        Eigen::MatrixXf yTrainElev = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
                yTrainVecElev.data(), yTrainVecElev.size(), 1);

        // Test data (current grid)
        vector<float> xTestVec;
        xTestVec.push_back(testPoint.x);
        xTestVec.push_back(testPoint.y);
        Eigen::MatrixXf xTest = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVec.data(),
                                                                                                xTestVec.size() / 2,
                                                                                                2);
        // Predict
        Eigen::MatrixXf Ks; // covariance matrix
        covSparse(xTest, xTrain, Ks); // sparse kernel

        Eigen::MatrixXf ybarElev = (Ks * yTrainElev).array();
        Eigen::MatrixXf kbar = Ks.rowwise().sum().array();

        // Update Elevation with Prediction
        if (std::isnan(ybarElev(0, 0)) || std::isnan(kbar(0, 0)))
            continue;

        if (kbar(0, 0) == 0)
            continue;

        float elevation = ybarElev(0, 0) / kbar(0, 0);

        error1 += (elevation - testPoint.z) * (elevation - testPoint.z);

        laserCloudTestValid->points.push_back(laserCloudTest->points[i]);

        if (fabs(elevation - testPoint.z) < 0.02)
            valid1++;

        count1++;

    }

    error1 = sqrt(error1 / count1);

//    cout << "count1:  " << count1 << endl;
//    cout << "valid1: " << valid1 << endl;
//    cout << "error1:  " << error1 << endl;
//
//    cout << "----------------------------------------------------------------" << endl;

}


int main() {

    fileDirectory = "/home/zlp/catkin_ws/src/mba_traversability_mapping/slope_pcd_2/";

    string errorBGKPath = "errorBGK_slope_2.txt";

    string errorBsplinePath = "errorBspline_slope_2.txt";

    string timeBGKPath = "timeBGK_slope_2.txt";

    string timeBsplinePath = "timeBspline_slope_2.txt";

    laserCloudTrain.reset(new pcl::PointCloud<Point>);
    laserCloudTrainDS.reset(new pcl::PointCloud<Point>);
    laserCloudTest.reset(new pcl::PointCloud<Point>);
    laserCloudTestValid.reset(new pcl::PointCloud<Point>);

    //downSizeFilter
    pcl::VoxelGrid<Point> downSizeFilter;

    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

    // ############################################################################
    // load point cloud
    //--------------加载点云-----------------------------

    for (int countScan = 1; countScan <= 568; countScan++) {

        int countTrain = 0;
        int countTest = 0;

        for (int i = 0; i < 2; i++) {
            pcl::PointCloud<Point>::Ptr laserCloudTemp(new pcl::PointCloud<Point>);
            readCloud(countScan + i, laserCloudTemp);
            *laserCloudTrain += *laserCloudTemp;
        }
        readCloud(countScan + 2, laserCloudTest);

        laserCloudTrainDS->clear();

        downSizeFilter.setInputCloud(laserCloudTrain);
        downSizeFilter.filter(*laserCloudTrainDS);

        cout << laserCloudTrainDS->points.size() << endl;


        // ############################################################################

        pcl::console::TicToc time1;
        time1.tic();

        pcl::PointXYZ minPt;//用于存放三个轴的最小值
        pcl::PointXYZ maxPt;//用于存放三个轴的最大值
        pcl::getMinMax3D(*laserCloudTrain, minPt, maxPt);

        cIndexMin = (int) floor(minPt.x / grid_size);
        cIndexMax = (int) floor(maxPt.x / grid_size);
        rIndexMin = (int) floor(minPt.y / grid_size);
        rIndexMax = (int) floor(maxPt.y / grid_size);

//        cout << cIndexMin << endl;
//        cout << cIndexMax << endl;
//        cout << rIndexMin << endl;
//        cout << rIndexMax << endl;

        column = cIndexMax - cIndexMin + 1;
        row = rIndexMax - rIndexMin + 1;

//        cout << "column: " << column << endl;
//        cout << "row: " << row << endl;

        minHeight = new float *[row];
        for (int i = 0; i < row; ++i)
            minHeight[i] = new float[column];

        maxHeight = new float *[row];
        for (int i = 0; i < row; ++i)
            maxHeight[i] = new float[column];

        initFlag = new bool *[row];
        for (int i = 0; i < row; ++i)
            initFlag[i] = new bool[column];

        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < column; ++j) {
                initFlag[i][j] = false;
            }
        }

        int idx, idy;

        for (int m = 0; m < laserCloudTrain->size(); m++) {

            idy = (int) floor(laserCloudTrain->points[m].y / grid_size) - rIndexMin;
            idx = (int) floor(laserCloudTrain->points[m].x / grid_size) - cIndexMin;

            if (initFlag[idy][idx] == false) {
                minHeight[idy][idx] = laserCloudTrain->points[m].z;
                maxHeight[idy][idx] = laserCloudTrain->points[m].z;
                initFlag[idy][idx] = true;
            } else {
                minHeight[idy][idx] = std::min(minHeight[idy][idx], laserCloudTrain->points[m].z);
                maxHeight[idy][idx] = std::max(maxHeight[idy][idx], laserCloudTrain->points[m].z);
            }
        }

        // ############################################################################
        // Bayesian Generalized Kernel Inference

        cout << "******************************Bayesian Generalized Kernel Inference**********************************"
             << endl;

        float error1 = 0.0;

        predictCloudBGK(error1);

//        cout << "----------------------------------------------------------------" << endl;
//        cout << "Applied " << "Bayesian Generalized Kernel Inference in " << time1.toc() / 1000 << " s" << endl;

        double timeUsed1 = time1.toc() / 1000.0;

        std::ofstream foutTG(timeBGKPath, std::ios::app);

        foutTG << timeUsed1 << endl;

        foutTG.close();

        std::ofstream foutG(errorBGKPath, std::ios::app);

        foutG << error1 << endl;

        foutG.close();


        // ############################################################################
        // Multilevel Bspline surface fitting

        cout << "**********************************Multilevel Bspline surface fitting********************************"
             << endl;

        pcl::console::TicToc time2;
        time2.tic();

        typedef std::vector<double> dVec;
        boost::shared_ptr<dVec> x_arr(new std::vector<double>);
        boost::shared_ptr<dVec> y_arr(new std::vector<double>);
        boost::shared_ptr<dVec> z_arr(new std::vector<double>);

        double min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10;

        for (int i = 0; i < laserCloudTrainDS->points.size(); i++) {

            float x = laserCloudTrainDS->points[i].x;
            float y = laserCloudTrainDS->points[i].y;
            float z = laserCloudTrainDS->points[i].z;

            if (!std::isnan(x) && !std::isnan(y) && !std::isnan(z)) {
                x_arr->push_back(static_cast<double>(x));
                y_arr->push_back(static_cast<double>(y));
                z_arr->push_back(static_cast<double>(z));
                if (x < min_x)
                    min_x = x;
                if (x > max_x)
                    max_x = x;
                if (y < min_y)
                    min_y = y;
                if (y > max_y)
                    max_y = y;
            }
        }

//        cout << "min_x:" << min_x << " " << "max_x:" << max_x << endl;
//        cout << "min_y:" << min_y << " " << "max_y:" << max_y << endl;
//
//        cout << "----------------------------------------------------------------" << endl;


        MBA mba(x_arr, y_arr, z_arr);

        // Create spline surface.
        mba.MBAalg(1, 1, 10);

        UCBspl::SplineSurface surf = mba.getSplineSurface();

        // ############################################################################
        // Accuracy Verification
        //------------------------------------------------------

        double xr, yr, zr, zs;

        int count2 = 0;
        int valid2 = 0;
        double error2 = 0.0;

        pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudInvalid(new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 0; i < laserCloudTestValid->points.size(); i++) {

            xr = static_cast<double >(laserCloudTestValid->points[i].x);
            yr = static_cast<double>(laserCloudTestValid->points[i].y);
            zr = static_cast<double>(laserCloudTestValid->points[i].z);

            if (xr > min_x && xr < max_x && yr > min_y && yr < max_y) {

                zs = surf.f(xr, yr);

                error2 += (zs - zr) * (zs - zr);

                if (fabs(zs - zr) < 0.02) {
                    valid2++;
                } else {
                    Point invalidPoint;
                    invalidPoint.x = laserCloudTestValid->points[i].x;
                    invalidPoint.y = laserCloudTestValid->points[i].y;
                    invalidPoint.z = laserCloudTestValid->points[i].z;
                    laserCloudInvalid->points.push_back(invalidPoint);
                }
                count2++;
            }
        }

        error2 = sqrt(error2 / count2);

//        cout << "count2:  " << count2 << endl;
//
//        cout << "valid2:  " << valid2 << endl;
//
//        cout << "error2:  " << error2 << endl;


        double timeUsed2 = time2.toc() / 1000.0;

        std::ofstream foutTS(timeBsplinePath, std::ios::app);

        foutTS << timeUsed2 << endl;

        foutTS.close();

        std::ofstream foutS(errorBsplinePath, std::ios::app);

        foutS << error2 << endl;

        foutS.close();

//        cout << "----------------------------done------------------------------------" << endl;
//        cout << "Applied " << "Multilevel B-spline Surface Fitting in " << time2.toc() / 1000.0 << " s" << endl;

        laserCloudTrain->clear();
        laserCloudTrainDS->clear();
        laserCloudTest->clear();
        laserCloudTestValid->clear();
        laserCloudInvalid->clear();

    }

    return 0;
}