# msckf_mono
Monocular MSCKF with ROS Support

# Requirements
- ROS Kinetic with Boost, OpenCV and Eigen
- https://github.com/uzh-rpg/fast build and install according to their instructions

# Euroc Dataset -- ROS Bag
Download MH_03_medium.bag from into the euroc folder in this repository.

```
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.bag
mv MH_03_medium.bag <path_to_msckf_mono>/euroc/.
```

Now run the MSCKF on this sequence
```
roslaunch msckf_mono euroc.launch
```

RViz will come up by default showing the odometry and image with tracks.


# Euroc Dataset -- ASL Format
Download one (or more) of the datasets from https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets in the ASL dataset format
Place
```
%YAML:1.0
```
At the top of each YAML file. Currently OpenCV is being used to load these YAMLs and expects this header to exist.

The stand_still_end is going to be the time just before the quad takes off for the actual sequence--take care to find this before starting the MSCKF.

Now you can run the MSCKF on the individual sequence
```
roslaunch msckf_mono asl_msckf.launch data_set_path:=<directory of mav0 inside of sequence> stand_still_end:=<time to start at with dot at the end>
```

RViz will come up by default and display the image with tracks on the left and the generated path and map on the right.

![Machine Hall 03 Medium](https://github.com/daniilidis-group/msckf_mono/raw/master/euroc/MH03.png)

The two paths shown, green is ground truth and red is from the MSCKF.

# MSCKF

The actual MSCKF is fully templated based on the floating point type that you want. It should be easy to compile for applications that see real speedups from smaller floating point sizes.

We have run this on platforms ranging from the odroid to a modern laptop, so hopefully it should work on whatever device you want.

# Used in
- The Euroc dataset was evaluated in http://rpg.ifi.uzh.ch/docs/ICRA18_Delmerico.pdf
- The core MSCKF was used in http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Event-Based_Visual_Inertial_CVPR_2017_paper.pdf

# TODO
- ROS Nodelet
- Remove OpenCV from opening YAML files
- PennCOSYVIO Dataset support
