# msckf_mono
Monocular MSCKF with ROS Support

# Euroc Dataset
Build this project inside of a ROS workspace

```
roslaunch msckf_mono asl_mskcf.launch data_set_path:=<directory of mav0 inside of sequence> stand_still_end:=<time to start at with dot at the end>
```

RViz will come up by default and display the image with tracks on the left and the generated path and map on the right.

![Machine Hall 05 Difficult](https://github.com/daniilidis-group/msckf_mono/raw/master/EurocMH05.png)

The two paths shown, green is ground truth and red is from the MSCKF.

# MSCKF

The actual MSCKF is fully templated based on the floating point type that you want. It should be easy to compile for applications that see real speedups from smaller floating point sizes.

We have run this on platforms ranging from the odroid to a modern laptop, so hopefully it should work on whatever device you want.

# TODO
- ROS Node and Nodelet
- PennCOSYVIO Dataset support
