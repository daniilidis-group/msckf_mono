# msckf_mono
Monocular MSCKF ROS Node

# Euroc Dataset
Build this project inside of a ROS workspace

```
roslaunch msckf_mono asl_mskcf.launch data_set_path:=<directory of mav0 inside of sequence> stand_still_end:=<time to start at with dot at the end>
```

We have run this on platforms ranging from the odroid to a modern laptop, so hopefully it should work on whatever device you want.

# TODO
- ROS Node and Nodelet
- PennCOSYVIO Dataset support

# MSCKF

The actual MSCKF is fully templated based on the floating point type that you want. It should be easy to compile for applications that see real speedups from smaller floating point sizes.
