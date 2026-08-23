int32 class_id
string class_name
float32 confidence
float32 x_min
float32 y_min
float32 x_max
float32 y_max

# Binary mono8 mask for the bounding-box ROI, stored row-major.
uint32 mask_x
uint32 mask_y
uint32 mask_width
uint32 mask_height
uint8[] mask
