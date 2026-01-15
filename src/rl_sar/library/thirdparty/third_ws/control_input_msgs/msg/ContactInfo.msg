# Contact information message for quadruped robot

std_msgs/Header header

# Estimated contact forces (12 values: 3 per foot in x,y,z)
float64[12] est_contact_forces

# Contact flags from different sources (4 values: one per foot)
bool[4] feet_contact      # From foot force sensors
bool[4] gait_contact      # From gait scheduler
bool[4] est_contact       # From torque estimation

# Magnitude of estimated forces (4 values: one per foot)
float64[4] est_force_magnitude

# Z-component of estimated forces (4 values: one per foot)
float64[4] est_force_z
