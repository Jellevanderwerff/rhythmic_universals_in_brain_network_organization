# Load the package
library(brainGraph)

# Access the dk atlas data
dk_data <- destrieux.scgm

# Display the MNI coordinates
mni_coords <- dk_data[, c("name", "x.mni", "y.mni", "z.mni")]

# Print the first few rows to verify
head(mni_coords)

# Save the MNI coordinates to a CSV file
write.csv(mni_coords, file = "dk_mni_coordinates.csv", row.names = FALSE)

# Check which atlases include cerebellum regions
# The AAL atlases typically include cerebellum

# First check the AAL116 atlas which has cerebellum regions
aal_data <- aal116

# Filter for cerebellum regions
cerebellum_regions <- aal_data[aal_data$lobe == "Cerebellum", ]

# Display the cerebellar regions
head(cerebellum_regions)

# Get just the MNI coordinates for cerebellum
cerebellum_coords <- cerebellum_regions[, c("name", "x.mni", "y.mni", "z.mni")]

# Create data frame with exactly one point for left cerebellum and one for right cerebellum
cerebellum_points <- data.frame(
  name = c("Left_Cerebellum", "Right_Cerebellum"),
  x.mni = c(-36.067, 37.456),     # Using CRUS1.L and CRUS1.R x-coordinates
  y.mni = c(-66.72, -67.137),     # Using CRUS1.L and CRUS1.R y-coordinates
  z.mni = c(-28.934, -29.547)     # Using CRUS1.L and CRUS1.R z-coordinates
)

# Combine with your dk coordinates
all_coords <- cerebellum_points

# Save the cerebellum coordinates to a CSV file
write.csv(all_coords, file = "cerebellum_mni_coordinates.csv", row.names = FALSE)

