# Metformer

A machine learning application designed around working with 4 dimension atmospheric
data.

Assumptions:
Length, Width, Lat, Lon, X, Y are all interchangeable terms. In most cases
Length and Width will be used.

Time

## Terminology and Tensor/Array Structure

Discrete data is a numerical type of data that includes whole, concrete numbers
with specific and fixed data values determined by counting. Continuous data
includes complex numbers and varying data values measured over a particular
time interval. A simple way to describe the difference between the two is to
visualize a scatter plot graph versus a line graph.

Continuous data is data that can take on any value within a specified range
(which may be infinite) while discrete data can only take on certain values.


Image:
`Tensor[[Length, Width]]`
> assumed to be a 2D array of values that share a common spatial and temporal
> domain. The values are assumed to be continuous and not discrete.

ChannelFeature:
`Tensor[[Channel, Length, Width]]`
> assumed to be a 3D array of values that share a common spatial and temporal
> domain. In many ViT applications Channel is represented and RGB values. The
> in this space we look to stack various channels of data that share a common
> spatial and temporal domain.

TimeSeriesFeature:
`Tensor[[Time, Length, Width]]`
> assumed to be a 3D array of values that share a common spatial and temporal
> domain. The steps in time should be equal across the entire array.

Feature:
`Tensor[[Channel | Time, Length, Width]]`
> assumed to be a 3D array of values that share a common spatial or temporal
> domain. The steps in time should be equal across the entire array.

Sample:
`Tensor[[Channel, Time, Length, Width]]`
> assumed to be a 4D array of values that share a common spatial and temporal
> domain. The steps in time should be equal across the entire array.

Batch:
`Tensor[[Batch, Channel, Time, Length, Width]]`
> assumed to be a 5D array of values that share a common spatial and temporal
> domain. The steps in time should be equal across the entire array.

```python
BatchFeatures = Array[[Batch, Channel, Time, Length, Width], ...]
Samples = Array[[Channel, Time, Length, Width], ...]
TimeSeriesFeature = Array[[Time, Length, Width], ...]
ChannelFeature = Array[[Channel, Length, Width], ...]
Image = Array[[Length, Width], ...]

# 
batch = get_batch(...) # type: Array[[Batch, Channel, Time, Length, Width], ...]
samps = batch[0] # type: Array[[Channel, Time, Length, Width], ...]
times = batch[0, 0] # type: Array[[Time, Length, Width], ...]
chans = batch[0, :, 0] # type: Array[[Channel, Length, Width], ...]
image = batch[0, 0, 0] # type: Array[[Length, Width], ...]
```
