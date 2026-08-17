# Domain

## Context

Our training data represents simulated vehicle runs over drive cycle traces (typically 1hz).
At each point we simulate the vehicle dynamics and get an energy estimation for that point.
We take these point level results and aggregate them up to the trip/road segment level.
Then, the road segments have attributes like total distance, average speed, average road gradiant, time to traverse, etc.

This tree targets a **BEV (battery electric)** vehicle, the 2017 Chevy Bolt. Link energy can be
**negative** (regenerative braking), so the target distribution is asymmetric and heavy-tailed.

### Model Inference Environment

Note that when we're applying these models for inference, we only have limited data (which is why we're developing these models in the first place).
Our inference environment has the following features:

- Average Speed: The speed is given as an average over the link, either from assuming the speed driven is the posted speed, or, using probe data to gather average speeds.
- Average Road Gradient: The gradient is an average over the link, either from taking the elevation difference of the link endpoints, or, using probe data to sample the gradient at subpoints on a link.
- Link Distance: The distance of the link
- Geometry: The link geometry in the well known binary format using the 4326 CRS (latitude and longitude points)

Think about the inference environment as applying these models during a shortest path search in Google Maps where we only have limited information.
If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

## Constraints

Do not include any features that we do not have in our model inference environment.
For example, we do not have acceleration based data when doing model inference and so we do not want our model trained on acceleration data.
That being said, you could consider novel features like the speed on the previous link or average_speed^2.

If you're considering any kind of link sequencing, we will only have the context of the previous links that have been traversed and know nothing about the future links that might be traversed.

Do not filter or remove data points to reduce error. The model must be able to predict all values in the dataset, including extreme energy rates such as heavy regenerative braking. Filtering outliers artificially lowers RMSE without improving the model's actual predictive capability — we need accurate predictions across the full distribution.

Do not include a feature like link position since at inference time, we will not know the position of a link relative to a whole trajectory.
