# Session Seed

Notes and ideas to kickstart this experiment session.
You do not have to follow these instructions explicitly, these are just context for the session to guide you.

## seed notes
Since our link distances are variable, it's possible that a 5 link lookback could span 10m or 1km. 
Let's experiment with creating "virtual" N links that get fed into the CNN where each virtual link covers a fixed distance (say 10m).
If we had 10 virtual links, we would know our lookback would need to be 100m and so when predicting the energy on a current link we could look back 100m and build our virtual links, coming up with an intelligent way map the links onto our virtual links.

Also, remove link position from the feature set since we won't have access to a feature like that in our inference environment.