# ELEC3442 - Adaptive Schedule Design
## Workflow
1. Full-scale data pre-processing (data-preprocessing.ipynb) to extract the sleeping records and categorize them by dates and weekdays.
2. Routine learning (routine-learning.ipynb) to extract the features and learn the routine of the user by the sleeping data.
    - The sleeping routines are clustered by K-means into 5 clusters, representing 5 different sleeping patterns and potential outcome of the day. It should be manually labeled by the user to indicate the meaning of each cluster, which can be used for future prediction and schedule design.
3. The RasPerry Pi captures the user's current sleeping pattern and predicts the potential outcome of the day according to the pre-trained model.
4. The outcome cluster label is fed into an external LLM along with the user's TODO list/schedule for the day, and the LLM will give suggestions on how to adjust the schedule for the day according to the predicted outcome.

## Advantages
1. Useful: The adaptive schedule can help users to better manage their time and energy, and improve their productivity and well-being.
2. Personalized: The schedule is tailored to the user's specific routine and needs, rather than a one-size-fits-all approach.
3. Private: the raw sensitive data is not uploaded to the LLM, but only processed by the edge models, and the LLM will only receive the cluster label and the schedule, which is more abstract and less sensitive.

## Finished part
- Data pre-processing
- K-means model training

## To be done
- RasPerry Pi utilization (using SenseHAT / Pi Camera to capture the sleeping pattern and other data)
- LLM prompt engineering
- ...