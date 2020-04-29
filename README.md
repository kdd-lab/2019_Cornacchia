## Modeling Human Mobility considering Spatial, Temporal and Social Dimensions.


<i>Generative models of human mobility have the purpose of generating realistic trajectories for a set of agents.</i>

The models considered in the thesis exploit two competing mechanisms to describe human mobility: <b>exploration</b> and <b>preferential return</b>. The
exploration mechanism models the scaling law presented by Song et al. (https://www.researchgate.net/publication/47278344_Modelling_the_scaling_properties_of_human_mobility): the tendency to explore new locations decreases over time.
Preferential return reproduces the significant propensity of individuals to return to locations they explored before.

Most generative models focus only on the spatial and temporal dimensions of human mobility.
We propose an extension of <b>GeoSim</b> (https://www.researchgate.net/publication/271855850_Coupling_Human_Mobility_and_Social_Ties), a state-of-the-art model which takes into account the social dimension introducing two mechanisms in addition to the explore and preferential return ones: <b>individual preference</b> and <b>social influence</b>. We include incrementally three mobility mechanisms to improve its modeling capability. In the first extension, we include a mechanism that takes into account the spatial distance between locations. Then, we include a spatial mechanism for considering both the relevance of a location together with the spatial distance. In the last extension, we include a Mobility Diary Generator, a data-driven algorithm able to capture the tendency of individuals to follow a circadian rhythm (https://www.researchgate.net/publication/305471223_Data-driven_generation_of_spatio-temporal_routines_in_human_mobility). We also propose several additional features that can be included in the proposed generative algorithm to model human mobility, also considering other aspects such as the popularity of a node or include a constrain in the location an agent can reach in a given time.

All the presented extension of GeoSim can be instantiated using specific values for the parameters as follow.
