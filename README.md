## Modeling Human Mobility considering Spatial, Temporal and Social Dimensions.


Generative models of human mobility have the purpose of generating realistic trajectories for a set of agents.

The models considered exploit two competing mechanisms to describe human mobility: exploration and preferential return. The
exploration mechanism models the scaling law presented by Song et al.: the tendency to explore new locations decreases over time.
Preferential return reproduces the significant propensity of individuals to return to locations they explored before \cite{im_model_song,ditras,d_epr}.
An agent explores a new location with probability $P_{exp}=\rho S^{-\gamma}$, or returns to a previously visited location with a complementary probability $P_{ret}=1-\rho S^{-\gamma}$, where $S$ is the number of unique locations visited by the individual and $\rho=0.6$, $\gamma=0.21$ are constants \cite{im_model_song}.
When the agent returns, the location is selected with a probability proportional to its location frequency (Equation

Most generative models focus only on the spatial and temporal dimensions of human mobility.
We propose an extension of GeoSim, a state-of-the-art model which takes into account the social dimension introducing two mechanisms in addition to the explore and preferential return ones: individual preference and social influence. We include incrementally three mobility mechanisms to improve its modeling capability. In the first extension, we include a mechanism that takes into account the spatial distance between locations. Then, we include a spatial mechanism for considering both the relevance of a location together with the spatial distance. In the last extension, we include a Mobility Diary Generator, a data-driven algorithm able to capture the tendency of individuals to follow a circadian rhythm. We also propose several additional features that can be included in the proposed generative algorithm to model human mobility, also considering other aspects such as the popularity of a node or include a constrain in the location an agent can reach in a given time.
