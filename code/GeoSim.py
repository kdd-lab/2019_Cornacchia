import skmob
import pandas
import numpy
import scipy
import datetime
import powerlaw
import geopandas
from tqdm import tqdm
from igraph import *
from math import sqrt, sin, cos, pi, asin, pow, ceil





'''
Implementation of GeoSim

'''



class GeoSim():
    
    
    def __init__(self, name='GeoSim', rho=0.6, gamma=0.21, alpha=0.2, beta=0.8, tau=17, min_wait_time_hours=1):
        
        self.name = name
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.abstract_space = True
        self.min_wait_time_hours = min_wait_time_hours
        self.agents = {}
        self.lats_lngs = []
        self.distance_matrix = None
        self.map_uid_gid = None
        
        # map_uid_gid is a dataframe with the columns user id || graph id
        # where graph id is an integer in [0, n_agents) and uid the user is the id of the agent
        # and it is considered a string
        
    
    '''
    Location selection methods
    '''
    
    #return the graph id of the node (internal representation) from the UID
    def uid_2_gid(self, uid):
        return self.map_uid_gid[self.map_uid_gid['user_id']==uid].iloc[0]['graph_id']
        
    def gid_2_uid(self, gid):
        return self.map_uid_gid[self.map_uid_gid['graph_id']==gid].iloc[0]['user_id']
        
    def make_social_choice(self, agent, mode):
        
        ''' 
        
        The agent A makes a social choice in this way:

        1. select a contact C with probability proportional to the mobility similarity between A and C.
        2. filter the locations of C to make them feasible according to the mode (exp or ret) of A.
        3. select one of the feasible location of C proportionally to C's visitation frequency

        '''
        
        contact_sim = []

        # the weights in this case are the neighbour's degree
        if self.degree_exp_social and mode == 'exp':
            for ns in self.social_graph.neighbors(agent):
                degree = self.social_graph.degree(ns)
                contact_sim.append(degree)
        
        else:
            #check and update the mobility similarity if 'expired'
        
            for ns in self.social_graph.neighbors(agent):
                eid = self.social_graph.get_eid(agent,ns)

                if self.social_graph.es(eid)['next_update'][0] <= self.current_date:
                    #update
                    lv1 = self.agents[agent]['location_vector']
                    lv2 = self.agents[ns]['location_vector']
                    self.social_graph.es(eid)['mobility_similarity'] = self.cosine_similarity(lv1,lv2)
                    self.social_graph.es(eid)['next_update'] = self.current_date + datetime.timedelta(hours=self.dt_update_mobSim)


                contact_sim.append(self.social_graph.es(eid)['mobility_similarity'][0])

        contact_sim = numpy.array(contact_sim)

        if len(contact_sim)!=0 :
            if numpy.sum(contact_sim)!=0:
                contact_pick = self.random_weighted_choice(contact_sim)
                
            else:
                contact_pick = numpy.random.randint(0, len(contact_sim))

            contact = [i for i in self.social_graph.neighbors(agent)][contact_pick]

        else:
            #no contact in the social network, can not make a social choice
            return -1

        # get the location vectors of the agent and contact
        location_vector_agent = self.agents[agent]['location_vector']
        location_vector_contact = self.agents[contact]['location_vector']

        # compute (vInd) a vector of indices containing all the feasible choice for the agent (depends on the mode)
        # for example, if the agent is in 'exp' mode, we can consider only the locations i such that
        # location_vector_agent[i] == 0, from the moment that i need a new and unvisited location

        if mode == 'exp':
            id_locs_feasible = numpy.where(location_vector_agent==0)[0]
        if mode == 'ret':
            id_locs_feasible = numpy.where(location_vector_agent>=1)[0]
        
        if self.mobility_diary:       
            id_locs_constrain_diary = self.agents[agent]['constrains_mobility_diary']+[self.agents[agent]['home_location']]
            id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary ]
              
        #no choice left for the agent in the current mode
        if len(id_locs_feasible) == 0:
            return -1


        # distance constrain
        if self.distance:
            
            if self.max_speed_km_h is not None:
                src = self.agents[agent]['current_location']
                #getting the row in the distance_matrix relative at the location 'src'
                self.compute_od_row(src)
                distance_row = numpy.array((self.distance_matrix[src].todense())[0])[0]
                max_distance = self.agents[agent]['dt'] * self.max_speed_km_h
                id_locs_reachable = numpy.where(distance_row<=max_distance)[0]
                # compute the indicies of the location feasible & reachable
                id_locs_valid = self.intersect_lists(id_locs_feasible, id_locs_reachable)
            else:
                id_locs_valid = id_locs_feasible
           

            if len(id_locs_valid)==0:
                return -2
        else:
            id_locs_valid = id_locs_feasible

        #project v_location with the indices in id_locs_valid
        v_location_proj = [location_vector_contact[i] for i in id_locs_valid]


        if numpy.sum(v_location_proj) != 0:
            ##wighted choice
            idx = self.random_weighted_choice(v_location_proj)
            location_id = id_locs_valid[idx]
        else:
            if mode == 'ret':
                #assign the most frequent location of the agent, since the contact has 0 visits for every feasible loc.
                location_id = self.make_preferential_choice(agent)
            elif mode == 'exp':
                location_id = self.make_exploration_solo_choice(agent)

        return location_id
          
        
    def make_preferential_choice(self, agent):
        
        ''' 
            The agent A makes a preferential choice selecting a VISITED location
            with probability proportional to the number of visits to that location   
        '''
        
        v_location = self.agents[agent]['location_vector']
        
        # compute the indices of all the feasible locations for the agent A (the visited ones)
        id_locs_feasible = numpy.where(v_location>=1)[0]

        if self.mobility_diary:            
            id_locs_constrain_diary = self.agents[agent]['constrains_mobility_diary']+[self.agents[agent]['home_location']]
            # delete from the feasible locations the ones visited since the last home visit
            # this should be garanteed for free, since the feasible location are the ones such that
            # have at least one visist, and so they are already filtered-out in the building of id_locs_feasible
            id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary ]

        
        
        
        if self.distance:
            if self.max_speed_km_h is not None:
                src = self.agents[agent]['current_location']
                #getting the row in the distance_matrix relative at the location 'src'
                self.compute_od_row(src)
                #distance_row = self.distance_matrix[src]
                distance_row = numpy.array((self.distance_matrix[src].todense())[0])[0]
                max_distance = self.agents[agent]['dt'] * self.max_speed_km_h
                id_locs_reachable = numpy.where(distance_row<=max_distance)[0]
                # compute the indicies of the location feasible & reachable
                id_locs_valid = self.intersect_lists(id_locs_feasible, id_locs_reachable)
            else:
                id_locs_valid = id_locs_feasible 
        
        else:
            id_locs_valid = id_locs_feasible
     
        if len(id_locs_valid)==0:
            return -1

        
        #project v_location with the indices in id_locs_valid
        v_location_proj = [v_location[i] for i in id_locs_valid]
        
        idx = self.random_weighted_choice(v_location_proj)
        location_id = id_locs_valid[idx]
        
        
        return location_id
    
          
    def make_exploration_solo_choice(self, agent):
        '''
            The agent A selects uniformly at random an UNVISITED location if distance == False
            otherwise, starting at location i selects an UNVISITED location j with 
            probability proportional to 1/d_ij or proportional to (r_i * r_j)/ d_ij^2 if gravity == True
        
            output: the location_id if the agent can visit that location 
                    -1 no more UNVISITED locations for the agent
                    -2 there are UNVISITED locations but they aren't reachable with the current wt
        '''
        
        
        v_location = self.agents[agent]['location_vector']   
        
        # compute the indices of all the feasible locations for the agent A (the unvisited ones)
        id_locs_feasible = numpy.where(v_location==0)[0]
        
        if self.mobility_diary:            
            id_locs_constrain_diary = self.agents[agent]['constrains_mobility_diary']+[self.agents[agent]['home_location']]
            # delete from the feasible locations the ones visited since the last home visit
            # this should be garanteed for free, since the feasible location are the ones such that
            # have at least one visist, and so they are already filtered-out in the building of id_locs_feasible
            id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary ]

                        
        if len(id_locs_feasible) == 0:
            return -1
        
        # geosim base (uniformally at random)
        if not self.distance:
            return id_locs_feasible[numpy.random.randint(0, len(id_locs_feasible))]
            
        if self.distance:
            
            src = self.agents[agent]['current_location']
            self.compute_od_row(src)
            distance_row = numpy.array((self.distance_matrix[src].todense())[0])[0]
                   
            if self.max_speed_km_h is not None:
                max_distance = self.agents[agent]['dt'] * self.max_speed_km_h
                id_locs_reachable = numpy.where(distance_row<=max_distance)[0]
                # compute the indicies of the location feasible & reachable
                id_locs_valid = self.intersect_lists(id_locs_feasible, id_locs_reachable)
            else:
                id_locs_valid = id_locs_feasible     

            if len(id_locs_valid) == 0:
                return -2

            #this is made to avoid d/0 
            distance_row[src]=1
            
            if self.gravity:
                #compute (r_i * r_j)/ d_ij^2                
                relevance_src = self.relevances[src]

                distance_row_score = numpy.array(1/distance_row**2)
                
                distance_row_score = distance_row_score * self.relevances * relevance_src
            else:
                #compute 1/d
                distance_row_score = numpy.array(1/distance_row)
               
            distance_row[src]=0
                        
            v_location_proj = numpy.array([distance_row_score[i] for i in id_locs_valid])
            
            #weighted choice
            idx = self.random_weighted_choice(v_location_proj)
            location_id = id_locs_valid[idx]
            
            return location_id
            
    
    def random_weighted_choice(self, weights):
        
        probabilities = weights/numpy.sum(weights)
        t =  numpy.random.multinomial(1, probabilities)
        pos_choice = numpy.where(t==1)[0][0]
        
        return pos_choice

        
    '''
        Initialization methods
    '''
    
    def get_social_factor(self):
        
        if self.alpha >= 0:
            return self.alpha
        else:
            n = numpy.random.normal(0.15,0.1)
            if n>=0:
                return n
            else:
                return 0
    
    
    def init_agents(self):

        for i in range(self.n_agents):

            agent = {
                'ID':i,
                'current_location':-1,
                'home_location':-1,
                'location_vector':numpy.array([0]*self.n_locations),
                'S':0,
                'alpha':self.get_social_factor(),
                'rho':self.rho,
                'gamma':self.gamma,   
                'time_next_move':self.start_date,
                'dt':0,
                'force_move':{'choice':'','tries':0},
                'mobility_diary':None,
                'index_mobility_diary':None,
                'constrains_mobility_diary':[]
            }

            self.agents[i] = agent
    
    
    def init_social_graph(self, mode = 'random'):
        
        #generate a random graph
        if isinstance(mode, str):
            if mode == 'random':
                self.social_graph = (Graph.GRG(self.n_agents, 0.5).simplify())
        
        #edge list (src,dest):
        elif isinstance(mode, list):
            #create self.map_uid_gid
            
            #assuming mode is a list of couple (src,dest)
            user_ids = []
            for edge in mode:
                user_ids.append(edge[0])
                user_ids.append(edge[1])
            
            user_ids = list(set(user_ids))
            graph_ids = numpy.arange(0,len(user_ids))
            
            #update the number of agents n_agents
            self.n_agents = len(user_ids)
            
            self.map_uid_gid = pandas.DataFrame(columns=['user_id','graph_id'])
            
            self.map_uid_gid['user_id'] = user_ids
            self.map_uid_gid['graph_id'] = graph_ids
            
            #create graph
            
            #add vertices
            self.social_graph = Graph()
            self.social_graph.add_vertices(len(user_ids))
            
            #add edges
            
            for edge in mode:
                uid_src = edge[0]
                uid_dest = edge[1]
                             
                gid_src = self.uid_2_gid(uid_src)
                gid_dest = self.uid_2_gid(uid_dest)
                
                e = (gid_src,gid_dest)
                
                self.social_graph.add_edges([e])
            
              
    def assign_starting_location(self, mode='uniform'):

        #For each agent
        for i in range(self.n_agents):

            if mode == 'uniform':                
                #compute a random location
                rand_location = numpy.random.randint(0, self.n_locations)
            if mode == 'relevance':
                #random choice proportional to relevance
                p_location = self.relevances / numpy.sum(self.relevances)
                t = numpy.random.multinomial(1, p_location)
                rand_location = numpy.where(t==1)[0][0]

                         
            #update the location vector of the user
            self.agents[i]['location_vector'][rand_location] = 1
            #set the number of unique location visited to 1 (home)
            self.agents[i]['S'] = 1
            #update currentLocation
            self.agents[i]['current_location'] = rand_location
            #set the home location
            self.agents[i]['home_location'] = rand_location
            
            #update timeNextMove
            if self.mobility_diary:               
                self.agents[i]['time_next_move'] = self.agents[i]['mobility_diary'].loc[1]['datetime']
                self.agents[i]['index_mobility_diary']= 1
                self.agents[i]['dt'] = 1
             
            else:   
                dT = self.get_waiting_time()
                self.agents[i]['time_next_move'] = self.current_date + datetime.timedelta(hours=dT)
                self.agents[i]['dt'] = dT

            if self.map_ids:
                i = self.gid_2_uid(i)

            if self.abstract_space:
                self.trajectories.append((i, rand_location, rand_location, self.current_date))
            else:
                lat = self.lats_lngs[rand_location][0]
                lng = self.lats_lngs[rand_location][1]
                self.trajectories.append((i, lat, lng, self.current_date))

                
    def compute_mobility_similarity(self):
        #compute the mobility similarity from every connected pair of agents
        tot_edge = len(self.social_graph.es)
        
        for edge in self.social_graph.es:
            lv1 = self.agents[edge.source]['location_vector']
            lv2 = self.agents[edge.target]['location_vector']
            self.social_graph.es(edge.index)['mobility_similarity'] = self.cosine_similarity(lv1,lv2)
            self.social_graph.es(edge.index)['next_update'] = self.current_date + datetime.timedelta(hours=self.dt_update_mobSim)

                          
    def cosine_similarity(self,x,y):
        '''Cosine Similarity (x,y) = <x,y>/(||x||*||y||)'''
        num = numpy.dot(x,y)
        den = numpy.linalg.norm(x)*numpy.linalg.norm(y)
        return num/den
         
    
    def get_waiting_time(self):
        
        float_time = powerlaw.Truncated_Power_Law(xmin=self.min_wait_time_hours,
                                            parameters=[1. + self.beta, 1.0 / self.tau]).generate_random()[0]
        return float_time
                                                       
          
    def get_greater_wt(self, agent, max_tries, add_h):
        
        dt = self.get_waiting_time()
        tries = 1
        while dt <= self.agents[agent]['dt'] and tries <= max_tries:
            dt = self.get_waiting_time()
            tries += 1

        # add 20 minutes
        if dt <= self.agents[agent]['dt']:
            dt = self.agents[agent]['dt']+add_h
            
        return dt
            
            
    def store_tmp_movement(self, t, agent, loc, dT):
                                                        
        self.tmp_upd.append({'agent':agent, 'timestamp':t, 'location':loc, 'dT':dT})
        
           
    def update_agent_movement_window(self, to):
                                                        
        # take each tuple in tmp_upd and update the agent info (S locationVector and trajectory)
        # if timestamp <= to
        toRemove=[]
        i=0

        for el in self.tmp_upd:

            if el['timestamp'] <= to:

                agent=int(el['agent'])

                if self.agents[agent]['location_vector'][el['location']] == 0:
                    self.agents[agent]['S']+=1

                self.agents[agent]['location_vector'][el['location']] += 1

                #current location       
                self.agents[agent]['current_location'] = el['location']
                
                if self.map_ids:
                    agent = self.gid_2_uid(agent)
                
                if self.abstract_space:
                    self.trajectories.append((agent, el['location'], el['location'], el['timestamp']))
                else:
                    lat = self.lats_lngs[el['location']][0]
                    lng = self.lats_lngs[el['location']][1]
                    self.trajectories.append((agent, lat, lng, el['timestamp']))

                toRemove.append(i)
            i+=1      

        ##now removing
        toRemove.reverse()

        for ind in toRemove:
            self.tmp_upd.pop(ind)
        
    
    def compute_distance_matrix(self):
        
        self.distance_matrix = numpy.zeros((len(self.spatial_tessellation),len(self.spatial_tessellation)))
        
        for i in range(0,len(self.spatial_tessellation)):
            for j in range(0,len(self.spatial_tessellation)):
                if i != j:
                    d = self.distance_earth_km({'lat':self.lats_lngs[i][0],'lon':self.lats_lngs[i][1]},
                                             {'lat':self.lats_lngs[j][0],'lon':self.lats_lngs[j][1]})
                    self.distance_matrix[i,j] = d
                    
    
    def compute_od_row(self, row):
         
        ## if the "row" is already computed do nothing
        ## i test two column, say column 1 and 0: if they are both zero i'am sure that the row has to be compute
        if self.distance_matrix[row,0] != 0 or self.distance_matrix[row,1] != 0:
            return
            
        for i in range(0,len(self.spatial_tessellation)):
                if i != row:
                    d = self.distance_earth_km({'lat':self.lats_lngs[i][0],'lon':self.lats_lngs[i][1]},
                                             {'lat':self.lats_lngs[row][0],'lon':self.lats_lngs[row][1]})
                    self.distance_matrix[row,i] = d

    
             
    def distance_earth_km(self, src, dest):
                
        lat1, lat2 = src['lat']*pi/180, dest['lat']*pi/180
        lon1, lon2 = src['lon']*pi/180, dest['lon']*pi/180
        dlat, dlon = lat1-lat2, lon1-lon2

        ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2.0) ** 2))
        return 6371.01 * ds
    
    
    def intersect_lists(self, a, b):
        intersection = list(set(a)&set(b))
        intersection.sort()
        intersection = numpy.array(intersection)
        return intersection
     
        
    def init_mobility_diaries(self, hours, start_date):
        
        #For each agent
        for i in range(self.n_agents):          
            diary = self.diary_generator.generate(hours, start_date)
            
            #ensure mobility (at least two checkins)
            while len(diary) < 2:
                diary = self.diary_generator.generate(hours, start_date)
                               
            self.agents[i]['mobility_diary'] = diary
            
            
    def get_current_abstract_location_from_diary(self, agent):
            
            row = self.agents[agent]['index_mobility_diary']
            
            return self.agents[agent]['mobility_diary'].loc[row]['abstract_location']
    
    
    def confirm_action(self, agent, location_id, mode='standard', correction_action=''):
        
        if mode == 'standard':
            if self.mobility_diary:           
                self.agents[agent]['index_mobility_diary']+=1
                row_diary = self.agents[agent]['index_mobility_diary'] 

                if row_diary < len(self.agents[agent]['mobility_diary']):
                    self.agents[agent]['time_next_move'] = self.agents[agent]['mobility_diary'].loc[row_diary]['datetime']
                    delta_T = self.agents[agent]['time_next_move']-self.current_date
                    dT = delta_T.components[0]*24 + delta_T.components[1]
                else:
                    self.agents[agent]['time_next_move'] = self.end_date + datetime.timedelta(hours=1)
                    dT = 1

            else:
                dT = self.get_waiting_time()
                self.agents[agent]['time_next_move']= self.current_date + datetime.timedelta(hours=dT)
                
            self.agents[agent]['dt'] = dT    
            self.agents[agent]['force_move']['choice'] = ''
            self.agents[agent]['force_move']['tries'] = 0
            self.store_tmp_movement(self.current_date, agent, location_id, dT)
                
        elif mode == 'correct':    
            dT = self.get_greater_wt(agent, self.max_tries, 1/3)
            old_dt = self.agents[agent]['dt']
            self.agents[agent]['dt'] = dT    
            self.agents[agent]['time_next_move'] = self.current_date + datetime.timedelta(hours=(dT-old_dt))

            # force the agent to make the same Action
            self.agents[agent]['force_move']['choice'] = correction_action
            self.agents[agent]['force_move']['tries']+= 1
        
        
    def action_correction_diary(self, agent, choice):
        
        if choice == 'RetSocial':
            location_id = self.make_preferential_choice(agent)
            if location_id < 0:
                choice  = 'RetSolo'

        elif choice == 'ExpSocial':
            location_id = self.make_exploration_solo_choice(agent)
            if location_id < 0:
                choice  = 'ExpSolo'

        if choice == 'RetSolo':
            location_id = self.make_exploration_solo_choice(agent)
            if location_id < 0:
                choice  = 'ExpSolo'

        elif choice == 'ExpSolo':
            location_id = self.make_preferential_choice(agent)
            choice  = 'RetSolo'
            if location_id < 0:
                choice  = 'RetSolo'

        if location_id < 0:
            location_id = self.agents[agent]['home_location']
            self.agents[agent]['constrains_mobility_diary'] = []
        else:
            self.agents[agent]['constrains_mobility_diary'].append(location_id)

        return location_id
        
               
    
        
    def generate(self, start_date, end_date, social_graph='random', spatial_tessellation=None, n_agents=500, 
                 n_locations=50, rsl=False, distance_matrix=None, relevance_column=None, distance = False,
                 gravity = False, random_state=None, log_file=None, show_progress=False,dt_update_mobSim = 24*7, 
                 indipendency_window = 0.5, min_relevance = 0.1, max_speed_km_h=None, degree_exp_social=False, diary_generator=None): 
        
        
        if gravity and not distance:
            raise ValueError("distance must be True if gravity is True")
        
        
 
        # init data structures and parameters   
        self.social_graph = None
        self.n_agents = n_agents
        self.agents = {}
        self.tmp_upd = []
        self.distance = distance
        self.gravity = gravity

       
        #update the mobility similarity every dt_update_mobSim hours
        self.dt_update_mobSim = dt_update_mobSim
        self.indipendency_window = indipendency_window
                
        self.n_locations = n_locations
        
        if rsl:            
            self.starting_locations_mode = 'relevance'
        else:
            self.starting_locations_mode = 'uniform'
        
        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date      
        self.map_ids = False
        
        self.trajectories = []
        self.abstract_space = True
        self.spatial_tessellation = []
        self.lats_lngs = []
        self.relevances = []
        self.distance_matrix = None
        
        self.degree_exp_social = degree_exp_social
        self.max_speed_km_h = max_speed_km_h
        
        self.max_tries = 2

        
        if spatial_tessellation is not None:
            
            self.n_locations = len(spatial_tessellation)
            self.spatial_tessellation = spatial_tessellation
            self.abstract_space = False
            self.lats_lngs = self.spatial_tessellation.geometry.apply(skmob.utils.utils.get_geom_centroid, args=[True]).values
            
            
            if self.gravity or self.starting_locations_mode == 'relevance':        
                
                if list(self.spatial_tessellation.columns).count(relevance_column) == 0:
                    raise ValueError("the relevance column is invalid")
                  
                self.relevances = numpy.array(self.spatial_tessellation[relevance_column])
                            
                #map relevance 0 in min_rel               
                self.relevances = numpy.where(self.relevances == 0, min_relevance, self.relevances) 
                
            
            if self.distance:
                self.distance_matrix = scipy.sparse.lil_matrix((len(self.spatial_tessellation),len(self.spatial_tessellation)))
        
        if random_state is not None:
            numpy.random.seed(random_state)

        
        #initialization
        
        if diary_generator is not None:
            self.mobility_diary = True
            self.diary_generator = diary_generator
        else:
            self.mobility_diary = False
            
        delta_T = (self.end_date - self.start_date)
        total_h = delta_T.components[0]*24 + delta_T.components[1]
                    
        #in this case the parameter n_agents is used for the random generation of the social_graph       
        if isinstance(social_graph, str):  
            
            self.map_ids = False
            
            #1. agents
            self.init_agents()
            if self.mobility_diary:
                self.init_mobility_diaries(total_h, self.start_date)
            self.assign_starting_location(mode = self.starting_locations_mode)

            #2. social Graph
            self.init_social_graph(mode = social_graph)
            self.compute_mobility_similarity()
                

        #in this case the parameter n_agents is inferred from the edge list        
        elif isinstance(social_graph, list):
            
            self.map_ids = True
            
            #1. social Graph
            self.init_social_graph(mode = social_graph)         

            #2. agents
            self.init_agents()
            if self.mobility_diary:
                self.init_mobility_diaries(total_h, self.start_date)
            self.assign_starting_location(mode = self.starting_locations_mode)
                              
            #3 mob. similarity
            self.compute_mobility_similarity()

             
        #4. init the progress bar with hours precision
        if show_progress:
            last_t = self.start_date
            pbar = tqdm(total=total_h)        
            elapsed_h = 0
        
    
        while self.current_date < self.end_date:
            
                    
            # we can update all the trajectories made OUTSIDE the indipendence window.       
            sup_indipendency_win = self.current_date - datetime.timedelta(hours=self.indipendency_window)  
            
            self.update_agent_movement_window(sup_indipendency_win)
            
            
            # for every agent in the simulation          
            min_time_next_move = self.end_date
            
            for agent in range(self.n_agents):
                
                location_id = None
                
                #if the user is spending its visiting time do nothing 
                if self.current_date != self.agents[agent]['time_next_move']:
                    if self.agents[agent]['time_next_move'] < min_time_next_move:
                        min_time_next_move = self.agents[agent]['time_next_move']
                    continue
                
                if self.mobility_diary:
                    # check the current abstract location, if it is 0 i can skip all the
                    # location selection phase and return at the Home Location, otherwise i can
                    # use the usual procedure (d-EPR+social choice) with the constrain that i can't visit the
                    # visited location since the last home visis.
                    abstract_location = self.get_current_abstract_location_from_diary(agent)

                    if abstract_location == 0:           
                        location_id = self.agents[agent]['home_location']

                    
                if location_id is None:                 
                    #compute p_exp, the probability that the user will explore a new location           
                    p_exp = self.agents[agent]['rho'] * (self.agents[agent]['S'] ** -self.agents[agent]['gamma'])

                    #generate a random number for the choice: Explore or Return respectively with probability pS^-gamma and 1-pS^-gamma                
                    p_rand_exp = numpy.random.rand()  

                    #generate a random number for the social or solo choice (alpha, 1-alpha)
                    p_rand_soc = numpy.random.rand()

                    #pendent action    
                    p_action = self.agents[agent]['force_move']['choice']
                    
                else:
                    p_action = 'RetHomeDiary'
                
                if p_action == '':
                    # compute which action the agent has to perform
                    if p_rand_exp < p_exp:
                        if p_rand_soc < self.agents[agent]['alpha']:
                            choice = 'ExpSocial'
                        else:
                            choice = 'ExpSolo'
                    else:
                        if p_rand_soc < self.agents[agent]['alpha']:
                            choice = 'RetSocial'
                        else:
                            choice = 'RetSolo'
                                        
                else:
                    choice = p_action
                    

                #EXPLORATION_SOCIAL choice
                if choice == 'ExpSocial':
                    #Select a UNVISITED location based on social contacts                   
                    location_id = self.make_social_choice(agent, 'exp')

                #EXPLORATION_SOLO choice 
                elif choice == 'ExpSolo':
                    #Select a UNVISITED location
                    location_id = self.make_exploration_solo_choice(agent)

                #RETURN_SOCIAL choice
                elif choice == 'RetSocial':
                    #Select a VISITED location based on social contacts
                    location_id = self.make_social_choice(agent, 'ret')

                #RETURN_SOLO choice 
                elif choice == 'RetSolo':
                    #Select a VISITED location
                    location_id = self.make_preferential_choice(agent)
                        
                #location selected correctly (location_id >= 0)            
                if location_id >= 0:             
                    if self.mobility_diary:
                        if abstract_location == 0:
                            self.agents[agent]['constrains_mobility_diary'] = []
                        else:
                            self.agents[agent]['constrains_mobility_diary'].append(location_id)
                          
                    
                #ACTION CORRECTION PHASE
            
                # -1 no selectable location
                # -2 no reachable location
                   
                elif location_id == -1:
                    #no selectable location
                    
                    if self.mobility_diary:
                        location_id = self.action_correction_diary(agent, choice)                                                
                    else:
                        location_id = self.make_preferential_choice(agent)

    
                elif location_id == -2:
                    #in this case the user cannot cover the distance with the waiting time picked
                                       
                    if choice == 'ExpSolo' or 'ExpSocial':                        

                        if self.agents[agent]['force_move']['tries'] < self.max_tries:
                                self.confirm_action(agent, location_id, mode='correct', correction_action='ExpSolo')            
                        else:                    
                            location_id = self.make_preferential_choice(agent)

                    elif choice == 'RetSocial':
                        location_id = self.make_preferential_choice(agent)

                if location_id >= 0:
                    self.confirm_action(agent, location_id)
  
                if self.agents[agent]['time_next_move']< min_time_next_move:
                        min_time_next_move = self.agents[agent]['time_next_move']
                        
                
                    
            #UPDATE THE MODEL

            # now i can update all the movements made by the agents. if the movement are updated during the execution the
            # movement of agent i can influence te movement of agent j (i<j) at the same step 

            
            self.current_date = min_time_next_move
            
            if show_progress:                
                dT2 = self.current_date - last_t   
                if(dT2.components[0]!=0 or dT2.components[1]!=0):
                    pbar.update(dT2.components[0]*24 + dT2.components[1])
                    last_t = self.current_date
                    elapsed_h += dT2.components[0]*24 + dT2.components[1]
        
        if show_progress:
            pbar.update(total_h - elapsed_h)
            pbar.close()
        
        self.update_agent_movement_window(self.end_date)
        tdf = skmob.TrajDataFrame(self.trajectories, user_id=0, latitude=1, longitude=2, datetime=3)
        tdf = tdf.sort_by_uid_and_datetime() 
        return tdf
        