import numpy as np

def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError(caller + ' must be implemented in subclass')

class Aggregator(object):
    
    def __init__(self, domain_labels, directed = False):
        self.domain_labels = domain_labels # The list of labels in the domain
        self.directed = directed # Whether we should use edge directions for creating the aggregation
    
    def aggregate(self, graph, node, conditional_node_to_label_map):
        ''' Given a node, its graph, and labels to condition on (observed and/or predicted)
        create and return a feature vector for neighbors of this node.
        If a neighbor is not in conditional_node_to_label_map, ignore it.
        If directed = True, create and append two feature vectors;
        one for the out-neighbors and one for the in-neighbors.
        '''
        abstract()

class CountAggregator(Aggregator):
    '''The count aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map): 
        initial_list=[0.0]
        #directed graph case
        list_size = len(self.domain_labels)
        
        if self.directed:
            #make two lists then join
            in_list = list_size * initial_list
            out_list = list_size * initial_list
            
            in_neighbors = graph.get_in_neighbors(node)
            out_neighbors = graph.get_out_neighbors(node)
            for ineighbor in in_neighbors:
                if ineighbor in conditional_node_to_label_map.keys():
                    in_list[self.domain_labels.index(conditional_node_to_label_map[ineighbor])] += 1.0
            
            for oneighbor in out_neighbors:
                if oneighbor in conditional_node_to_label_map.keys():
                    out_list[self.domain_labels.index(conditional_node_to_label_map[oneighbor])] += 1.0
            
            result_list = in_list + out_list
            return result_list
        
        #undirected graph case
        else:
            neighbors_list = list_size * initial_list
            neigbors = graph.get_neighbors(node)
            for neighbor in neigbors:
                if neighbor in conditional_node_to_label_map.keys():
                    neighbors_list[self.domain_labels.index(conditional_node_to_label_map[neighbor])]+=1.0
            return neighbors_list
        
class ProportionalAggregator(Aggregator):
    '''The proportional aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map): 
        #first use the count aggregator then find the proportian
        count_aggregator = CountAggregator(self.domain_labels,self.directed)
        agg_res = count_aggregator.aggregate(graph,node,conditional_node_to_label_map)
        
        domain_size = len(self.domain_labels)
        
        #directed graph case
        if self.directed:
            in_sum = sum(agg_res[:domain_size])
            out_sum = sum(agg_res[domain_size:])
            if in_sum !=0:
                for i in range(0,domain_size):
                    agg_res[i] /= 1. * in_sum
            if out_sum !=0:
                for i in range(domain_size,-1):
                    agg_res[i] /= 1. * out_sum
            return agg_res

        #undirected graph case
        else:
            all_sum=sum(agg_res)
            if all_sum !=0:
                for i in range(len(agg_res)):
                    agg_res[i] /= 1. * all_sum
            return agg_res

class ExistAggregator(Aggregator):
    '''The exist aggregate'''
    
    def aggregate(self, graph, node, conditional_node_to_label_map): 
        #first use the count aggregator and then check if exist
        count_aggregator = CountAggregator(self.domain_labels,self.directed)
        agg_res = count_aggregator.aggregate(graph,node,conditional_node_to_label_map)
        
        for i in range(len(agg_res)):
            if agg_res[i]>= 1:
                agg_res[i]= 1
        return agg_res

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

class Classifier(object):
    '''
    The base classifier object
    '''

    def __init__(self, scikit_classifier_name, **classifier_args):        
        classifer_class=get_class(scikit_classifier_name)
        self.clf = classifer_class(**classifier_args)

    
    def fit(self, graph, train_indices):
        '''
        Create a scikit-learn classifier object and fit it using the Nodes of the Graph
        that are referenced in the train_indices
        '''
        abstract()
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        '''
        This function should be called only after the fit function is called.
        Predict the labels of test Nodes conditioning on the labels in conditional_node_to_label_map.
        '''
        abstract()

class LocalClassifier(Classifier):

    def fit(self, graph, train_indices):
        X=[graph.node_list[t].feature_vector for t in train_indices]
        y=[graph.node_list[t].label for t in train_indices]
        self.clf.fit(X,y)


    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        X=[graph.node_list[t].feature_vector for t in test_indices]
        X=np.array(X,dtype=float)
        return self.clf.predict(X)

class RelationalClassifier(Classifier):
    
    def __init__(self, scikit_classifier_name, aggregator, use_node_attributes = False, **classifier_args):
        super(RelationalClassifier, self).__init__(scikit_classifier_name, **classifier_args)
        self.aggregator = aggregator
        self.use_node_attributes = use_node_attributes
        
    def fit(self, graph, train_indices):
        conditional_map={}
        for i in train_indices:
            conditional_map[graph.node_list[i]]=graph.node_list[i].label
        X = []
        y = []
        for i in train_indices:
            agg_res = self.aggregator.aggregate(graph,graph.node_list[i],conditional_map)
            data = []
            if self.use_node_attributes:
                data = list(graph.node_list[i].feature_vector)
                data.extend(agg_res)
            else:
                data.extend(agg_res)
            X.append(data)
            y.append(graph.node_list[i].label)
        self.clf.fit(X,y)

    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        X=[]
        for i in test_indices:
            agg_res = self.aggregator.aggregate(graph,graph.node_list[i],conditional_node_to_label_map)
            data = []
            if self.use_node_attributes:
                data = list(graph.node_list[i].feature_vector)
                data.extend(agg_res)
            else:
                data.extend(agg_res)
            X.append(data)
        X = np.array(X,dtype=float)
        return self.clf.predict(X)

class ICA(Classifier):
    
    def __init__(self, local_classifier, relational_classifier, max_iteration = 10):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.max_iteration = 10
    
    
    def fit(self, graph, train_indices):
        self.local_classifier.fit(graph, train_indices)
        self.relational_classifier.fit(graph, train_indices)
    
    def predict(self, graph, test_indices, conditional_node_to_label_map = None):
        #1. You should use your local classifier to predict labels for each of the test item 
        local_classifier_result = self.local_classifier.predict(graph, test_indices)
        
        #and add their mapping to conditional_node_to_label_map.
        for i in range(len(local_classifier_result)):
            conditional_node_to_label_map[graph.node_list[test_indices[i]]] = local_classifier_result[i]
            
        #2. While max iteration is not eached
        for iteration in range(self.max_iteration):
            #For each test item:
            for ind in test_indices:
                #Predict its label using your relational classifier by calling the predict function of
                #your relational classifier. Pass the up to date conditional_node_to_label_map to
                #the predict function.
                rel_predict=self.relational_classifier.predict(graph,[ind],conditional_node_to_label_map)
                
                #Update conditional_node_to_label_map based on the latest label you received
                #for this test item one by one.                
                conditional_node_to_label_map[graph.node_list[ind]] = rel_predict[0]                
                
        result = []
        
        for ind in test_indices:
            result.append(conditional_node_to_label_map[graph.node_list[ind]])
        return result
    