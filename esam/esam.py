from pprint import PrettyPrinter
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def siblings(start_node):
    '''
    Go accross the tree at the same level from lower to upper
    '''
    node = start_node
    while True:
        yield node
        if node.parent.lower == node:
            node = node.parent.upper
        elif node.parent.upper == node and node.parent is None:
            break
        else:
            node = node.parent.parent.upper.lower

        
            
@njit(fastmath=True)
def sum_at_offset(dout, lower, upper, off):
    '''
    Sum lower and upper 1D arrays at a given offset in samples and return in dout.
    Offset is negative for dedispersion.
    For offset <= 0, the first off samples is lower only
    The remainng samples are lower + upper.

    if offset > 0, teh first nt-off samples are lower + upper
    The remaining off samples are upper only
    '''
    dout[:] = lower
    nt = len(dout)
    if off <= 0: # this is the usual sign for the dedispersion of an FRB
        dout[-off:] += upper[:nt+off]
    elif off > 0:
        dout[:nt-off] += upper[ off:]
        
    return dout

@njit
def sum_at_offset_or_copy(dout, lower, upper, off):
    '''
    Handles None values for lower and upper without doing summing
    TODO: Need to think about zeroing the memory
    maybe zeroing is quick or maybe it's slow.
    '''
    nt = len(dout)
    if lower is not None and upper is not None:
        sum_at_offset(dout, lower, upper, off)
    elif lower is not None and upper is None:
        assert off <= 0, f'This assumption is only correct if offset is ngative. Always tru for dedispersion'
        dout[:] = lower[:] # just a copy
    elif lower is None and upper is not None:
        # TODO: make this better
        if off <= 0:
            dout[:-off] = 0
            dout[-off:] = upper[:nt+off]
        elif off > 0:
            dout[:nt-off] = upper[ off:]
            dout[nt-off:] = 0
    return dout

def sum_at_offset_or_copy(dout, lower, upper, off):
    nt = len(dout)
    if lower is None:
        lower = upper*0

    if upper is None:
        upper = lower*0

    return sum_at_offset(dout, lower, upper, off)

class IterProduct:
    def __init__(self, pid_lower, pid_upper, offset):
        self.pid_lower = pid_lower
        self.pid_upper = pid_upper
        self.offset = offset

    def __str__(self):
        s = f"IterProduct with pid_lower={self.pid_lower}, pid_upper={self.pid_upper}, offset={self.offset}"
        return s

    def __repr__(self):
        r = f"l{self.pid_lower}_u{self.pid_upper}_o{self.offset}"
        return r

    def __eq__(self, other):
        if isinstance(other, IterProduct):
            iseq = self.offset == other.offset and\
                    self.pid_lower == other.pid_lower and\
                    self.pid_upper == other.pid_upper
            return iseq

        return False

    def __call__(self, din):
        '''
        Implements the execution of the iteration product on a dataset
        '''
        off = self.offset

class EndProduct:
    def __init__(self, kernel, similarity_score=None):
        self.kernel = kernel
        self.similarity_score = similarity_score
        
    def __str__(self):
        s = f'EndProduct kernel={self.kernel}'
        return s
    
    def __eq__(self, other):
        '''
        Need this otherwise we keep adding the same kernel at the bottom level
        '''    
        iseq = np.array_equal(self.kernel, other.kernel)
        
        if self.similarity_score is not None:
            conv = np.convolve(self.kernel, other.kernel)
            sq_weights = np.sqrt(np.sum(self.kernel**2))
            matched_filter_snr = np.max(conv) / sq_weights
            expected_snr = np.sqrt(np.sum(other.kernel**2))
            if matched_filter_snr / expected_snr > self.similarity_score:
                iseq = True

        return iseq
    
    __repr__ = __str__

    def __call__(self, din, squared_weights = False):
        '''
        Implements the execution of the EndProduct on the data
        din: np.ndarray
            1-D array on which the Endproduct has to be executed
        '''
        out = np.zeros_like(din)
        kernel_size = len(self.kernel)
        for isamp in range(din.size):
            if isamp + kernel_size-1 == din.size:
                break
            if squared_weights:
                weights = self.kernel**2
            else:
                weights = self.kernel
            out[isamp] = np.sum(din[isamp : isamp + kernel_size] * weights)

        #out =  np.convolve(din, self.kernel, mode='same')
        '''
        plt.figure()
        plt.plot(out)
        plt.show()
        '''
        return out 


def sum_offsets(trace):
    #print(f"sum_offsets got trace {trace}")
    osum = 0
    nchan = len(trace)
    for ichan in range(nchan):
        #print(f"chan in trace is {trace[ichan]}")
        if ichan == 0:
            #If this the 0th channel in the trace, then it's offset wouldn't have been adjusted yet
            pass
        else:
            osum += trace[ichan][0]
    #print(f"Returning osum = {osum}")
    return osum


class EsamTree:
    def __init__(self, nchan, ichan = 0, similarity_score = 0.9, parent=None):
        self._products = [] # list containing Products
        self.nchan = nchan # number of channels this iteration is responsible for.
        self._ichan = ichan
        #self.similarity_score = 0.9
        self.similarity_score = None
        assert self.nchan == 1 or self.nchan % 2 == 0, f"EsamTree can only have even number of channels, given = {nchan}"
        
        if nchan == 1:
            self.upper = None
            self.lower = None
        else:
            # normally we'd do FdmtDescriptor here, butif we make a new __class_ if we have a subclass
            # then it'll create the right class
            self.upper = self.__class__(nchan // 2, 2*ichan+1, parent=self)
            self.lower = self.__class__(nchan // 2, 2*ichan, parent=self)
        
        # parent is None for the root node
        self.parent = parent

        # We're going to lazy allocate the output data because we don't know nt
        self.__dout = None

    def __str__(self):
        s = f'Nchan={self.nchan} chan={self._ichan} nprod={self.ndm}'
        return s
    
    __repr__ = __str__

    @property
    def ichan(self):
        return self._ichan

    @property
    def start_chan(self):
        '''
        Channel number of the first channel at the finest resolution
        '''
        c = self.ichan * self.nchan
        return c

    @property
    def end_chan(self):
        '''
        Channel number of the last channel at the finest resolution. Inclusive.
        '''
        c = self.start_chan + self.nchan - 1
        return c

    @property
    def ndm(self):
        return len(self._products)
    
    @property
    def nprod(self):
        return self.ndm
    
    @property
    def total_products(self):
        '''
        Returns total number of products in hierarchy
        '''
        if self.nchan == 1:
            return self.nprod
        else:
            return self.upper.total_products + self.lower.total_products + self.nprod
        
    def descriptor_tree(self, tree=None, level=0):
        '''
        Returns a list. Each element contains another list. That list contains all descriptors for that iteration
        '''
        if tree is None:
            tree = []
        if self._ichan == 0:
            tree.append([])
            
        tree[level].append(self)
        if self.nchan != 1:
            self.lower.descriptor_tree(tree, level+1)
            self.upper.descriptor_tree(tree, level+1)
        
        return tree              
    
    def get_all_pids(self, all_pids = None):
        '''
        Makes a list of lists containing all PIDs for each iteration
        Modifies the provided 'all_pids' list in place
        '''
        if all_pids is None:
            all_pids = [[] for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        if self.nchan == 1:
            all_pids[list_idx].extend(self._products)
        else:
            all_pids[list_idx].extend(self._products)
            self.upper.get_all_pids(all_pids)
            self.lower.get_all_pids(all_pids)

        return all_pids


    def count_all_operations(self, op_counts = None):
        '''
        Counts the number of operations in each iteration and saves them in a list
        '''

        if op_counts is None:
            op_counts = [0 for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        #print(f"{pid_counts}, {type(pid_counts)}, {pid_counts[list_idx]}, {type(pid_counts[list_idx])}")
        if list_idx == 0:
            #Means we are the lowest level - endnodes
            #Then we should not just count the number of products, but how many sums it will do during the convolution phase as well in each product
            for iprod in self._products:
                op_counts[list_idx] += iprod.kernel.size
        else:
            op_counts[list_idx] += len(self._products)

        if self.nchan > 1:
            self.lower.count_all_operations(op_counts)
            self.upper.count_all_operations(op_counts)

        return op_counts


    def count_all_pids(self, pid_counts = None):
        '''
        Counts the number of products in each iteration and saves them in a list
        '''

        if pid_counts is None:
            pid_counts = [0 for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        #print(f"{pid_counts}, {type(pid_counts)}, {pid_counts[list_idx]}, {type(pid_counts[list_idx])}")
        pid_counts[list_idx] += len(self._products)
        
        if self.nchan > 1:
            self.lower.count_all_pids(pid_counts)
            self.upper.count_all_pids(pid_counts)

        return pid_counts

    def max_npid_by_iteration(self, pid_counts = None):
        '''
        Counts the number of products in each iteration and saves them in a list
        '''

        if pid_counts is None:
            pid_counts = [0 for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))

        #print(f"{pid_counts}, {type(pid_counts)}, {pid_counts[list_idx]}, {type(pid_counts[list_idx])}")
        pid_counts[list_idx] = max(pid_counts[list_idx], len(self._products))
        
        if self.nchan > 1:
            self.lower.max_npid_by_iteration(pid_counts)
            self.upper.max_npid_by_iteration(pid_counts)

        return pid_counts

    def npid_by_iteration(self, pid_counts = None):
        '''
        Counts the number of products in each iteration and saves them in a list
        '''

        if pid_counts is None:
            pid_counts = [[] for i in range(int(np.log2(self.nchan)) + 1) ]

        list_idx = int(np.log2(self.nchan))
        pid_counts[list_idx].append(len(self._products))
        
        if self.nchan > 1:
            self.lower.npid_by_iteration(pid_counts)
            self.upper.npid_by_iteration(pid_counts)

        return pid_counts

    def get_trace_pid(self, trace) -> int:
        '''
        Returns the product ID for the given trace
        trace:list of nchan values. Each value is tuple(width, offset)'
        '''
        assert len(trace) == self.nchan, f'Unexpedcted trace length in {self}. Was {len(trace)} expected {self.nchan}'
        n2 = self.nchan // 2
        if self.nchan == 1:
            #print(f"Got trace as {trace}, giving trace[0] = {trace[0][1]} to EndProduct")
            prod = EndProduct(trace[0][1], self.similarity_score)
            mid_offset = None
            offsets_added_so_far = None
            #cum_offset = trace[0][0]
        else:
            pid_lower = self.lower.get_trace_pid(trace[:n2])
            pid_upper = self.upper.get_trace_pid(trace[n2:])
            assert n2 >= 0
            #pid_lower, offset_lower = self.lower.get_trace_pid(trace[:n2])
            #pid_upper, offset_upper = self.upper.get_trace_pid(trace[n2:])
            #assert n2 >= 0
            #cum_offset = offset_lower + offset_upper
            #'''
            mid_offset, _ = trace[n2] # offset and width of the lower of the 2 middle channels
            offsets_added_so_far = sum_offsets(trace[:n2])

            offset = mid_offset + offsets_added_so_far

            assert type(mid_offset) == int, f'offset has wrong type {type(mid_offset)} {mid_offset}'
            #if hasattr(self.lower._products[pid_lower], 'offset'):
            #    offset = mid_offset + self.lower._products[pid_lower].offset
            #else:
            #    offset = mid_offset + 0 
            #'''
            #prod = IterProduct(pid_upper, pid_lower, offset_lower)
            prod = IterProduct(pid_lower, pid_upper, offset)
        
        #print(f"self.nchan = {self.nchan}, self._ichan = {self._ichan}, trace = {trace}, mid_offset = {mid_offset}, offsets_added_so_far={offsets_added_so_far}") 
        added = False
        if prod not in self._products:
            self._products.append(prod)
            added = True
            
        pid = self._products.index(prod)
        #print(f'{self} of trace {trace}={prod}=PID{pid} added?={added}')
        
        return pid#, cum_offset
        
     
    def __get_dout(self, din):
        '''
        Lazy allocate dout if it's not already allocated or if the nt is different
        '''
        nt = din.shape[1]
        dout = getattr(self, '_EsamTree__dout', None) # need to do this because pickled tree may not have the attribute
        if dout is None or dout.shape[1] != nt:
            self.__dout = np.zeros((self.nprod, nt)) # NT here is a bit tricky
            dout = self.__dout

        dout[:] = 0
        return dout


    def __call__(self, din, squared_weights = False, lower_chan=0, upper_chan=None):
        '''
        Actualy compute ESAM of the given input data
        din: ndarray with shape (nchan, nt)
        squared_weights: bool, if True, then the weights are squared
        lower_chan: int, the channel number of the lower channel - channels below this will be ignored
        not, calculated and be essentially zero. These channel numbers are raw -i.e. at the bottom iteration
        or raw data level. 
        upper_chan: int, the channel number of the upper channel to include. Inclusive.
        '''
        assert din.shape[0] == self.nchan

        if upper_chan is None:
            upper_chan = self.ichan + self.nchan 
        
        assert lower_chan <= upper_chan, f'lower_chan {lower_chan} must be less than upper_chan {upper_chan}'
        assert lower_chan >= 0, f'lower_chan {lower_chan} must be non-negative'
      
        # terminate recursion if we're outside the channel range
        if not (self.start_chan >= lower_chan and self.end_chan <= upper_chan):
            return None # terminate recursion early if we're outside the channel range. Signals "no data here"
            
        # get cached output data
        dout = self.__get_dout(din)

        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'
            
            for iprod, prod in enumerate(self._products):
                dout[iprod, :] = prod(din[0], squared_weights)   #din[0] because din is a 1-D data but has 2-D shape (nf, nt) where nf = 1 
  
        else:
            nf2 = self.nchan // 2 
            lower = self.lower(din[:nf2,...], squared_weights, lower_chan, upper_chan)
            upper = self.upper(din[nf2:,...], squared_weights, lower_chan, upper_chan)
            for iprod, prod in enumerate(self._products):               
                off = prod.offset
                sum_at_offset(dout[iprod, :], lower[prod.pid_lower, :], upper[prod.pid_upper, :], off)
        return dout
            
    
def main():
    nchan = 256
    dedisperser = EsamTree(nchan)

    



class PartitionEsamTree:
    def __init__(self, tree:EsamTree, chan_starts, parent=None):
        '''
        Mirrors the given ESAM tree but adds functionality to partition the results according 
        to the chan_starts array.
        :tree: origianl ESAM tree
        chan_starts: np.array of ints. start channel is inclusive. If it doesn't start with zero, we'll prepend zer
        Returns a list of arrays, each of which is the result of the ESAM tree on the data for the corresponding channel range.
        ''' 
        self.tree = tree
        if len(chan_starts) == 0:
            chan_starts = np.array([tree.start_chan])

        if chan_starts[0] != tree.start_chan:
            chan_starts = np.concatenate([[tree.start_chan], chan_starts])

        if len(chan_starts) == 1:
            chan_ends = chan_starts + tree.nchan  # python index. End channel is exclusive
        else:
            chan_ends = chan_starts[1:] # python indexing - end channel is exclusive
            chan_ends = np.concatenate([chan_ends, [tree.end_chan]]) # last channel is exclusive

        partition_nchans = chan_ends - chan_starts + 1
        assert len(chan_ends) == len(chan_starts), f'Channel ends and starts must have the same length, got {len(chan_ends)} and {len(chan_starts)}'
        assert len(partition_nchans) == len(chan_starts), f'Partition nchans and starts must have the same length, got {len(partition_nchans)} and {len(chan_starts)}'


        assert chan_starts[0] == tree.start_chan, f'First channel start must be 0, got {chan_starts[0]}'
        assert chan_starts[-1] <= tree.end_chan, f'Last channel start must be less than {tree.end_chan}, got {chan_starts[-1]}'
        assert np.all(partition_nchans > 0), f'Channel starts must be in increasing order, got {chan_starts} and partition_nchans {partition_nchans}'

        max_nch = max(partition_nchans)
        top_iteration = int(np.ceil(np.log2(max_nch)))

        self.top_iteration = top_iteration
        self.partition_nchans = partition_nchans
        self.max_nch = max_nch
        self.top_iteration = top_iteration
        self.chan_starts = chan_starts
        self.chan_ends = chan_ends
        self.parent = parent

        # parition chan_starts and chan_ends for the upper and lower branches
        
        mid_chan = self.tree.start_chan + self.tree.nchan // 2
        lower_chan_starts = []
        upper_chans_starts = []
        
        for chan in chan_starts:
            if chan < mid_chan:                                
                lower_chan_starts.append(chan)
            else:
                upper_chans_starts.append(chan)

        if tree.nchan == 1:
            self.upper = None
            self.lower = None
        else:
            self.upper = self.__class__(tree.upper, np.array(upper_chans_starts), parent=self)
            self.lower = self.__class__(tree.lower, np.array(lower_chan_starts), parent=self)

    @property
    def nchan(self) -> int:
        return self.tree.nchan

    @property
    def nprod(self):
        return self.tree.nprod
    def process_recursively(self, din, squared_weights = False):
        '''
        Evaluates the ESAM tree on the given input data, but partitions the results according 
        to the channel starts.
        Returns list of tuples (partition_index, start_chan, end_chan, dout)
        where partition_index is the index of the partition, start_chan is the start channel of the partition,
        end_chan is the end channel of the partition, and dout is the output data for the partition.

        '''
        assert din.shape[0] == self.nchan
        nt = din.shape[1]        

        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'            
            # only compute stuff if our channel number is between the lower and upper channels
            # calculate all products.
            # There can be only one product for nchan = 1, so don't bother to iterate over partitions
            dout = self.tree(din, squared_weights)
            ichan = self.tree.ichan          
            partition_index = np.where(self.chan_starts <= ichan <= self.chan_ends)[0][0]
            start_chan = ichan
            end_chan = ichan+1 # exclusive. python indexing rules.
            # For nchan = 1 you can only have 1 partition. But otherwise might be different.
            full_data = [(partition_index, start_chan, end_chan, dout)]

        else:
            nf2 = self.nchan // 2 
            mid_chan = self.tree.start_chan + self.tree.nchan // 2
            lower = self.lower.process_recursively(din[:nf2,...], squared_weights)
            upper = self.upper.process_recursively(din[nf2:,...], squared_weights)
            full_data = []
            for ipart, (start_chan, end_chan) in enumerate(zip(self.chan_starts, self.chan_ends)):
                dout = np.zeros((self.nprod, nt)) # NT here is a bit tricky
                assert start_chan < mid_chan
                assert end_chan >= mid_chan
                expecteddout = self.tree(din, squared_weights)
                expected_upper = self.tree.upper(din[nf2:,...], squared_weights)
                expected_lower = self.tree.lower(din[:nf2,...], squared_weights)
                

                for iprod, prod in enumerate(self.tree._products):
                    if end_chan < mid_chan: # everything below mid_chan is in lower
                        # need to find corresponding lower data from the lower set
                        lower_data = list(filter(lambda x: x[1] == start_chan and x[2] == end_chan, lower))
                        assert len(lower_data) == 1, f'Expected 1 lower data, got {len(lower_data)}'
                        lower_dout = lower_data[0][3]
                        sum_at_offset_or_copy(dout[iprod, :], lower_dout[prod.pid_lower, :], None, prod.offset)
                    elif start_chan >= mid_chan:
                        # need to find corresponding upper data from the upper set
                        upper_data = list(filter(lambda x: x[1] == start_chan and x[2] == end_chan, upper))
                        assert len(upper_data) == 1, f'Expected 1 upper data, got {len(upper_data)}'
                        upper_dout = upper_data[0][3]
                        # need to apply the appropriate delay to the upper data. Using sum at offset, but could equally just offset
                        # or save the offset with some extra metadata
                        sum_at_offset_or_copy(dout[iprod, :], None, upper_dout[prod.pid_upper, :], prod.offset)
                    else:
                        assert start_chan < mid_chan and end_chan >= mid_chan
                        lower_data = list(filter(lambda x: x[1] == start_chan and x[2] == mid_chan, lower))
                        assert len(lower_data) == 1, f'Expected 1 lower data, got {len(lower_data)}'
                        lower_dout = lower_data[0][3]
                        upper_data = list(filter(lambda x: x[1] == mid_chan and x[2] == end_chan, upper))
                        assert len(upper_data) == 1, f'Expected 1 upper data, got {len(upper_data)}'
                        upper_dout = upper_data[0][3]
                        sum_at_offset(dout[iprod, :], lower_dout[prod.pid_lower, :], upper_dout[prod.pid_upper, :], prod.offset)
                        np.testing.assert_allclose(lower_dout, expected_lower)
                        np.testing.assert_allclose(upper_dout, expected_upper)


                np.testing.assert_allclose(dout, expecteddout)
                full_data.append((ipart, start_chan, end_chan, dout))

        return full_data

    __call__ = process_recursively

    def process_iteratively(self, din, squared_weights = False):
        '''
        Evaluates the ESAM tree on the given input data, but partitions the results according 
        to the channel starts.

        Also instead of doing it recusively, we're going to do it iteratively, bottom up.
        We might also use pingpong buffers, but we can maybe just do iteration buffers for now.

        :tree: origianl ESAM tree
        :din: ndarray with shape (nchan, nt)
        :squared_weights: bool, if True, then the weights are squared
        '''
        assert din.shape[0] == self.tree.nchan
        nt = din.shape[1]
        #dout = np.zeros((self.tree.nprod, nt)) # NT here is a bit tricky
        dout = []

        # first, find the bottom node in the bottom iteration
        node = self.tree
        while node.lower is not None:
            node = node.lower

        # create initial input buffer - lists rather than arras
        din_buffers = [din[i,...] for i in range(din.shape[0])]
        while node.parent is not None:
            din_buffers = self.process_all_nodes_for_iteration(node, din_buffers, squared_weights)
            node = node.parent
        
        dout = din_buffers

        return dout

    def process_all_nodes_for_iteration(self, bottom_node, din, squared_weights):
        '''
        Processes all nodes for the given iteration
        '''
        raise NotImplementedError('This method is not implemented')
        nt = din.shape[1]
        assert node.ichan == 0, 'Node must be at the bottom of the tree'
        
        dout = [] # different number of products per node
        partition_info = []
        for inode, node in enumerate(simblings(bottom_node)):
            if node.nchan == 1:
                node_dout = np.zeros((node.nprod, nt))
                dout.append(node_dout)
                ichan = inode
                for iprod, prod in enumerate(node._products):
                    node_dout[iprod, :] = prod(din[ichan], squared_weights)   #din[0
            else:
                lower = self.lower(din[:nf2,...], squared_weights)
                upper = self.upper(din[nf2:,...], squared_weights)
                for iprod, prod in enumerate(node._products):               
                    off = prod.offset
                    sum_at_offset(node_dout[iprod, :], lower[prod.pid_lower, :], upper[prod.pid_upper, :], off)

        
        



