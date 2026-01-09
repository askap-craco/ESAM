from pprint import PrettyPrinter
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

        
            
@njit(fastmath=True, cache=False)
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

@njit(cache=False)
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
        assert off <= 0, f'You can copy lower to dout only if offset is negative. Always true for dedispersion'
        dout[:] = lower[:] # just a copy
    elif lower is None and upper is not None:
        # TODO: make this better maybe with an explicit loop with numba
        if off <= 0:
            dout[:-off] = 0
            dout[-off:] = upper[:nt+off]
        elif off > 0:
            dout[:nt-off] = upper[ off:]
            dout[nt-off:] = 0
    return dout


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

@njit(fastmath=True, cache=False)
def convolve_with_kernel(din, kernel, out, squared_weights=False):
    '''
    Convolve the input data with the kernel
    Vivik's slow but easy to undertand code.
    Will replace with np.convolve    
    '''
    kernel_size = len(kernel)
    if squared_weights:
        weights = kernel**2
    else:
        weights = kernel
        
    nsamp = din.size - (kernel_size-1)
    for isamp in range(nsamp):        
        vsum = 0
        for ik in range(kernel_size):        
            vsum += din[isamp + ik] * weights[ik]

        out[isamp] = vsum        

    return out

class EndProduct:
    def __init__(self, kernel, similarity_score=None):
        self.kernel = kernel
        self.similarity_score = similarity_score
        self.__dout = None
        
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

    def __get_dout(self, din):
        '''
        Lazy allocate dout if it's not already allocated or if the nt is different
        '''
        nt = len(din)
        dout = getattr(self, '_EndProduct__dout', None) # need to do this because pickled tree may not have the attribute
        if dout is None or dout.shape != din.shape:
            dout = np.zeros_like(din)

        dout[:] = 0
        return dout

    def __call__(self, din, squared_weights = False):
        '''
        Implements the execution of the EndProduct on the data
        din: np.ndarray
            1-D array on which the Endproduct has to be executed
        '''
        out = self.__get_dout(din)
        convolve_with_kernel(din, self.kernel, out, squared_weights)

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


    def __call__(self, din, squared_weights = False, lower_chan=0, upper_chan=None, pad_with_zeros=True):
        '''
        Actualy compute ESAM of the given input data
        din: ndarray with shape (nchan, nt)
        squared_weights: bool, if True, then the weights are squared
        lower_chan: int, the channel number of the lower channel - channels below this will be ignored
        not, calculated and be essentially zero. These channel numbers are raw -i.e. at the bottom iteration
        or raw data level. 
        upper_chan: int, the channel number of the upper channel to include. Inclusive.
        pad_with_zeros: bool, if True, and the channel range has been trimmed, then the output will be as
        though the input were padded with zeros. If pad_with_zeros is false, it will return the smallest
        output data that included the specified channel range. 
        '''
        assert din.shape[0] == self.nchan

        if upper_chan is None:
            upper_chan = self.ichan + self.nchan 
        
        assert lower_chan <= upper_chan, f'lower_chan {lower_chan} must be less than upper_chan {upper_chan}'
        assert lower_chan >= 0, f'lower_chan {lower_chan} must be non-negative'
      
        # terminate recursion if requested processing range is outside this iteration's channel range
        out_of_range = upper_chan < self.start_chan or lower_chan > self.end_chan
        if out_of_range:
            return None # terminate recursion early if we're outside the channel range. Signals "no data here"
            
        if self.nchan == 1:
            assert din.shape[0] == 1, f'Expected 1 channel. Got {din.shape}'
             # get cached output data
            dout = self.__get_dout(din)
            for iprod, prod in enumerate(self._products):
                dout[iprod, :] = prod(din[0], squared_weights)   #din[0] because din is a 1-D data but has 2-D shape (nf, nt) where nf = 1 
  
        else:
            nf2 = self.nchan // 2 
            lower = self.lower(din[:nf2,...], squared_weights, lower_chan, upper_chan, pad_with_zeros)
            upper = self.upper(din[nf2:,...], squared_weights, lower_chan, upper_chan, pad_with_zeros)

            # check for early termination - in this case, we don't pad with zeros
            # we  terminate early if we've been asked to (i.e. pad_width_zero is False)
            # and either upper or lower has returned None. In that case, we return
            # the valid data.
            early_termination = (lower is None or upper is None) and (not pad_with_zeros)
            assert not (lower is None and upper is None), 'Invalid. Should have quit above'
            #print(f'{self.nchan} me={self.start_chan}:{self.end_chan} target={lower_chan}:{upper_chan} upper?{upper is None} lower? {lower is None} early?{early_termination}')
            

            if early_termination:
                dout = lower if upper is None else upper
            else:
                # get cached output data
                dout = self.__get_dout(din) 
                for iprod, prod in enumerate(self._products):
                    lower_dout = None if lower is None else lower[prod.pid_lower, :]    
                    upper_dout = None if upper is None else upper[prod.pid_upper, :]
                    sum_at_offset_or_copy(dout[iprod, :], lower_dout, upper_dout, prod.offset)

        return dout
            
    
def main():
    nchan = 256
    dedisperser = EsamTree(nchan)

    


        



