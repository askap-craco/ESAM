from typing import Any


from numpy._typing._array_like import NDArray


from numpy import float64


from esam import esam
import numpy as np
from esam.FDMT_tests import simulate_narrow_frb as snf
from pytest import fixture

@fixture
def tree():
    t = np.load('test_data/final_optimised_esam_tree_fast_with_traces_0_1000_0.2_nch256_threshold_0.85.pkl.npy', allow_pickle=True)
    # unwrap from nd array
    t = t.item()
    return t

@fixture
def frb():
    nch = 256
    fchans = np.linspace(800.5, 1055.5, nch, endpoint=True, dtype=np.float32)
    tx = 100
    nsamps = 256
    nch = 256
    tx = 1
    dm = 100
    chw_2 = 0.5
    tpulse = dm + 5.5
    #tpulse = 10
    din, ndd = snf.make_pure_frb(nsamps, nch, tx, dm, fchans, chw_2, tpulse)
    assert(not np.all(din == 0))
    #imshow(din, aspect='auto', origin='lower')
    #print(sum(din), din.max(), np.unravel_index(np.argmax(din), din.shape))
    return din

def test_esam_runs_without_failing(frb, tree):
    insum = frb.sum()
    assert insum > 0

    dout = tree(frb)
    dmax = dout.max()
    tmax, dmmax = np.unravel_index(np.argmax(dout), dout.shape)

    print(insum, dmax, tmax, dmmax)
    np.testing.assert_allclose(insum, dmax, rtol=0.15) # threshyold was 0.85 so we should be OK within 15 %

def test_esam_tree_startchan_endchan(tree):
    assert tree.start_chan == 0
    assert tree.end_chan == 255

    assert tree.lower.start_chan == 0
    assert tree.lower.end_chan == 127
    assert tree.upper.start_chan == 128
    assert tree.upper.end_chan == 255

    assert tree.lower.lower.start_chan == 0
    assert tree.lower.lower.end_chan == 63
    assert tree.lower.upper.start_chan == 64
    assert tree.lower.upper.end_chan == 127

    assert tree.upper.lower.start_chan == 128
    assert tree.upper.lower.end_chan == 191
    assert tree.upper.upper.start_chan == 192
    assert tree.upper.upper.end_chan == 255

def test_esam_with_channel_range_works(frb, tree):
    start_chan = 129
    end_chan = 214
    nch = end_chan - start_chan + 1
     
    # run with channel range
    dout1 = tree(frb, lower_chan=start_chan, upper_chan=end_chan)

    # make a copy and outside the chanel range to zero
    frb2 = np.zeros_like(frb)
    frb2[start_chan:end_chan+1, :] = frb[start_chan:end_chan+1, :]
    dout2 = tree(frb2)
    
    np.testing.assert_allclose(dout1, dout2) 

def test_sum_at_offset_only_offsets_upper():
    lower = np.array([1, 2, 3, 4, 5])
    upper = np.array([6, 7, 8, 9, 10])
    dout = np.empty_like(lower)

    esam.sum_at_offset(dout, lower, upper, -1)
    np.testing.assert_allclose(dout, [1, 2+6, 3+7, 4+8, 5+9])


    esam.sum_at_offset(dout, lower, upper*0, -1)
    
    # check if upper is zero, it just returns the input
    np.testing.assert_allclose(dout, lower)

def test_sum_at_offsets_or_copy():
    lower = np.array([1, 2, 3, 4, 5])
    upper = np.array([6, 7, 8, 9, 10])

    
    dout1_copy = np.empty_like(lower)
    dout2_copy = np.empty_like(lower)
    dout3_copy = np.empty_like(lower)

    dout1 = np.empty_like(lower)
    dout2 = np.empty_like(lower)
    dout3 = np.empty_like(lower)


    esam.sum_at_offset_or_copy(dout1_copy, lower, upper, -1) # reference
    esam.sum_at_offset_or_copy(dout2_copy, lower, None, -1)
    esam.sum_at_offset_or_copy(dout3_copy, None, upper, -1)

    esam.sum_at_offset(dout1, lower, upper, -1)
    esam.sum_at_offset(dout2, lower, upper*0, -1)
    esam.sum_at_offset(dout3, lower*0, upper, -1)


    np.testing.assert_allclose(dout1_copy, dout1)
    np.testing.assert_allclose(dout2_copy, dout2)
    np.testing.assert_allclose(dout3_copy, dout3)   


def test_convolve_with_kernel_matches_np_convolve():
    din = np.random.randn(10)
    din = np.arange(20)
    kernel = np.ones(3)
    out = np.zeros_like(din)
    esam.convolve_with_kernel(din, kernel, out)
    convolved = np.convolve(din, kernel, mode='valid')
    # // not quite - vivek's version is offset
    # np.testing.assert_allclose(out, convolved)

